from pprint import pp

from ..model_converter import __Q4NX_Converter
from .. import imatrix_prune as imx
from ..constants import ModelArch, ModelArchNames, QWEN35_VARIANT_DIMS
from ..gguf_tensor import GGUFTensor
from gguf import GGUFReader, dequantize, quantize, GGMLQuantizationType
from safetensors.torch import save_file
from einops import rearrange, repeat
import numpy as np
import torch
import json

# Tensors whose last dimension is the hidden size (hidden as INPUT):
# zero-padding appends inert columns. --pad-to-fit uses these so a finetune
# with an off-nominal width fits an engine variant compiled for the nearest
# official dim (downstream matmuls have zero columns there).
PAD_HIDDEN_INPUT_SUFFIXES = (
    "token_embd.weight",
    "output.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ffn_up.weight",
    "ffn_gate.weight",
)

# Tensors whose FIRST dimension is the hidden size (hidden as OUTPUT):
# attention/mlp output projections feed a fixed-width engine residual stream.
PAD_HIDDEN_OUTPUT_SUFFIXES = (
    "attn_output.weight",
    "ssm_out.weight",
    "ffn_down.weight",
)

# 1-D RMS norm weights sized to hidden.
PAD_HIDDEN_NORM_SUFFIXES = (
    "attn_norm.weight",
    "post_attention_norm.weight",
)


def pad_hidden_axis(w: torch.Tensor, actual: int, target: int) -> torch.Tensor:
    """Zero-pad the hidden axis of a weight from actual to target.

    2-D tensors: pad columns (hidden-as-input convention). Use transpose via
    pad_hidden_axis_t for output projections whose rows are hidden. 1-D norms:
    pad length. Returns w unchanged when no axis matches or target <= actual.
    """
    if target <= actual:
        return w
    if w.dim() == 2 and w.shape[1] == actual:
        return torch.cat([w, w.new_zeros((w.shape[0], target - actual))], dim=1)
    if w.dim() == 1 and w.shape[0] == actual:
        return torch.cat([w, w.new_zeros(target - actual)])
    return w


def pad_hidden_axis_t(w: torch.Tensor, actual: int, target: int) -> torch.Tensor:
    """Zero-pad ROWS (hidden-as-output convention) from actual to target."""
    if target <= actual:
        return w
    if w.dim() == 2 and w.shape[0] == actual:
        return torch.cat([w, w.new_zeros((target - actual, w.shape[1]))])
    return w


def v_untile(t: torch.Tensor, grp: int, unit: int, axis: int = 0) -> torch.Tensor:
    """llama.cpp's tiled value-head order -> HF's grouped one, along rows (axis 0) or columns.

    llama.cpp's converter stores the gated DeltaNet's value heads TILED so ggml can broadcast
    the key heads: position r * num_k + kh holds HF's head kh * grp + r, grp = num_v // num_k.
    The engine reads HF's grouped order. `unit` is one head's extent along the axis -- its
    rows in a weight, its columns in an unpacked block array, 1 for a per-head vector.
    grp is 2 for the 4B / 9B and 3 for the 27B; it was hard-coded to 2."""
    if axis == 0:
        return rearrange(t, '(q g p) ... -> (g q p) ...', q=grp, p=unit).contiguous()
    return rearrange(t, 'r (q g p) -> r (g q p)', q=grp, p=unit).contiguous()


def untile_qkv(t: torch.Tensor, qk_rows: int, grp: int, head_v: int) -> torch.Tensor:
    """qkv's rows (or conv1d's channels): q and k first, `qk_rows` = 2 * num_k * head_k of
    them in their own order, then the value rows, tiled. The split is the tensor's half only
    when v is exactly as wide as q | k (32 value heads over 16 key heads)."""
    return torch.cat([t[:qk_rows], v_untile(t[qk_rows:], grp, head_v)]).contiguous()

class Qwen35(__Q4NX_Converter, model_arch=ModelArch.QWEN35_4B):
    pad_to_fit = False  # --pad-to-fit: zero-pad the hidden axis to the variant dim
    # --prune-ffn K: narrow the dense FFN to K intermediate neurons, chosen per
    # layer by imatrix importance. None = pack the FFN at its trained width.
    prune_ffn = None
    imatrix_path_hint = None   # --imatrix, or a sidecar found next to the GGUF
    imatrix = None        # imx.Imatrix, resolved once per convert
    _mtp_cache = None     # block indices that are MTP scaffolding

    @property
    def mtp_dropped(self) -> int:
        """How many speculative blocks this pack omits. Not prune-dependent: the
        runtime has no speculative decoding, so the MTP block's weights are dead
        for a pruned and an unpruned pack alike (b3eccb5 left the unpruned case
        open; the block is dropped either way)."""
        if self.gguf_reader is None:
            return 0
        return len(self._mtp_blocks())
    _ffn_keep = {}        # layer -> sorted index array, the ONE set per layer

    def __init__(self, source, config_json_path=None):
        variant = ModelArchNames.get(self.model_arch, str(self.model_arch))
        print(f"[INFO] Using Qwen35 converter (variant: {variant})")
        self.gguf_reader = None
        self.gguf_tensors = []
        self.hf_source = None
        self.hf_dir = None
        self.weight_map = {}
        self.hf_shards = {}
        if isinstance(source, GGUFReader):
            self.gguf_reader = source
            self.gguf_tensors = {t.name: t for t in source.tensors}
            self.initialize()
        else:
            self.hf_source = source
            self.hf_dir = self._resolve_source(source)
            self.initialize(config_json_path=config_json_path)

    def initialize(self, config_json_path=None):
        self._load_config(config_file_path=config_json_path)
        if self.gguf_reader is not None:
            self._read_gguf_tensors()
            self._read_gguf_metadata()
        else:
            self._read_hf_index()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        if self.gguf_reader is not None:
            self._convert_gguf(q4nx_path, weights_type)
        else:
            self._convert_hf(q4nx_path, weights_type)

    def _resolve_pad_target(self):
        """--pad-to-fit setup: hidden actual vs the variant's official dim.

        Only upward padding is supported: the engine variant expects a fixed
        width, and zero columns are inert in every downstream matmul. A width
        larger than any variant cannot be shrunk.
        """
        self._pad_hidden = None
        self._pad_actual = None
        if not getattr(self, "pad_to_fit", False):
            return
        if self.gguf_reader is None:
            print("[WARN] --pad-to-fit currently applies to GGUF sources only")
            return
        field = self.gguf_reader.fields.get("qwen35.embedding_length")
        if field is None:
            print("[WARN] --pad-to-fit: GGUF has no qwen35.embedding_length; padding disabled")
            return
        actual = int(field.contents())
        target = QWEN35_VARIANT_DIMS.get(self.model_arch)
        if target is None:
            return
        if target < actual:
            print(f"[WARN] --pad-to-fit cannot shrink hidden {actual} -> {target} "
                  f"({self.model_arch.name}); padding disabled")
            return
        if target == actual:
            print(f"[INFO] --pad-to-fit: hidden {actual} already matches "
                  f"{self.model_arch.name}; nothing to pad")
            return
        self._pad_actual = actual
        self._pad_hidden = target
        print(f"[INFO] --pad-to-fit: zero-padding hidden axis {actual} -> {target} "
              f"({self.model_arch.name})")

    def _maybe_pad(self, gguf_tensor, unpacked, target_dtype):
        """Re-quantize a hidden-axis tensor after zero-padding it to fit."""
        if self._pad_hidden is None:
            return unpacked
        name = gguf_tensor.name
        if name.endswith(PAD_HIDDEN_OUTPUT_SUFFIXES):
            pad_fn = pad_hidden_axis_t  # hidden is the output (row) axis
        elif name.endswith(PAD_HIDDEN_INPUT_SUFFIXES) or name.endswith(PAD_HIDDEN_NORM_SUFFIXES):
            pad_fn = pad_hidden_axis  # hidden is the input (column) axis / 1-D norm
        else:
            return unpacked
        w = torch.from_numpy(gguf_tensor.dequantize())
        padded = pad_fn(w, self._pad_actual, self._pad_hidden)
        if padded is w:
            return unpacked
        np_w = np.ascontiguousarray(padded.to(torch.float32).numpy())
        if target_dtype == GGMLQuantizationType.Q4_1:
            q = quantize(np_w, target_dtype)
            d, m, qw = GGUFTensor.unpack_q4_1(q, np_w.shape[1])
            return (d, m, qw)
        if target_dtype == GGMLQuantizationType.Q8_0:
            q = quantize(np_w, target_dtype)
            d, m, qw = GGUFTensor.unpack_q8_0(q, np_w.shape[1])
            return (d, m, qw)
        if len(unpacked) == 1:
            # Unquantized (F32/BF16) single-block tensors: store the padded array.
            return (np_w.astype(unpacked[0].dtype),)
        print(f"[WARN] --pad-to-fit: unsupported dtype {target_dtype.name} for {name}; left unpadded")
        return unpacked

    def _linear_heads(self):
        """(num_k, num_v, head_k, head_v) of the gated DeltaNet, from the GGUF's own ssm keys
        (llama.cpp writes group_count = key heads, time_step_rank = value heads, state_size =
        the key head dim, inner_size = value heads x the value head dim)."""
        f = self.gguf_reader.fields

        def get(key):
            if key not in f:
                raise KeyError(f"GGUF lacks {key}: cannot place the DeltaNet value heads")
            return int(f[key].contents())

        num_k, num_v = get("qwen35.ssm.group_count"), get("qwen35.ssm.time_step_rank")
        return num_k, num_v, get("qwen35.ssm.state_size"), get("qwen35.ssm.inner_size") // num_v

    def _resolve_prune(self):
        """--prune-ffn setup: one index set per layer, from the imatrix.

        Fails LOUDLY rather than packing full width. A wide FFN does not fit a
        core's L1, so the export dies in aiecc minutes later with a message that
        names memory rather than the real cause; and a silently-unpruned pack
        would produce a 20 GB container that cannot run, which is the worst
        outcome of all.
        """
        self._ffn_keep = {}
        self.imatrix = None
        keep = self.prune_ffn
        if not keep:
            return
        if self.gguf_reader is None:
            raise ValueError("--prune-ffn needs a GGUF source; the HF-safetensors "
                             "path has no imatrix to rank columns with")
        if self.imatrix_path_hint is None:
            raise FileNotFoundError(
                f"--prune-ffn {keep} was asked for but no imatrix was found. Pass "
                f"--imatrix PATH, or put the imatrix GGUF beside the model file. "
                f"Without one the only alternative is chopping the first {keep} "
                f"neurons, which retains {keep / self._ffn_width() * 100:.0f}% of the "
                f"FFN's activation mass by construction.")
        self.imatrix = imx.Imatrix(self.imatrix_path_hint)
        layers = self.imatrix.layers()
        if not layers:
            raise ValueError(f"--imatrix {self.imatrix_path_hint}: no *.in_sum2 tensors; "
                             f"that file does not look like an importance matrix")
        w = self._ffn_width()
        if keep >= w:
            print(f"[INFO] --prune-ffn {keep} is not narrower than the FFN ({w}); "
                  f"nothing to prune")
            return
        if keep % (64 * 8):
            print(f"[WARN] --prune-ffn {keep} is not a multiple of 512 (64-row bands "
                  f"over 8 cores); the kernel recipe will refuse this width. Nearest "
                  f"is {(keep // 512) * 512}.")
        mtp = self._mtp_blocks()
        if mtp:
            missing = mtp - set(layers)
            print(f"[INFO]   MTP block(s) {sorted(mtp)} are dropped (no speculative "
                  f"decoding), so their FFN is never narrowed"
                  + (f"; the imatrix also lacks them" if missing else ""))
        live = [L for L in layers if L not in mtp]
        for L in live:
            self._ffn_keep[L] = imx.ffn_index_set(self.imatrix, L, keep)
        rep = imx.retention_report(self.imatrix, keep, live)
        mean = sum(r for _, r in rep) / len(rep)
        worst = min(rep, key=lambda x: x[1])
        print(f"[INFO] --prune-ffn {keep}: narrowing the FFN {w} -> {keep} across "
              f"{len(layers)} layers by imatrix importance")
        print(f"[INFO]   activation mass retained: mean {mean * 100:.1f}%, "
              f"worst layer {worst[0]} at {worst[1] * 100:.1f}% "
              f"(a tail chop would keep {keep / w * 100:.1f}%)")
        self.prune_ffn_kept = keep
        self.prune_ffn_from = w
        self.prune_ffn_retained = mean

    def _declared_layers(self):
        """num_hidden_layers from the GGUF, or None when it is not declared."""
        f = self.gguf_reader.fields.get("qwen35.block_count")
        if f is None:
            return None
        n = int(f.contents())
        return n - len(self._mtp_blocks())

    def _ffn_width(self) -> int:
        f = self.gguf_reader.fields.get("qwen35.feed_forward_length")
        return int(f.contents()) if f else 0

    def _mtp_blocks(self) -> set:
        """Blocks that exist only to serve multi-token prediction, and are dropped.

        Qwen3.5 exports its MTP head as one EXTRA transformer block: the whole
        model ships `block_count` blocks of which the last carries the `.nextn.*`
        tensors (`eh_proj`, `enorm`, `hnorm`, `shared_head_norm`) and nothing else
        does. Qwen3.8-27B is block_count 65 with `num_hidden_layers` 64 and
        `mtp_num_hidden_layers` 1 -- so `blk.64` is that block, and it is
        speculative-decode scaffolding: this runtime has no MTP, so the weights
        are dead.

        Identified by the `.nextn.` tensors being PRESENT in a block, not by index
        arithmetic against the layer count: that is what actually marks the block,
        and it stays right for a model that ships no MTP at all, or ships two.
        """
        if self._mtp_cache is None:
            self._mtp_cache = {L for L in (imx.layer_of(n) for n in self.gguf_tensors)
                               if L is not None and any(
                                   n.startswith(f"blk.{L}.nextn.") for n in self.gguf_tensors)}
        return self._mtp_cache

    def _is_mtp_block(self, layer: int) -> bool:
        return layer in self._mtp_blocks()

    def _prune_tensor(self, gguf_tensor):
        """(d, m, qw) for one narrowed FFN tensor, or None when it is not one.

        Applies ONLY to a name this layer's index set was derived for, and uses
        that set verbatim -- `ffn_down`'s ranking, applied to all three tensors.
        Scoring each tensor's own columns separately would give a container that
        converts cleanly and computes a different network than the log claims.

        The dequantize -> gather -> quantize -> unpack sequence mirrors
        gguf_tensor's own `_requantize_to`, so the packed bytes are laid out by
        the same rules as an unpruned tensor: quantize the NARROWED matrix, then
        unpack it back into the (scale, min, quants) triple `_pack` takes. The
        unpack is fed the narrowed row count, not the source's -- passing the
        source's would mis-view the block array.
        """
        if self.imatrix is None:
            return None
        name = gguf_tensor.name
        L = imx.layer_of(name)
        if L is None or L not in self._ffn_keep:
            # No index set for this layer: the imatrix does not cover it. That is
            # NORMAL, not an error -- a model can carry an MTP block past the last
            # calibrated layer (Qwen3.8-27B has 65 blocks, the imatrix 64), and the
            # converter skips those tensors anyway. Leaving one FFN at full width
            # would be worse than useless: the container would declare a narrowed
            # intermediate_size while this layer still carried 17408 columns, which
            # the kernel recipe would build for and the runtime would misread.
            if L is not None and self.imatrix is not None and any(
                    name.endswith(r) for r in (imx.FFN_DOWN, imx.FFN_UP, imx.FFN_GATE)) \
                    and not self._is_mtp_block(L):
                raise ValueError(
                    f"--prune-ffn: layer {L} ({name}) is a real converted layer with an "
                    f"FFN, but the imatrix has no entry for it, so it cannot be narrowed. "
                    f"A pruned container must have EVERY FFN at the same width, because "
                    f"config.json declares ONE intermediate_size and the kernel recipe "
                    f"builds for it -- a layer left at {self._ffn_width()} would be read "
                    f"as {self.prune_ffn_kept}. "
                    f"The imatrix covers layers {min(self._ffn_keep)}.."
                    f"{max(self._ffn_keep)}; this model has a layer past that. "
                    f"Re-run the imatrix calibration over the whole model (it must include "
                    f"layer {L}), or pass an imatrix that does.")
            return None
        role = None
        for r in (imx.FFN_DOWN, imx.FFN_UP, imx.FFN_GATE):
            if name.endswith(r):
                role = r
                break
        if role is None:
            return None
        idx = self._ffn_keep[L]
        w = gguf_tensor.dequantize()
        if not torch.is_tensor(w):          # gguf returns ndarray in some paths
            w = torch.from_numpy(np.ascontiguousarray(w))
        w = imx.gather_ffn(w, idx, ffn_width=self._ffn_width())
        rows = w.shape[0]
        cols = w.shape[1]
        target = self.tensor_q4nx_type_map[name]
        # unpack_* takes the COLUMN COUNT of the gathered weight (its second
        # dim), not its row count: the unpackers reshape the flat quant array
        # as (-1, cols). Alluded to by the old comment below -- missed it.
        if target in (GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                      GGMLQuantizationType.BF16):
            return [w]
        if target == GGMLQuantizationType.Q4_K:
            # No Q4_K encoder in ggml; the packer fits the super-block itself
            # from a Q4_1-grid triple, whose min sign is SUBTRACTED not added.
            data = quantize(w.to(torch.float32).numpy(), GGMLQuantizationType.Q4_1).copy()
            d, m, qw = gguf_tensor.unpack_q4_1(data, cols)
            return (d, -m, qw)
        # NOT the bfloat16 round trip gguf_tensor._requantize_to uses. There `w` is
        # a raw byte buffer and `.view(torch.float32)` reinterprets those bytes; here
        # `w` is an already-dequantized float tensor, so the same call halves the
        # last axis and quantizes HALF the FFN -- a container that declares
        # intermediate_size 12288 while carrying 6144 columns. Round-tripping through
        # a real cast costs nothing here, and the float16 scale that _pack_q8nx
        # asserts on is applied at the pack boundary instead (see _pack).
        data = quantize(w.to(torch.float32).numpy(), target).copy()
        if target == GGMLQuantizationType.Q4_1:
            d, m, qw = gguf_tensor.unpack_q4_1(data, cols)
        elif target == GGMLQuantizationType.Q4_0:
            d, m, qw = gguf_tensor.unpack_q4_0(data, cols)
        elif target == GGMLQuantizationType.Q8_0:
            d, m, qw = gguf_tensor.unpack_q8_0(data, cols)
        else:
            raise ValueError(f"--prune-ffn: cannot quantize {name} to {target.name}")
        return (d, m, qw)

    def _convert_gguf(self, q4nx_path: str, weights_type: str):
        if weights_type == "language":
            self._resolve_pad_target()
            self._resolve_prune()
            # llama.cpp tiles the value heads only when there are more of them than key heads
            # (the 4B / 9B / 27B; not the 2B / 0.8B), and q | k precede v in qkv and conv1d.
            num_k, num_v, head_k, head_v = self._linear_heads()
            reorder_linear_required = num_v != num_k
            grp, qk_rows = num_v // num_k, 2 * num_k * head_k
            if reorder_linear_required:
                print(f"[INFO] Reorder linear required! ({num_v} value heads over {num_k} key heads)")

            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                emb = self.gguf_tensors["token_embd.weight"]
                unpacked = emb.unpack(GGMLQuantizationType.Q8_0)
                target_dtype = emb.get_used_quantization_type(GGMLQuantizationType.Q8_0)
                unpacked = self._maybe_pad(emb, unpacked, GGMLQuantizationType.Q8_0)
                self.q4nx_tensors["lm_head.weight"] = self._pack(*unpacked, tensor_type=target_dtype)

            # The MTP block is dropped from EVERY pack, pruned or not. It exists
            # only to feed speculative decoding, which this runtime has no use
            # for, and converting it yields a container with MORE layers than
            # config.json declares: the kernel recipe builds a per-layer set and
            # the engine walks num_hidden_layers, so that is a link-time mismatch
            # and the extra layer is not in the file to be found.
            mtp = self._mtp_blocks()
            if mtp:
                print(f"[INFO] Dropping MTP block(s) {sorted(mtp)}: the next-token "
                      f"prediction head has no use without speculative decoding, and "
                      f"config.json declares num_hidden_layers "
                      f"{self._declared_layers()} which already excludes it")
            for key, gguf_tensor in self.gguf_tensors.items():
                if ".nextn." in gguf_tensor.name:
                    print(f"[SKIP] {gguf_tensor.name} (MTP next-token prediction weights, absent from official Q4NX)")
                    continue
                if imx.layer_of(gguf_tensor.name) in mtp:
                    # Not one of the .nextn. tensors, but the rest of a block that
                    # exists only to feed them. Converting it would produce a
                    # container with MORE layers than config.json declares, and at a
                    # pruned FFN width the kernel set would be built for the wrong
                    # depth.
                    print(f"[SKIP] {gguf_tensor.name} (MTP scaffolding block)")
                    continue
                target_dtype = gguf_tensor.get_used_quantization_type(self.tensor_q4nx_type_map[gguf_tensor.name])
                print(f"Processing tensor: {gguf_tensor.name} with type {gguf_tensor.tensor_type.name} -> {self.forward_name_map[gguf_tensor.name]} with dtype {target_dtype.name}")
                if "token_embd.weight" in gguf_tensor.name:
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = torch.from_numpy(w).contiguous()
                    if self._pad_hidden is not None:
                        w = pad_hidden_axis(w, self._pad_actual, self._pad_hidden)
                        print(f"[INFO] Padded token_embd.weight hidden axis to {self._pad_hidden}")
                    w = w.to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    continue
                
                new_name = self.forward_name_map[gguf_tensor.name]

                pruned = self._prune_tensor(gguf_tensor)
                if pruned is not None:
                    # The FFN axis was narrowed before quantization, so this
                    # tensor is re-quantized from the gathered weight rather
                    # than unpacked from the source GGUF's own layout.
                    if isinstance(pruned, list):
                        self.q4nx_tensors[new_name] = pruned[0]
                    else:
                        self.q4nx_tensors[new_name] = self._pack(*pruned, tensor_type=target_dtype)
                    continue

                unpacked = gguf_tensor.unpack(target_dtype)
                unpacked = self._maybe_pad(gguf_tensor, unpacked, target_dtype)

                # Only full-attention layers produce a self_attn.q_proj target
                # (linear layers fuse q/k/v into attn_qkv). Branch on the
                # target name rather than layer_id % interval: the trailing
                # MTP block is full attention but sits at an index the modulo
                # test misclassifies (e.g. 24 with interval 4).
                if "self_attn.q_proj" in new_name:
                    print("[INFO] Seperate q, gate for q_proj")
                    DH = self.gguf_reader.fields["qwen35.attention.value_length"].contents()
                    d, m, qw = unpacked
                    d = rearrange(d, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                    m = rearrange(m, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                    qw = rearrange(qw, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                    unpacked = (d, m, qw)

                else:
                    if "self_attn.gate_proj" in self.forward_name_map[gguf_tensor.name]:
                        if reorder_linear_required:
                            print("[INFO] Reorder Gate")
                            unpacked = tuple(v_untile(x, grp, head_v) for x in unpacked)

                    if "qkv_proj" in self.forward_name_map[gguf_tensor.name]:
                        if reorder_linear_required:
                            print("[INFO] Reorder the value rows of qkv")
                            unpacked = tuple(untile_qkv(x, qk_rows, grp, head_v) for x in unpacked)

                    if "ssm_out_proj" in self.forward_name_map[gguf_tensor.name]:
                        if reorder_linear_required:
                            print(f"[INFO] Reorder for {self.forward_name_map[gguf_tensor.name]}")
                            d, m, qw = unpacked
                            # columns: d / m per 32-value block, qw per value
                            BLOCK_SIZE = 32
                            d = v_untile(d, grp, head_v // BLOCK_SIZE, axis=1)
                            m = v_untile(m, grp, head_v // BLOCK_SIZE, axis=1)
                            qw = v_untile(qw, grp, head_v, axis=1)

                            unpacked = (d, m, qw)

                    if "ssm_alpha_proj" in self.forward_name_map[gguf_tensor.name] or "ssm_beta_proj" in self.forward_name_map[gguf_tensor.name]:
                        d, m, qw = unpacked
                        w = gguf_tensor.dequantize()
                        if reorder_linear_required:
                            w = v_untile(w, grp, 1)

                        new_name = self.forward_name_map[gguf_tensor.name]
                        new_name = new_name.replace("alpha_proj", "alpha_proj.bf16").replace("beta_proj", "beta_proj.bf16")
                        self.q4nx_tensors[new_name] = w
                        if reorder_linear_required:
                            print(f"[INFO] Reorder for {self.forward_name_map[gguf_tensor.name]}")
                            d, m, qw = (v_untile(x, grp, 1) for x in (d, m, qw))

                        if (d.shape[0] < 32):
                            d = repeat(d, 'd c -> (r d) c', r = 2).contiguous()
                            m = repeat(m, 'd c -> (r d) c', r = 2).contiguous()
                            qw = repeat(qw, 'd c -> (r d) c', r = 2).contiguous()

                        unpacked = (d, m, qw)


                    if "ssm_conv1d" in self.forward_name_map[gguf_tensor.name]:
                        print("[INFO] transpose conv1d")

                        d = unpacked[0]

                        if reorder_linear_required:
                            d = untile_qkv(d, qk_rows, grp, head_v)      # the conv channels are qkv's rows
                        d = d.T.contiguous()
                        unpacked = [d]
                
                    if "ssm_a" in gguf_tensor.name[-5:]:
                        val = unpacked[0].to(torch.float32).contiguous()
                        if reorder_linear_required:
                            val = v_untile(val, grp, 1)
                        self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = val
                        continue

                    if "ssm_dt" in gguf_tensor.name:
                        val = unpacked[0].to(torch.float32).contiguous()
                        if reorder_linear_required:
                            val = v_untile(val, grp, 1)
                        self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = val
                        continue

                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack(*unpacked, tensor_type=target_dtype)
            self._extract_tokenizer_json(q4nx_path)                
        elif weights_type == "vision":
            # Some GGUF quantizers ship vision weights as a separate mmproj
            # file, leaving the language GGUF without any v.* tensors. Skip
            # cleanly: assemble_model_assets then sources vision_weight.q4nx
            # from the skeleton repo instead (see _ensure_qwen35_vision_weight).
            if not any(name.startswith("v.") for name in self.gguf_tensors):
                print("[WARN] No vision tensors in this GGUF; skipping vision conversion "
                      "(vision weights will be sourced from the skeleton if available)")
                return
            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Vision model tensors"
                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    weights = weights.to(torch.bfloat16)

                new_name = self.forward_name_map[gguf_tensor.name]

                if new_name.endswith("fc2.weight") or new_name.endswith("fc1.weight")\
                    or new_name.endswith("attn.proj.weight") or new_name.endswith("attn.qkv.weight"):
                    weights = self.vision_mm_weight_rearrange(weights)

                self.q4nx_tensors[new_name] = weights

            pe = "model.visual.patch_embed.proj.weight"
            pe1 = pe + ".1"
            if pe in self.q4nx_tensors and pe1 in self.q4nx_tensors:
                combined_patched_embeding = torch.stack(
                    [self.q4nx_tensors[pe], self.q4nx_tensors[pe1]], dim=2
                )
                del self.q4nx_tensors[pe]
                del self.q4nx_tensors[pe1]
                self.q4nx_tensors[pe] = combined_patched_embeding
    
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen35 model")

        self._export_weights(q4nx_path, weights_type)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        if weights_type == "language":
            self._convert_hf_language(q4nx_path)
        elif weights_type == "vision":
            self._convert_hf_vision(q4nx_path)
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen35 HF conversion")

    _Q8_0_NAMES = {"lm_head", "ssm_out_proj", "ssm_alpha_proj", "ssm_beta_proj"}

    def _store_q(self, q4nx_name: str, w: torch.Tensor):
        """Quantize + pack a 2D weight into the Q4NX block layout.

        To match the GGUF-derived reference, Q4_1-target tensors are quantized
        through an intermediate Q8_0 pass: the Q4_1 quantization is performed on
        the dequantized Q8_0 values, carrying the same round-trip error a Q8_0
        GGUF source would introduce. This makes the HF path produce identical
        bytes to the GGUF path for the same upstream weights.
        """
        target = (
            GGMLQuantizationType.Q8_0
            if any(n in q4nx_name for n in self._Q8_0_NAMES)
            else GGMLQuantizationType.Q4_1
        )
        w_np = w.to(torch.float32).numpy()
        columns = w_np.shape[1]
        # Quantize to Q8_0 first (intermediate, matches GGUF reference)
        q80 = quantize(w_np, GGMLQuantizationType.Q8_0).copy()
        # Dequantize back to float; this is what the GGUF path operates on
        w_deq = dequantize(q80, GGMLQuantizationType.Q8_0)
        w_deq = torch.from_numpy(w_deq).contiguous().to(torch.bfloat16)
        w_deq = w_deq.to(torch.float32).numpy()
        if target == GGMLQuantizationType.Q4_1:
            quantized = quantize(w_deq, target).copy()
            d, m, qw = GGUFTensor.unpack_q4_1(quantized, columns)
            # Apply the q_proj reorder to the unpacked blocks (matches GGUF path,
            # which reorders after unpacking rather than on the raw weight).
            if "self_attn.q_proj" in q4nx_name and self.head_dim is not None:
                DH = self.head_dim
                d = rearrange(d, '(g p h) c -> (p g h) c', p=2, h=DH).contiguous()
                m = rearrange(m, '(g p h) c -> (p g h) c', p=2, h=DH).contiguous()
                qw = rearrange(qw, '(g p h) c -> (p g h) c', p=2, h=DH).contiguous()
            self.q4nx_tensors[q4nx_name] = self._pack(d, m, qw, tensor_type=target)
        else:
            d, _, qw = GGUFTensor.unpack_q8_0(q80, columns)
            self.q4nx_tensors[q4nx_name] = self._pack(d, None, qw, tensor_type=target)

    def _convert_hf_language(self, q4nx_path: str):
        import re
        self.q4nx_tensors = {}
        # Explicit HF -> Q4NX name mapping for linear_attn tensors whose
        # HF names (in_proj_*, out_proj, A_log, dt_bias, norm) differ from
        # the Q4NX config names (qkv_proj, ssm_*, etc.)
        _HF_TO_Q4NX_LINEAR = {
            "linear_attn.in_proj_qkv.weight": "linear_attn.qkv_proj.weight",
            "linear_attn.in_proj_z.weight": "self_attn.gate_proj.weight",
            "linear_attn.in_proj_a.weight": "linear_attn.ssm_alpha_proj.weight",
            "linear_attn.in_proj_b.weight": "linear_attn.ssm_beta_proj.weight",
            "linear_attn.out_proj.weight": "linear_attn.ssm_out_proj.weight",
            "linear_attn.conv1d.weight": "linear_attn.ssm_conv1d.weight",
            "linear_attn.A_log": "linear_attn.ssm_a",
            "linear_attn.dt_bias": "linear_attn.ssm_dt.bias",
            "linear_attn.norm.weight": "linear_attn.ssm_norm.weight",
        }
        # Build HF name map with {bid} expanded to actual layer numbers.
        # HF names have prefix model.language_model.layers.{bid}.X
        # Q4NX names have prefix model.layers.{bid}.X
        # After stripping model.language_model. from HF, key is layers.{bid}.X
        # So we need Q4NX names with model. prefix stripped for {bid} entries.
        hf_name_map = {}
        for param_info in self.q4nx_config["name_map"].values():
            q4nx_name = param_info["q4nx_name"]
            if "rope_freqs" in q4nx_name:
                continue
            # HF names strip model.language_model. prefix, so keys become
            # layers.{bid}.X or embed_tokens.weight etc.
            # Q4NX names have model.layers.{bid}.X or model.embed_tokens.weight
            # We strip model. prefix so the map keys match the stripped HF keys.
            # Exception: visual/audio tensors keep model. prefix since HF has model.visual.*
            is_vision = "visual" in q4nx_name
            is_audio = "audio" in q4nx_name
            stripped = q4nx_name if (is_vision or is_audio) else q4nx_name.replace("model.", "", 1)
            if "{bid}" in q4nx_name:
                # For linear_attn entries, find the matching HF name
                hf_suffix = None
                for hf_pat, q4nx_suf in _HF_TO_Q4NX_LINEAR.items():
                    if stripped.endswith(q4nx_suf):
                        hf_suffix = hf_pat
                        break
                if hf_suffix is not None:
                    # Replace Q4NX suffix with HF suffix: layers.{bid}.linear_attn.qkv_proj.weight
                    # becomes layers.{bid}.linear_attn.in_proj_qkv.weight
                    hf_stripped = stripped[:-len(next(s for s in _HF_TO_Q4NX_LINEAR.values() if stripped.endswith(s)))] + hf_suffix
                    pattern = re.escape(hf_stripped).replace(r"\{bid\}", r"(\d+)")
                else:
                    pattern = re.escape(stripped).replace(r"\{bid\}", r"(\d+)")
                found = sorted(set(
                    int(m.group(1))
                    for n in self.weight_map
                    if (m := re.match("^" + pattern + "$", n.replace("model.language_model.", "")))
                ))
                for bid in found:
                    if hf_suffix is not None:
                        hf_key = hf_stripped.format(bid=bid)
                    else:
                        hf_key = stripped.format(bid=bid)
                    q4nx_key = q4nx_name.format(bid=bid)
                    hf_name_map[hf_key] = q4nx_key
            else:
                hf_name_map[stripped] = q4nx_name

        config_path = self.hf_dir / "config.json"
        head_dim = None
        ssm_state_size = None
        if config_path.is_file():
            with open(config_path) as f:
                cfg = json.load(f)
            text_cfg = cfg.get("text_config", {})
            head_dim = cfg.get("head_dim") or cfg.get("attention_value_length") \
                or text_cfg.get("head_dim") or text_cfg.get("attention_value_length")
            ssm_state_size = cfg.get("ssm_state_size") or cfg.get("conv_kernel_size") \
                or text_cfg.get("ssm_state_size") or text_cfg.get("conv_kernel_size")
        self.head_dim = head_dim

        reorder_linear = ssm_state_size is not None and ssm_state_size > 6144

        for name in sorted(self.weight_map):
            key = name.replace("model.language_model.", "")
            if ".nextn." in key or key.startswith("mtp."):
                continue
            if key.startswith("model.visual.") or key.startswith("model.audio."):
                continue
            if key not in hf_name_map:
                print(f"[WARN] Unmapped HF tensor: {name}")
                continue
            w = self._load_tensor(name)
            q4nx_name = hf_name_map[key]

            if key == "embed_tokens.weight":
                self.q4nx_tensors[q4nx_name] = w.to(torch.bfloat16)
                continue
            if key == "lm_head.weight":
                self._store_q(q4nx_name, w)
                continue

            # Norm/bias weights are stored as bf16.
            # Layernorm convention: engine expects weight + 1 (llama.cpp convention),
            # except ssm_norm which is stored raw.
            if any(key.endswith(s) for s in (".weight", ".bias")) and \
               any(p in key for p in ("layernorm", "norm", "_norm")):
                w = w.to(torch.bfloat16)
                if "ssm_norm" not in q4nx_name:
                    w = (w.float() + 1).to(torch.bfloat16)
                self.q4nx_tensors[q4nx_name] = w
                continue

            # ssm_a and ssm_dt.bias are stored as float32 (not quantized)
            if "linear_attn.ssm_a" in q4nx_name and q4nx_name.endswith("ssm_a"):
                # HF stores A_log (the raw log-magnitude parameter); the GGUF
                # convention (and Q4NX runtime) expects A = -exp(A_log).
                w = -torch.exp(w.float())
                if reorder_linear:
                    w = rearrange(w, '(q g) -> (g q)', q=2).contiguous()
                self.q4nx_tensors[q4nx_name] = w
                continue

            if "linear_attn.ssm_dt.bias" in q4nx_name:
                w = w.float()
                if reorder_linear:
                    w = rearrange(w, '(q g) -> (g q)', q=2).contiguous()
                self.q4nx_tensors[q4nx_name] = w
                continue

            # conv1d: squeeze + transpose to 2D
            if "linear_attn.ssm_conv1d.weight" in q4nx_name:
                w = w.squeeze()
                if w.dim() == 2:
                    w = w.T.contiguous()
                self.q4nx_tensors[q4nx_name] = w.to(torch.bfloat16)
                continue

            if reorder_linear:
                if "linear_attn.in_proj_qkv.weight" in key and ssm_state_size is not None:
                    d_half = w.shape[0] // 2
                    w0 = w[:d_half]
                    w1 = w[d_half:]
                    w1 = rearrange(w1, '(q g p) c -> (g q p) c', p=ssm_state_size, q=2).contiguous()
                    w = torch.cat([w0, w1], dim=0).contiguous()

                if "self_attn.gate_proj.weight" in q4nx_name and ssm_state_size is not None:
                    w = rearrange(w, '(q g p) c -> (g q p) c', p=ssm_state_size, q=2).contiguous()

                if "linear_attn.ssm_out_proj.weight" in q4nx_name and ssm_state_size is not None:
                    DH = ssm_state_size // 32
                    BLOCK_SIZE = 32
                    w = rearrange(w, 'r (q g p) -> r (g q p)', p=DH, q=2).contiguous()

                if "linear_attn.ssm_conv1d.weight" in q4nx_name and ssm_state_size is not None:
                    d_half = w.shape[0] // 2
                    w0 = w[:d_half]
                    w1 = w[d_half:]
                    w1 = rearrange(w1, '(q g p) c -> (g q p) c', p=ssm_state_size, q=2).contiguous()
                    w = torch.cat([w0, w1], dim=0).contiguous()
                    w = w.T.contiguous()

            # Alpha/beta: store bf16 variant + quantized variant
            if "linear_attn.ssm_alpha_proj.weight" in q4nx_name or "linear_attn.ssm_beta_proj.weight" in q4nx_name:
                # The bf16 variant is derived from the Q8_0 dequantization,
                # matching the GGUF path (which dequantizes the Q8_0 source
                # rather than keeping the original HF bf16).
                w_np = w.to(torch.float32).numpy()
                q80 = quantize(w_np, GGMLQuantizationType.Q8_0).copy()
                w_bf16 = dequantize(q80, GGMLQuantizationType.Q8_0)
                w_bf16 = torch.from_numpy(w_bf16).contiguous().to(torch.bfloat16)
                if reorder_linear:
                    w_bf16 = rearrange(w_bf16, '(q g) c -> (g q) c', q=2).contiguous()
                bf16_name = q4nx_name.replace("alpha_proj", "alpha_proj.bf16").replace("beta_proj", "beta_proj.bf16")
                self.q4nx_tensors[bf16_name] = w_bf16
                w_q = w_bf16.clone()
                if w_q.shape[0] < 32:
                    w_q = repeat(w_q, 'd c -> (r d) c', r=2).contiguous()
                self._store_q(q4nx_name, w_q)
                continue

            if w.dim() < 2:
                raise RuntimeError(f"1D tensor reached _store_q: {q4nx_name} key={key} shape={w.shape}")
            self._store_q(q4nx_name, w)

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
        self._export_weights(q4nx_path, "language")

    def _convert_hf_vision(self, q4nx_path: str):
        import re
        self.q4nx_tensors = {}
        # Build a proper HF name map with {bid} expanded to actual layer numbers.
        hf_name_map = {}
        for param_info in self.q4nx_config["name_map"].values():
            q4nx_name = param_info["q4nx_name"]
            if "rope_freqs" in q4nx_name or "{bid}" not in q4nx_name:
                continue
            # Detect how many layers exist for this pattern from weight_map
            pattern = re.escape(q4nx_name).replace(r"\{bid\}", r"(\d+)")
            found = sorted(set(
                int(m.group(1))
                for n in self.weight_map
                if (m := re.match("^" + pattern + "$", n))
            ))
            for bid in found:
                concrete = q4nx_name.format(bid=bid)
                hf_name_map[concrete] = concrete
        # Add non-bid entries (patch_embed, merger, etc.) — visual only
        for param_info in self.q4nx_config["name_map"].values():
            q4nx_name = param_info["q4nx_name"]
            if "rope_freqs" in q4nx_name or "{bid}" in q4nx_name:
                continue
            if "visual" not in q4nx_name:
                continue
            hf_name_map[q4nx_name] = q4nx_name

        for name in sorted(self.weight_map):
            if name not in hf_name_map:
                continue
            w = self._load_tensor(name)
            if w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)
            q4nx_name = hf_name_map[name]

            if q4nx_name.endswith("linear_fc2.weight") or q4nx_name.endswith("linear_fc1.weight") \
                or q4nx_name.endswith("attn.proj.weight") or q4nx_name.endswith("attn.qkv.weight"):
                w = self.vision_mm_weight_rearrange(w)

            self.q4nx_tensors[q4nx_name] = w

        # HF source already has merged patch_embed weight [C, 3, 2, H, W].
        # GGUF sources have two separate weights that need stacking.
        pe = "model.visual.patch_embed.proj.weight"
        pe1 = pe + ".1"
        if pe in self.q4nx_tensors and pe1 in self.q4nx_tensors:
            combined = torch.stack([self.q4nx_tensors[pe], self.q4nx_tensors[pe1]], dim=2)
            del self.q4nx_tensors[pe]
            del self.q4nx_tensors[pe1]
            self.q4nx_tensors[pe] = combined
        # If only `pe` exists and already has dim 2 == 2, it's pre-merged — leave as-is.

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX vision tensors")
        self._export_weights(q4nx_path, "vision")


class Qwen35_2B(Qwen35, model_arch=ModelArch.QWEN35_2B):
    pass


class Qwen35_08B(Qwen35, model_arch=ModelArch.QWEN35_08B):
    pass

class Qwen35_9B(Qwen35, model_arch=ModelArch.QWEN35_9B):
    pass


class Qwen35_27B(Qwen35, model_arch=ModelArch.QWEN35_27B):
    pass
