"""Qwen3.5 MoE (qwen35moe / qwen3.6-moe) converter: HF safetensors -> Q4NX.

The target Q4NX layout mirrors the official OpenFlowLM Qwen3.6-35B-A3B model.
Derived from the proven dense Qwen3.5 converter conventions and the official
Q4NX header (733 tensors, dtype policy):

- V heads stay in HF grouped order. The dense converter (num_k=16 != num_v=32,
  which llama.cpp tiles in GGUF) is re-untiled to grouped order by its proven
  (g q) reorders, so the engine consumes grouped order. No V reorder here.
- Full-attn q_proj only: (g p h) -> (p g h) head reorder, p=2, h=head_dim,
  matching the dense converter.
- Layernorm weights stored as weight + 1 (llama.cpp / dense-converter
  convention), except linear_attn.norm (ssm_norm) which is stored raw.
- A_log -> -exp(A_log); conv1d squeezed + transposed; alpha/beta transposed.
- dtype policy: BF16 norms/router/gates/alpha/beta/conv1d/ssm_norm/embed,
  F32 ssm_a + ssm_dt.bias, Q4_1 the three big expert mats, Q8_0 everything
  else quantized.
"""

from pathlib import Path
import json
import os

import numpy as np
import torch
from einops import rearrange
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
from safetensors import safe_open

from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from ..model_converter import __Q4NX_Converter

# Known names (suffixes) that appear without the trailing ".weight".
_HF_GATE_UP = "mlp.experts.gate_up_proj"
_HF_EXPERT_DOWN = "mlp.experts.down_proj"


class Qwen35Moe(__Q4NX_Converter, model_arch=ModelArch.QWEN35MOE):
    FULL_ATTENTION_INTERVAL = 4
    LINEAR_NUM_KEY_HEADS = 16
    LINEAR_NUM_VALUE_HEADS = 32
    LINEAR_KEY_HEAD_DIM = 128
    LINEAR_VALUE_HEAD_DIM = 128
    HEAD_DIM = 256
    Q_PROJ_P = 2
    NUM_ATTN_HEADS = 16
    NUM_KV_HEADS = 2

    def __init__(self, source: str | GGUFReader, config_json_path: str | None = None):
        print("[INFO] Using Qwen35Moe converter")
        self.gguf_reader: GGUFReader | None = None
        self.gguf_tensors: dict = {}
        self.hf_source: str | None = None
        self.hf_dir: Path | None = None
        self.weight_map: dict = {}
        self.hf_shards: dict = {}
        self.q4nx_tensors: dict = {}
        if isinstance(source, GGUFReader):
            print("[INFO] Qwen35Moe converter (GGUF source)")
            self.gguf_reader = source
            self.gguf_tensors = {t.name: t for t in source.tensors}
            self.initialize(config_json_path=config_json_path)
        else:
            print("[INFO] Qwen35Moe converter (HF safetensors source)")
            self.hf_source = source
            self.hf_dir = self._resolve_source(source)
            self.initialize(config_json_path=config_json_path)

    # ------------------------------------------------------------------ setup

    def initialize(self, config_json_path: str | None = None):
        self._load_config(config_file_path=config_json_path)
        if self.gguf_reader is None:
            self._read_hf_index()
        else:
            self._lin = self._linear_geom()
        self.stream_export = getattr(self, "stream_export", False)
        self._experts_pre_sliced = False

    def _linear_geom(self):
        """(num_k, num_v, head_k_dim, head_v_dim) of the gated DeltaNet from the GGUF's
        own ssm keys. Suffix-matched so qwen4exp.* and qwen35moe.* both resolve:
        llama.cpp writes group_count = key heads, time_step_rank = value heads,
        state_size = key head dim, inner_size = value heads x value head dim."""
        f = self.gguf_reader.fields

        def get(suffix):
            hits = [k for k in f if k.endswith(suffix)]
            if not hits:
                raise KeyError(f"GGUF lacks '*{suffix}': cannot place the DeltaNet value heads")
            return int(f[hits[0]].contents())

        nk = get("ssm.group_count")
        nv = get("ssm.time_step_rank")
        state = get("ssm.state_size")
        inner = get("ssm.inner_size")
        return nk, nv, state, inner // nv

    def _resolve_source(self, source: str) -> Path:
        path = Path(source)
        if path.is_dir():
            return path
        if "/" in source:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError("huggingface_hub is required to download HF sources")
            print(f"[INFO] Downloading {source} to HF cache...")
            local = snapshot_download(repo_id=source, allow_patterns=[
                "*.safetensors", "model.safetensors.index.json",
                "config.json", "tokenizer*", "*.json", "*.jinja",
            ])
            return Path(local)
        raise FileNotFoundError(f"HF source not found: {source}")

    def _read_hf_index(self):
        idx_path = self.hf_dir / "model.safetensors.index.json"
        if idx_path.is_file():
            index = json.loads(idx_path.read_text())
            self.weight_map = index["weight_map"]
            shards = sorted(set(self.weight_map.values()))
        else:
            single = self.hf_dir / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(
                    f"No safetensors weights found in {self.hf_dir}"
                )
            with safe_open(single, framework="torch") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
            shards = ["model.safetensors"]
        self.hf_shards = {s: self.hf_dir / s for s in shards}
        print(f"[INFO] Loaded {len(self.weight_map)} HF tensors across {len(shards)} shards")

    def _load_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(self.hf_shards[shard], framework="torch") as f:
            return f.get_tensor(name).contiguous()

    # ------------------------------------------------------------- conversion

    def convert(self, q4nx_path: str, weights_type: str = "language"):
        if weights_type != "language":
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen35Moe")
        self.q4nx_tensors = {}
        if self.gguf_reader is not None:
            self.prune_experts = getattr(self, "prune_experts", None)
            self.prune_moe_ffn = getattr(self, "prune_moe_ffn", None)
            self.imatrix_path_hint = getattr(self, "imatrix_path_hint", None)
            self._resolve_moe_prune()
            stream_writer = None
            if self.stream_export:
                from ..streaming_exporter import StreamingSafetensorsWriter, StreamingPack
                import os as _os
                out_path = _os.path.join(q4nx_path, "model.q4nx")
                _os.makedirs(q4nx_path, exist_ok=True)
                stream_writer = StreamingSafetensorsWriter(out_path)
                self.q4nx_tensors = StreamingPack(stream_writer)
                print(f"[INFO] Streaming tensor export to {out_path}; "
                      f"input pages are dropped as each tensor converts")
            # MTP scaffolding is dropped from every pack -- MTP (spec decoding)
            # is recorded as its own future artifact (see README), not half-
            # converted silently. A conforming source ships blk.N.nextn.*.
            mtp_layers = {
                int(nm.split(".")[1]) for nm in self.gguf_tensors
                if nm.startswith("blk.") and ".nextn." in nm}
            if mtp_layers:
                print(f"[INFO] Skipping MTP block(s) {sorted(mtp_layers)}: next-token-prediction "
                      f"scaffolding has no kernel consumer today and is intentionally not "
                      f"carried into the pack.")
            total_tensors = len(self.gguf_tensors)
            done = 0
            for name in sorted(self.gguf_tensors):
                done += 1
                if name.startswith("blk.") and ".nextn." in name:
                    print(f"[SKIP] {name} (MTP)")
                    continue
                if name.startswith("blk.") and int(name.split(".")[1]) in mtp_layers:
                    print(f"[SKIP] {name} (MTP block scaffolding)")
                    continue
                try:
                    self._process_gguf_tensor(name)
                finally:
                    from ..memfree import drop_pages
                    drop_pages(self.gguf_tensors[name].data)
                filled = round(24 * done / total_tensors)
                print(f"\r[INFO] convert {done}/{total_tensors} {'#'*filled}{'-'*(24-filled)} {name}   ", end="", flush=True)
            print()
            if stream_writer is not None:
                stream_writer.close()
                print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors (streamed to {stream_writer.path})")
            else:
                print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
                self._export_weights(q4nx_path, weights_type)
            self._extract_tokenizer_json(q4nx_path)
        else:
            self._convert_hf(q4nx_path, weights_type)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        """HF safetensors path. Supports fused experts and per-expert gate/up/down."""
        # Collect per-expert pieces so we can stack expert-major once per layer.
        expert_gate: dict[int, dict[int, torch.Tensor]] = {}
        expert_up: dict[int, dict[int, torch.Tensor]] = {}
        expert_down: dict[int, dict[int, torch.Tensor]] = {}

        for name in sorted(self.weight_map):
            key = name.replace("model.language_model.", "")
            if key.startswith("visual.") or key.startswith("model.visual."):
                continue
            if ".mlp.experts." in key and key.split(".")[-1] == "weight":
                # layers.{bid}.mlp.experts.{eid}.{gate,up,down}_proj.weight
                parts = key.split(".")
                if len(parts) >= 6 and parts[2] == "mlp" and parts[3] == "experts":
                    bid = int(parts[1])
                    eid = int(parts[4])
                    kind = parts[5]  # gate_proj / up_proj / down_proj
                    w = self._load_tensor(name)
                    if kind == "gate_proj":
                        expert_gate.setdefault(bid, {})[eid] = w
                    elif kind == "up_proj":
                        expert_up.setdefault(bid, {})[eid] = w
                    elif kind == "down_proj":
                        expert_down.setdefault(bid, {})[eid] = w
                    else:
                        print(f"[WARN] Unhandled expert tensor: {name}")
                    continue
            self._process_tensor(name)

        for bid in sorted(set(expert_gate) | set(expert_up) | set(expert_down)):
            prefix = f"model.layer.{bid}."
            if bid in expert_gate:
                eids = sorted(expert_gate[bid])
                gate = torch.stack([expert_gate[bid][e] for e in eids], dim=0)
                self._store_q(
                    prefix + "mlp.gate_exps_proj.weight",
                    gate.reshape(-1, gate.shape[-1]),
                )
            if bid in expert_up:
                eids = sorted(expert_up[bid])
                up = torch.stack([expert_up[bid][e] for e in eids], dim=0)
                self._store_q(
                    prefix + "mlp.up_exps_proj.weight",
                    up.reshape(-1, up.shape[-1]),
                )
            if bid in expert_down:
                eids = sorted(expert_down[bid])
                down = torch.stack([expert_down[bid][e] for e in eids], dim=0)
                self._store_q(
                    prefix + "mlp.down_exps_proj.weight",
                    down.reshape(-1, down.shape[-1]),
                )

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
        self._export_weights(q4nx_path, weights_type)

    # ------------------------------------------------------- GGUF conversion

    def _deq_gguf(self, name: str) -> torch.Tensor:
        """Dequantize a GGUF tensor to a float32 torch tensor, GGUF orientation.

        gguf.dequantize() already returns a correctly-shaped numpy array
        (row-major, e.g. [n_vocab, n_embd] for token_embd.weight). GGUFTensor.shape
        reports GGML's reversed ne[] axis order (e.g. [n_embd, n_vocab]) and must
        NOT be used to reshape the dequantized array - doing so reinterprets the
        buffer with the wrong strides and scrambles every element while leaving
        summary statistics (mean/abs) misleadingly unchanged.
        """
        gt = self.gguf_tensors[name]
        w = dequantize(gt.data, gt.tensor_type).copy()
        return torch.from_numpy(w).to(torch.float32).contiguous()

    def _carry_through(self, gguf_name: str, w: torch.Tensor) -> None:
        """Store a tensor the family has no mapping for.

        2D weights that quantize cleanly go through the normal Q8NX pack;
        anything that cannot (scalar-ish, or column counts that break the
        256-block pack) travels as bf16 under its GGUF name so no bytes are
        silently lost.
        """
        if w.ndim == 2:
            try:
                self._store_q(gguf_name, w)
                return
            except Exception:
                pass  # too narrow / odd shape for Q8NX, fall back to bf16
        self.q4nx_tensors[gguf_name] = self._bf16(w)

    def _deq_gguf_bf16(self, name: str, rows: int = 16384) -> torch.Tensor:
        """Row-chunked dequantise straight to bf16 (no full fp32 spike)."""
        gt = self.gguf_tensors[name]
        out = None
        for i in range(0, gt.data.shape[0], rows):
            W = dequantize(gt.data[i : i + rows], gt.tensor_type)
            if out is None:
                out = torch.empty((gt.data.shape[0], *W.shape[1:]), dtype=torch.bfloat16)
            out[i : i + rows] = torch.from_numpy(W).to(torch.bfloat16)
        return out

    def _deq_gguf_expert_slices(self, name: str) -> torch.Tensor:
        """Dequantize a fused expert tensor one expert at a time.

        ffn_{gate,up,down}_exps are [n_experts, inter, hidden] with the expert
        axis outermost in the byte layout, so each expert's quantized block is
        contiguous and can be dequantized on its own -- peak is one expert
        (~6.5 MB fp32 for Qwen3.8-Flash-Next) instead of the whole 3-of-48-
        layer [288, 640, 2560] fp32 (~1.9 GB). When expert pruning is active
        only the kept experts are ever read, so the pruned pack never even
        touches the dropped data off the stone.
        """
        gt = self.gguf_tensors[name]
        L = int(name.split(".")[1])
        kept = self._expert_idx.get(L)
        indices = range(gt.data.shape[0]) if kept is None or len(kept) == 0 else kept
        blocks = [dequantize(gt.data[e], gt.tensor_type) for e in indices]
        w = np.stack(blocks, axis=0)
        return torch.from_numpy(w).to(torch.float32).contiguous()

    def _untile_head_params(self, w: torch.Tensor) -> torch.Tensor:
        """llama.cpp tiled head params [NVP, NK] -> engine grouped order [NK, NVP]."""
        nk, nv = self._lin[0], self._lin[1]
        nvp = nv // nk
        w = w.reshape(nvp, nk).T.reshape(nk * nvp)
        return w.to(torch.float32).contiguous()

    def _untile_linear_rows(self, w: torch.Tensor) -> torch.Tensor:
        """Undo llama.cpp value-major tiling on linear-attn row dim: (q g p)->(g q p)."""
        nk, nv, _, head_v = self._lin
        return rearrange(
            w, "(q g p) c -> (g q p) c", p=head_v, q=nv // nk
        ).contiguous()

    def _untile_linear_cols(self, w: torch.Tensor) -> torch.Tensor:
        """Undo llama.cpp value-major tiling on linear-attn col dim: (q g p)->(g q p)."""
        nk, nv, _, head_v = self._lin
        return rearrange(
            w, "r (q g p) -> r (g q p)", p=head_v, q=nv // nk
        ).contiguous()

    def _untile_linear_heads(self, w: torch.Tensor) -> torch.Tensor:
        """Undo llama.cpp tiling on rank/head vectors: (q g)->(g q)."""
        nk, nv = self._lin[0], self._lin[1]
        return rearrange(w, "(q g) c -> (g q) c", q=nv // nk).contiguous()

    def _resolve_moe_prune(self):
        """Imatrix-derived index sets for optional MoE pruning.

        --prune-experts K: per layer, keep the K experts with the highest
        calibration dispatch frequency (the same `.counts` signal Guanaco's
        load_imatrix_prior uses to seed its hot-expert pinning).

        --prune-moe-ffn K: narrow the moe_intermediate axis to K columns of
        every routed expert, scored by the imatrix's per-neuron activation
        energy on ffn_down_exps (per-expert columns' in_sum2 summed over all
        experts). The SAME axis/index set is gathered from gate/up/down so
        the three still describe one network.

        Both verify the down-exps tensor names match the imatrix. Tied to
        AUDITS: the retention reports tell you the activation mass kept and
        are repeated from the dense path's semantics.

        Attaches self._moe_idx (ffn K axis idx per layer) and
        self._expert_idx (kept expert ids per layer). When a flag is given
        but no imatrix is reachable the build stops here, never ships an
        unmarked "full width with shorter name" container.
        """
        self._moe_ffn_idx = {}
        self._expert_idx = {}
        self.prune_moe_ffn_kept = None
        self.prune_moe_ffn_from = None
        self.prune_moe_ffn_retained = None
        self.prune_experts_kept = None
        self.prune_experts_from = None
        self.prune_experts_retained = None
        if getattr(self, "prune_ffn", None):
            raise ValueError("--prune-ffn (the dense FFN pruner) does not apply to a MoE; "
                             "use --prune-moe-ffn (intermediate width on the expert mats) "
                             "and/or --prune-experts (per-layer expert count).")
        if not (self.prune_experts or self.prune_moe_ffn):
            return
        if self.gguf_reader is None:
            raise ValueError("--prune-moe-ffn/--prune-experts need a GGUF source; "
                             "the HF-safetensors path has no imatrix to rank with")
        if not self.imatrix_path_hint:
            raise FileNotFoundError(
                f"--prune-moe-ffn/--prune-experts asked for but no imatrix was found. Pass "
                f"--imatrix PATH, or put a *.imatrix.gguf next to the model file.")
        from .. import imatrix_prune as imx
        self.imatrix = imx.Imatrix(self.imatrix_path_hint)
        layers = self.imatrix.layers()
        if not layers:
            raise ValueError(f"--imatrix {self.imatrix_path_hint}: no *.in_sum2 tensors")

        # The imatrix must describe the same expert set as the GGUF. The
        # matmul "reap" that produced a 288-expert GGUF does not invalidate
        # its 512-entry counts: without this check the prune paths apply
        # indices into the wrong axis entirely (this is a real mismatch the
        # bundled imatrix has on qwen3.8-flash-next-reap-288).
        counts0 = np.asarray(self.imatrix.counts(f"blk.{layers[0]}.ffn_gate_exps.weight")).reshape(-1)
        gt0 = self.gguf_tensors.get(f"blk.{layers[0]}.ffn_gate_exps.weight")
        if gt0 is not None:
            shape = getattr(gt0, "shape", None)
            E_gguf = int(shape[-1]) if shape is not None and len(shape) == 3 else None
            if E_gguf is not None and E_gguf != counts0.shape[0]:
                raise ValueError(
                    f"--prune-moe-ffn/--prune-experts: imatrix lists {counts0.shape[0]} "
                    f"experts but this GGUF tensor holds {E_gguf}. The imatrix predates\n"
                    f"        the reaping it was supposed to rank -- regenerate it "
                    f"(or drop the prune flags) and retry.")

        ffn_retained = []
        if self.prune_moe_ffn:
            # Snap K to a buildable MoE expert width *before* deriving the
            # per-layer index sets, so every layer's keep set is the width we
            # actually pack. The catalogue wants K as a multiple of 64 (the
            # Q4NX panel rule needs 128 * K % 8192 == 0 from the packer side);
            # an unbuildable K would produce a container even the recipe's own
            # checks later refuse. Print, let the dev Ctrl-C A if they want
            # the exact ff else we proceed — the reported line says what K the
            # container ended up at.
            first = np.asarray(self.imatrix.scores(f"blk.{layers[0]}.ffn_down_exps.weight"), dtype=np.float64)
            if first.ndim == 2:
                axis = 1 if first.shape[1] == len(counts0) else 0
                n = first.shape[1 - axis]
            else:
                n = first.shape[0]
            # The catalogue valid kernel frames for qwen36moe ask the expert
            # width to be a multiple of 128 whose stripe count (K // 128)
            # divides n_cols (= 8), i.e. K in {128, 256, 512, 1024}: an
            # unbuildable K would produce a container even the recipe's own
            # checks later refuse. Snap to the nearest valid one before we
            # derive the per-layer index sets, so every keep set uses the
            # width the container actually ships. Ctrl-C if you want the
            # exact (unbuildable) width back.
            candidates = [128 * d for d in (1, 2, 4, 8) if 128 * d <= n]
            k_req = min(self.prune_moe_ffn, n)
            if candidates and k_req not in candidates:
                snap = min(candidates, key=lambda c: (abs(c - k_req), c))
                print(f"[INFO] --prune-moe-ffn {k_req}: not packable as a qwen36moe "
                      f"expert width (the recipe's stripe rule lets a width be "
                      f"128*multiple-of-divisor-of-8: {candidates}); snapping to {snap} "
                      f"(nearest valid; Ctrl-C to stop, or pass K={snap}).")
                self.prune_moe_ffn = snap
        for L in layers:
            e_name = f"blk.{L}.ffn_gate_exps.weight"
            down_name = f"blk.{L}.ffn_down_exps.weight"
            counts = np.asarray(self.imatrix.counts(e_name)).reshape(-1)
            E = counts.shape[0]
            if self.prune_experts:
                keep = min(self.prune_experts, E)
                ranked = np.argsort(-counts)
                idx = np.sort(ranked[:keep])
                self._expert_idx[L] = idx
            if self.prune_moe_ffn:
                sc = np.asarray(self.imatrix.scores(down_name), dtype=np.float64)
                if sc.ndim == 2:
                    # [n, E] layout: the expert axis is whichever matches E.
                    axis = 1 if sc.shape[1] == E else 0
                    sc = sc.sum(axis=axis)
                n = sc.shape[0]
                keep = min(self.prune_moe_ffn, n)
                top = np.argpartition(sc, n - keep)[n - keep:]
                top.sort()
                self._moe_ffn_idx[L] = top.astype(np.int64)
                retained = float(sc[top].sum() / max(sc.sum(), 1e-12))
                ffn_retained.append(retained)
        if ffn_retained:
            self.prune_moe_ffn_retained = sum(ffn_retained) / len(ffn_retained)
        self._catalogue_prune_check()

        if self.prune_experts:
            counts_l = [np.asarray(self.imatrix.counts(f"blk.{L}.ffn_gate_exps.weight")).reshape(-1) for L in layers]
            self.prune_experts_kept = min(self.prune_experts, counts_l[0].shape[0])
            self.prune_experts_from = int(counts_l[0].shape[0])
            mean_r = sum(
                float(c[idx].sum() / max(c.sum(), 1e-12))
                for c, idx in zip(counts_l, [self._expert_idx[L] for L in layers])) / len(layers)
            self.prune_experts_retained = mean_r
            print(f"[INFO] --prune-experts {self.prune_experts_kept}: experts/layer {self.prune_experts_from} -> "
                  f"{self.prune_experts_kept}; calibration dispatch retained {self.prune_experts_retained * 100:.1f}% (mean)")
        if self.prune_moe_ffn:
            first = np.asarray(self.imatrix.scores(f"blk.{layers[0]}.ffn_down_exps.weight"), dtype=np.float64)
            if first.ndim == 2:
                countsE = np.asarray(self.imatrix.counts(f"blk.{layers[0]}.ffn_gate_exps.weight")).reshape(-1)
                axis = 1 if first.shape[1] == countsE.shape[0] else 0
                # The score runs over the NON-expert axis after summing the
                # expert axis away; a per-expert ffn length is that axis.
                n = first.shape[1 - axis] if first.ndim == 2 else first.shape[0]
            else:
                n = first.shape[0]
            self.prune_moe_ffn_kept = min(self.prune_moe_ffn, n)
            self.prune_moe_ffn_from = int(n)
            print(f"[INFO] --prune-moe-ffn {self.prune_moe_ffn_kept}: moe_intermediate/expert "
                  f"{self.prune_moe_ffn_from} -> {self.prune_moe_ffn_kept}; activation mass "
                  f"retained {self.prune_moe_ffn_retained * 100:.1f}% (mean over layers)")
            k = self.prune_moe_ffn_kept
            if k % 64 != 0 or (k == 0) or (128 * (k // 128)) != k or (8 % ((k // 128) or 1) != 0):
                print(f"[INFO]   --prune-moe-ffn {k} is not one this kernel set can pack: "
                      f"the qwen36moe recipe requires the expert width to be a "
                      f"multiple of 128 whose 128-row stripe count divides the 8 "
                      f"columns; {k} fails that. Valid widths: 128, 256, 512 (and "
                      f"1024 when the pool allows).")

    def _prune_shexp(self, w: torch.Tensor, L: int, is_down: bool) -> torch.Tensor:
        """Narrow the shared expert to the per-layer moe_ffn index set, IDENTICALLY
        to the routed experts: they ride one call-site path in the kernels, so a
        container that narrows only the routed experts is structurally refused.
        gate/up: the kept axis is the leading one ([inter, hidden]); down: it is
        the trailing one ([hidden, inter])."""
        ids = self._moe_ffn_idx.get(L)
        if ids is None or len(ids) == 0:
            return w
        ids = torch.as_tensor(ids, dtype=torch.long)
        return w.index_select(1 if is_down else 0, ids).contiguous()

    def _catalogue_prune_check(self) -> None:
        """Info (never an error) when the pruned MoE FFN width is not one of the
        kernel catalogue's validated points -- the losing path is simply 'no
        validated kernels for this K yet', surfaced up front rather than after
        a 20 GB container build. Mirrors _build_spec's discovery of open_kernels."""
        k = self.prune_moe_ffn_kept
        if k is None:
            return
        checkout = os.environ.get("OPEN_KERNELS_DIR")
        candidates = [Path(checkout)] if checkout else []
        candidates += [p / "open_kernels" for p in Path(__file__).resolve().parents]
        candidates.append(Path.cwd() / "open_kernels")
        root = next((c for c in candidates if (c / "recipes" / "catalogue.py").is_file()), None)
        if root is None:
            return
        import sys
        sys.path.insert(0, str(root))
        try:
            from recipes.catalogue import CATALOGUE
        except Exception:
            return
        try:
            template = CATALOGUE.get("moe")
            if template is None:
                return
            p = template.params.get("ff")
            if p is not None and not p.ok(int(k)):
                print(f"[INFO] --prune-moe-ffn {k}: this K is outside the validated "
                      f"kernel catalogue set {p.expected()} -- kernel export will "
                      f"report 'UNVALIDATED point allowed by OPEN_KERNELS_UNVALIDATED: "
                      f"moe: ff={k}'; pick one of {p.expected()} to ship with it")
        except Exception:
            return

    def _moe_prune_tensor(self, w: torch.Tensor, L: int, kind: str) -> torch.Tensor:
        """Gather one expert tensor by the resolved per-layer index sets.

        kind is 'gate'/'up': natural [E, inter, hidden]; 'down': [E, hidden, inter];
        'router': [E, hidden]."""
        if L in self._expert_idx and not self._experts_pre_sliced:
            ids = torch.as_tensor(self._expert_idx[L], dtype=torch.long)
            w = w.index_select(0, ids)
        if L in self._moe_ffn_idx and kind in ("gate", "up"):
            ids = torch.as_tensor(self._moe_ffn_idx[L], dtype=torch.long)
            w = w.index_select(1, ids)
        elif L in self._moe_ffn_idx and kind == "down":
            ids = torch.as_tensor(self._moe_ffn_idx[L], dtype=torch.long)
            w = w.index_select(2, ids)
        return w.contiguous()

    def _process_gguf_tensor(self, gguf_name: str):
        # --- globals ---
        # gguf.dequantize() already returns the natural [out, in] / [vocab, hidden]
        # numpy shape (same orientation as the HF safetensors path) - no transpose.
        if gguf_name == "token_embd.weight":
            self.q4nx_tensors["model.embed_tokens.weight"] = self._deq_gguf_bf16(gguf_name)
            return
        if gguf_name == "output.weight":
            self._store_q("lm_head.weight", self._deq_gguf(gguf_name))
            return
        if gguf_name == "output_norm.weight":
            self.q4nx_tensors["model.norm.weight"] = self._bf16(self._deq_gguf(gguf_name))
            return
        if gguf_name.startswith("blk."):
            self._process_gguf_layer_tensor(gguf_name)
            return
        if gguf_name == "per_layer_token_embd.weight":
            # The PLE hash table is a n-gram-indexed [160, ~320M] store (35 GB as
            # Q5_0): neither dequantizable nor a GEMM weight. A speculative pack
            # records its presence and drops it, with its siblings (PLE layer).
            print(f"[WARN] Skipping {gguf_name} (35 GB n-gram hash store); the speculative pack cannot dequantize it")
            return
        print(f"[WARN] Unhandled GGUF tensor: {gguf_name}; carrying through under its GGUF name")
        self._carry_through(gguf_name, self._deq_gguf(gguf_name))

    def _process_gguf_layer_tensor(self, gguf_name: str):
        # name like: blk.0.attn_qkv.weight
        parts = gguf_name.split(".")
        bid = int(parts[1])
        rest = ".".join(parts[2:])
        prefix = f"model.layer.{bid}."

        self._experts_pre_sliced = is_expert = rest in (
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
        if is_expert:
            # Streaming expert path: dequantize the (optionally pruned) set of
            # experts one byte-block at a time instead of materializing the
            # whole [n_experts, ...] float32 tensor at once.
            w = self._deq_gguf_expert_slices(gguf_name)
        else:
            w = self._deq_gguf(gguf_name)

        # --- norms (GGUF already stores the Q4NX weight + 1 convention) ---
        if rest == "attn_norm.weight":
            self.q4nx_tensors[prefix + "input_layernorm.weight"] = self._bf16(w)
            return
        if rest == "post_attention_norm.weight":
            self.q4nx_tensors[prefix + "post_attention_layernorm.weight"] = self._bf16(w)
            return
        if rest == "ssm_norm.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_norm.weight"] = self._bf16(w)
            return
        if rest == "attn_q_norm.weight":
            self.q4nx_tensors[prefix + "self_attn.q_norm.weight"] = self._bf16(w)
            return
        if rest == "attn_k_norm.weight":
            self.q4nx_tensors[prefix + "self_attn.k_norm.weight"] = self._bf16(w)
            return

        # --- linear-attention scalars / small projections (BF16/F32) ---
        # llama.cpp tiles linear-attn heads value-major; engine wants HF grouped
        # order. Matches dense qwen35.py reorder_linear_required path.
        # GGUF already stores A_log as -exp(A_log).
        if rest == "ssm_a":
            self.q4nx_tensors[prefix + "linear_attn.ssm_a"] = self._untile_head_params(w)
            return
        if rest == "ssm_dt.bias":
            self.q4nx_tensors[prefix + "linear_attn.ssm_dt.bias"] = self._untile_head_params(w)
            return
        if rest == "ssm_alpha.weight":
            w = self._untile_linear_heads(w)
            self.q4nx_tensors[prefix + "linear_attn.ssm_alpha_proj.weight"] = self._bf16(w.t())
            return
        if rest == "ssm_beta.weight":
            w = self._untile_linear_heads(w)
            self.q4nx_tensors[prefix + "linear_attn.ssm_beta_proj.weight"] = self._bf16(w.t())
            return
        if rest == "ssm_conv1d.weight":
            # Second half of channels is value-tiled; first half stays.
            nk, nv, state, head_v = self._lin
            w0, w1 = w[: 2 * nk * state], w[2 * nk * state:]
            w = torch.cat([w0, self._untile_linear_rows(w1)], dim=0).contiguous()
            self.q4nx_tensors[prefix + "linear_attn.ssm_conv1d.weight"] = self._bf16(w.t())
            return

        # --- router / shared-expert gates (BF16) ---
        # gguf.dequantize() returns ffn_gate_inp as natural [n_experts, hidden];
        # Q4NX wants [hidden, n_experts] (matches the HF mlp.gate.weight.t() convention).
        if rest == "ffn_gate_inp.weight":
            if self._expert_idx and bid in self._expert_idx:
                ids = torch.as_tensor(self._expert_idx[bid], dtype=torch.long)
                w = w.index_select(0, ids)
            self.q4nx_tensors[prefix + "moe_router.weight"] = self._bf16(w.t())
            return
        if rest == "ffn_gate_inp_shexp.weight":
            self.q4nx_tensors[prefix + "shared_expert_gate.weight"] = self._bf16(w.reshape(-1))
            return

        # --- quantized 2D weights ---
        # Linear-attn mats need the same llama.cpp untile as dense qwen35.
        # Full-attn q in GGUF is HF-ordered (g p h); the engine wants (p g h)
        # (matches the dense converter and the official Q4NX layout).
        if rest == "attn_qkv.weight":
            nk, nv, state, head_v = self._lin
            w0, w1 = w[: 2 * nk * state], w[2 * nk * state:]
            w = torch.cat([w0, self._untile_linear_rows(w1)], dim=0).contiguous()
            self._store_q(prefix + "linear_attn.qkv_proj.weight", w)
            return
        if rest == "attn_gate.weight":
            w = self._untile_linear_rows(w)
            self._store_q(prefix + "self_attn.gate_proj.weight", w)
            return
        if rest == "ssm_out.weight":
            w = self._untile_linear_cols(w)
            self._store_q(prefix + "linear_attn.ssm_out_proj.weight", w)
            return
        if rest == "attn_q.weight":
            w = rearrange(w, "(g p h) c -> (p g h) c", p=self.Q_PROJ_P, h=self.HEAD_DIM).contiguous()
            self._store_q(prefix + "self_attn.q_proj.weight", w)
            return
        if rest == "attn_k.weight":
            self._store_q(prefix + "self_attn.k_proj.weight", w)
            return
        if rest == "attn_v.weight":
            self._store_q(prefix + "self_attn.v_proj.weight", w)
            return
        if rest == "attn_output.weight":
            self._store_q(prefix + "self_attn.o_proj.weight", w)
            return
        if rest == "ffn_gate_shexp.weight":
            w = self._prune_shexp(w, bid, is_down=False)
            self._store_q(prefix + "mlp.share_gate_exps_proj.weight", w)
            return
        if rest == "ffn_up_shexp.weight":
            w = self._prune_shexp(w, bid, is_down=False)
            self._store_q(prefix + "mlp.share_up_exps_proj.weight", w)
            return
        if rest == "ffn_down_shexp.weight":
            w = self._prune_shexp(w, bid, is_down=True)
            self._store_q(prefix + "mlp.share_down_exps_proj.weight", w)
            return

        # --- routed experts: gguf.dequantize() natural shape is already
        # [n_experts, inter, hidden] - just flatten expert-major, no permute. ---
        if rest == "ffn_gate_exps.weight":
            w = self._moe_prune_tensor(w, bid, "gate")
            w = w.reshape(-1, w.shape[-1]).contiguous()
            self._store_q(prefix + "mlp.gate_exps_proj.weight", w)
            return
        if rest == "ffn_up_exps.weight":
            w = self._moe_prune_tensor(w, bid, "up")
            w = w.reshape(-1, w.shape[-1]).contiguous()
            self._store_q(prefix + "mlp.up_exps_proj.weight", w)
            return
        if rest == "ffn_down_exps.weight":
            w = self._moe_prune_tensor(w, bid, "down")
            w = w.reshape(-1, w.shape[-1]).contiguous()
            self._store_q(prefix + "mlp.down_exps_proj.weight", w)
            return

        print(f"[WARN] Unhandled GGUF layer tensor: {gguf_name}; carrying through under its GGUF name")
        self._carry_through(gguf_name, w)

    def _process_tensor(self, hf_name: str):
        key = hf_name.replace("model.language_model.", "")
        if key.startswith("layers."):
            self._process_layer_tensor(hf_name, key)
            return
        # globals
        if key == "embed_tokens.weight":
            self.q4nx_tensors["model.embed_tokens.weight"] = self._bf16(self._load_tensor(hf_name))
        elif key == "lm_head.weight":
            self._store_q("lm_head.weight", self._load_tensor(hf_name))
        elif key == "norm.weight":
            w = self._bf16(self._load_tensor(hf_name).float() + 1)
            self.q4nx_tensors["model.norm.weight"] = w
        else:
            print(f"[WARN] Unhandled global tensor: {hf_name}")

    def _process_layer_tensor(self, hf_name: str, key: str):
        # key like: layers.0.linear_attn.in_proj_qkv.weight
        parts = key.split(".")
        bid = int(parts[1])
        rest = ".".join(parts[2:])
        prefix = f"model.layer.{bid}."

        w = self._load_tensor(hf_name)

        # --- layernorms (weight + 1, except linear_attn.norm) ---
        if rest == "input_layernorm.weight":
            self.q4nx_tensors[prefix + "input_layernorm.weight"] = self._bf16(w.float() + 1)
            return
        if rest == "post_attention_layernorm.weight":
            self.q4nx_tensors[prefix + "post_attention_layernorm.weight"] = self._bf16(w.float() + 1)
            return

        # --- linear attention ---
        if rest == "linear_attn.in_proj_qkv.weight":
            self._store_q(prefix + "linear_attn.qkv_proj.weight", w)
            return
        if rest == "linear_attn.in_proj_z.weight":
            self._store_q(prefix + "self_attn.gate_proj.weight", w)
            return
        if rest == "linear_attn.in_proj_a.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_alpha_proj.weight"] = self._bf16(w.t())
            return
        if rest == "linear_attn.in_proj_b.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_beta_proj.weight"] = self._bf16(w.t())
            return
        if rest == "linear_attn.out_proj.weight":
            self._store_q(prefix + "linear_attn.ssm_out_proj.weight", w)
            return
        if rest == "linear_attn.conv1d.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_conv1d.weight"] = self._bf16(w.squeeze().t())
            return
        if rest == "linear_attn.A_log":
            self.q4nx_tensors[prefix + "linear_attn.ssm_a"] = (-torch.exp(w.float())).contiguous()
            return
        if rest == "linear_attn.dt_bias":
            self.q4nx_tensors[prefix + "linear_attn.ssm_dt.bias"] = w.to(torch.float32).contiguous()
            return
        if rest == "linear_attn.norm.weight":
            self.q4nx_tensors[prefix + "linear_attn.ssm_norm.weight"] = self._bf16(w)
            return

        # --- full attention ---
        if rest == "self_attn.q_proj.weight":
            w = rearrange(w, "(g p h) c -> (p g h) c", p=self.Q_PROJ_P, h=self.HEAD_DIM).contiguous()
            self._store_q(prefix + "self_attn.q_proj.weight", w)
            return
        if rest == "self_attn.k_proj.weight":
            self._store_q(prefix + "self_attn.k_proj.weight", w)
            return
        if rest == "self_attn.v_proj.weight":
            self._store_q(prefix + "self_attn.v_proj.weight", w)
            return
        if rest == "self_attn.o_proj.weight":
            self._store_q(prefix + "self_attn.o_proj.weight", w)
            return
        if rest == "self_attn.q_norm.weight":
            self.q4nx_tensors[prefix + "self_attn.q_norm.weight"] = self._bf16(w.float() + 1)
            return
        if rest == "self_attn.k_norm.weight":
            self.q4nx_tensors[prefix + "self_attn.k_norm.weight"] = self._bf16(w.float() + 1)
            return

        # --- MLP / MoE ---
        if rest == "mlp.gate.weight":
            self.q4nx_tensors[prefix + "moe_router.weight"] = self._bf16(w.t())
            return
        if rest == "mlp.shared_expert_gate.weight":
            self.q4nx_tensors[prefix + "shared_expert_gate.weight"] = self._bf16(w.reshape(-1))
            return
        if rest == "mlp.shared_expert.gate_proj.weight":
            self._store_q(prefix + "mlp.share_gate_exps_proj.weight", w)
            return
        if rest == "mlp.shared_expert.up_proj.weight":
            self._store_q(prefix + "mlp.share_up_exps_proj.weight", w)
            return
        if rest == "mlp.shared_expert.down_proj.weight":
            self._store_q(prefix + "mlp.share_down_exps_proj.weight", w)
            return
        if rest == _HF_GATE_UP:
            # [num_experts, 2 * moe_intermediate, hidden] -> gate + up, flattened expert-major
            gate, up = w.chunk(2, dim=1)
            self._store_q(prefix + "mlp.gate_exps_proj.weight", gate.reshape(-1, gate.shape[-1]))
            self._store_q(prefix + "mlp.up_exps_proj.weight", up.reshape(-1, up.shape[-1]))
            return
        if rest == _HF_EXPERT_DOWN:
            self._store_q(prefix + "mlp.down_exps_proj.weight", w.reshape(-1, w.shape[-1]))
            return

        print(f"[WARN] Unhandled tensor: {hf_name}")

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _bf16(w: torch.Tensor) -> torch.Tensor:
        return w.to(torch.bfloat16).contiguous()

    _Q4_1_NAMES = {
        "mlp.gate_exps_proj.weight",
        "mlp.up_exps_proj.weight",
        "mlp.down_exps_proj.weight",
    }

    def _store_q(self, q4nx_name: str, w: torch.Tensor):
        """Quantize + pack a 2D weight into the Q4NX block layout."""
        target = (
            GGMLQuantizationType.Q4_1
            if q4nx_name.endswith(tuple(Qwen35Moe._Q4_1_NAMES))
            else GGMLQuantizationType.Q8_0
        )
        w_np = w.to(torch.float32).numpy()
        quantized = quantize(w_np, target).copy()
        columns = w_np.shape[1]
        if target == GGMLQuantizationType.Q4_1:
            d, m, qw = GGUFTensor.unpack_q4_1(quantized, columns)
            self.q4nx_tensors[q4nx_name] = self._pack(d, m, qw, tensor_type=target)
        else:
            d, _, qw = GGUFTensor.unpack_q8_0(quantized, columns)
            self.q4nx_tensors[q4nx_name] = self._pack(d, None, qw, tensor_type=target)
