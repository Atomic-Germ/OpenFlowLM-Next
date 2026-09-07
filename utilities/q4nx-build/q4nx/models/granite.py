"""IBM Granite (dense) -> Q4NX.

Granite is Llama with four scalar multipliers applied at fixed points of the
forward pass. Tensor names, GQA, RoPE, SwiGLU and RMSNorm are all Llama's, and
`gguf`'s tensor set for `granite` is exactly Llama's dense subset -- so the only
thing this builder adds is the multipliers.

**All four fold exactly into the weights**, which is what makes Granite runnable
on the open kernels without a runtime change: nothing in `dx.py` or `attn.h`
reads a per-model attention scale. Reading HF's `GraniteAttention` /
`GraniteModel` forward pass, with `hd` for head_dim:

    attention_multiplier   replaces llama's implicit hd**-0.5 attention scale
                           -> q_proj      *= attention_multiplier * sqrt(hd)
    embedding_multiplier   inputs_embeds = embed(ids) * embedding_multiplier
                           -> embed_tokens *= embedding_multiplier
    residual_multiplier    h = residual + h * residual_multiplier, after BOTH
                           the attention block and the MLP block
                           -> o_proj      *= residual_multiplier
                              down_proj   *= residual_multiplier
    logits_scaling         logits = lm_head(h) / logits_scaling
                           -> lm_head     *= 1 / logits_scaling

Folding into an ALREADY QUANTIZED tensor is lossless, and cheaper than folding
into the float weights before quantizing. A Q4_1 block stores `w = code * d + m`
with `d = (max-min)/15` and `m = min`; scaling every weight in the block by a
constant `c > 0` scales `max` and `min` by `c`, hence `d` and `m` by `c`, and
leaves every 4-bit code untouched. So scaling `(d, m)` after quantization is
bit-identical to quantizing `c*W` directly -- see `tests/test_granite_fold.py`,
which asserts exactly that, including that the codes do not move.

For granite-4.2-3b three of the four are 1.0 and the fourth is
`attention_multiplier = 0.015625` at `hd = 64`, so the single fold is
`q_proj *= 0.015625 * 8 = 0.125`. **A power of two, so even the `d`/`m` scaling
is exact in bf16** -- it is an exponent shift.

RoPE commutes with the q_proj fold: RoPE is a rotation, and a rotation of a
scaled vector is the scaled rotation of it.

WHAT THE OPEN RECIPE THEN REQUIRES. `open_kernels/recipes/spec.py` refuses a
container whose `attention_multiplier` is not `head_dim ** -0.5`, and refuses
one that does not state it at all -- because after this fold the model uses
exactly the `1/sqrt(HD)` that `attn.h` hard-codes, and an unfolded container
would run and return plausible garbage. `model_assets.apply_granite_fold_to_config`
writes the post-fold value into the deployed `config.json`, so a container this
builder produces is one that recipe accepts, by construction.

Not supported here: the HF safetensors path. The shared HF flow in
`model_converter.py` stores raw float tensors and never calls `_pack_q4nx`, so
it cannot produce a Q4NX file the runtime reads. Convert from a GGUF instead.
"""

from __future__ import annotations

import math

import torch
from gguf import GGUFReader, dequantize

from ..constants import ModelArch
from ..model_converter import __Q4NX_Converter


# GGUF metadata keys are tried under both prefixes: a GGUF converted with
# `-f granite` from a file whose own architecture string is `llama` still
# carries `llama.*` keys.
ARCH_PREFIXES = ("granite", "llama")


def fold_factors(attention_multiplier: float, embedding_multiplier: float,
                 residual_multiplier: float, logits_scaling: float,
                 head_dim: int) -> dict[str, float]:
    """Per-tensor multiplicative folds, keyed by q4nx-name suffix.

    Only entries that change something are returned, so a checkpoint whose
    multipliers are all 1.0 does no work and touches no weights. Pure and
    module-level so the arithmetic is testable without a GGUF.
    """
    folds: dict[str, float] = {}
    q_fold = attention_multiplier * math.sqrt(head_dim)
    if q_fold != 1.0:
        folds["self_attn.q_proj.weight"] = q_fold
    if residual_multiplier != 1.0:
        folds["self_attn.o_proj.weight"] = residual_multiplier
        folds["mlp.down_proj.weight"] = residual_multiplier
    if embedding_multiplier != 1.0:
        folds["model.embed_tokens.weight"] = embedding_multiplier
    if logits_scaling != 1.0:
        folds["lm_head.weight"] = 1.0 / logits_scaling
    return folds


def fold_factor_for(q4nx_name: str, folds: dict[str, float]) -> float | None:
    for suffix, factor in folds.items():
        if q4nx_name.endswith(suffix):
            return factor
    return None


def scale_unpacked(unpacked, factor: float):
    """Scale a tensor in whatever form `GGUFTensor.unpack` returned it.

    `(d, m, qs)` for a block-quantized tensor -- scale the per-block scale and
    minimum and leave the 4-bit codes alone. `(w,)` for a float passthrough --
    scale the values.
    """
    if len(unpacked) == 3:
        d, m, qs = unpacked
        return (d * factor, m * factor, qs)
    if len(unpacked) == 1:
        return (unpacked[0] * factor,)
    raise ValueError(f"Unexpected unpacked tensor arity: {len(unpacked)}")


class Granite(__Q4NX_Converter, model_arch=ModelArch.GRANITE):
    """IBM Granite 4.x dense (granite-4.2-3b and its shape-mates)."""

    def __init__(self, source, config_json_path=None):
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
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        if self.gguf_reader is None:
            raise ValueError(
                "Granite conversion reads a GGUF: the HF safetensors path stores tensors "
                "unquantized, which the open kernels' q4_1 GEMV cannot read. Convert the "
                "model with llama.cpp's convert_hf_to_gguf.py (or use "
                "ibm-granite/granite-4.2-3b-GGUF) first."
            )
        self._convert_gguf(q4nx_path, weights_type)

    def _meta(self, suffix: str):
        for prefix in ARCH_PREFIXES:
            field = self.gguf_reader.fields.get(f"{prefix}.{suffix}")
            if field is not None:
                return field.contents()
        return None

    def _head_dim(self) -> int:
        """head_dim, from rope.dimension_count with a head-count cross-check."""
        rope_dim = self._meta("rope.dimension_count")
        heads = self._meta("attention.head_count")
        embedding_length = self._meta("embedding_length")
        derived = (embedding_length // heads) if (heads and embedding_length) else None

        if rope_dim is None:
            if derived is None:
                raise KeyError("Cannot determine head_dim: the GGUF has neither "
                               "rope.dimension_count nor embedding_length/head_count")
            return int(derived)
        if derived is not None and derived != rope_dim:
            # Granite applies RoPE to the full head, so these must agree. A
            # disagreement means a partial rotary factor, which would make the
            # q/k permutation below wrong -- refuse rather than emit silently
            # broken weights.
            raise ValueError(
                f"head_dim is ambiguous: rope.dimension_count={rope_dim} but "
                f"embedding_length/head_count={derived}. Partial-rotary Granite "
                f"variants are not supported.")
        return int(rope_dim)

    def _folds(self) -> dict[str, float]:
        head_dim = self._head_dim()
        attn = self._meta("attention.scale")
        # A GGUF without the key is a plain Llama-scaled model, whose implicit
        # hd**-0.5 is already what attn.h applies.
        attn = head_dim ** -0.5 if attn is None else float(attn)
        emb = self._meta("embedding_scale")
        res = self._meta("residual_scale")
        logit = self._meta("logit_scale")
        emb = 1.0 if emb is None else float(emb)
        res = 1.0 if res is None else float(res)
        logit = 1.0 if logit is None else float(logit)

        print(f"[INFO] Granite multipliers: attention={attn}, embedding={emb}, "
              f"residual={res}, logits_scaling={logit} (head_dim={head_dim})")
        folds = fold_factors(attn, emb, res, logit, head_dim)
        for name, factor in folds.items():
            print(f"[INFO] Granite fold: {name} *= {factor}")
        if not folds:
            print("[INFO] Granite folds: none needed (every multiplier is 1.0)")
        return folds

    def _convert_gguf(self, q4nx_path: str, weights_type: str):
        print("[INFO] Converting granite model to Q4NX format...")
        folds = self._folds()
        head_dim = self._head_dim()

        if not self._has_lm_head():
            # Tied embeddings: lm_head comes from the UNSCALED table, so it is
            # built here -- before the embedding fold is applied in the loop --
            # and carries only its own logits_scaling fold.
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
            head_fold = folds.get("lm_head.weight")
            if head_fold is not None:
                unpacked = scale_unpacked(unpacked, head_fold)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        for gguf_tensor in self.gguf_tensors.values():
            if gguf_tensor.name not in self.forward_name_map:
                print(f"[WARN] Unmapped GGUF tensor, skipping: {gguf_tensor.name}")
                continue
            q4nx_name = self.forward_name_map[gguf_tensor.name]

            if "token_embd.weight" in gguf_tensor.name:
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                factor = fold_factor_for(q4nx_name, folds)
                if factor is not None:
                    w = (w.float() * factor).to(torch.bfloat16)
                self.q4nx_tensors[q4nx_name] = w
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)

            if "q_proj" in q4nx_name or "k_proj" in q4nx_name:
                # llama.cpp's GraniteModel inherits LlamaModel, so its converter
                # applies Llama's q/k permutation and the GGUF carries the
                # INTERLEAVED pair order. attn.h rotates half-split pairs
                # (i, i + ROT/2), so the rows are put back the way HF stored
                # them -- the same rearrange llama.py does, for the same reason.
                # HunYuan needs no such step because its converter defers to the
                # base class; the difference is per-family and not guessable.
                from einops import rearrange

                pp = head_dim // 2
                d, m, qw = unpacked
                d = rearrange(d, '(g p q) c -> (g q p) c', p=pp, q=2).contiguous()
                m = rearrange(m, '(g p q) c -> (g q p) c', p=pp, q=2).contiguous()
                qw = rearrange(qw, '(g p q) c -> (g q p) c', p=pp, q=2).contiguous()
                unpacked = (d, m, qw)

            factor = fold_factor_for(q4nx_name, folds)
            if factor is not None:
                unpacked = scale_unpacked(unpacked, factor)

            self.q4nx_tensors[q4nx_name] = self._pack_q4nx(*unpacked)

        self._export_weights(q4nx_path, weights_type)
        self._extract_tokenizer_json(q4nx_path)
