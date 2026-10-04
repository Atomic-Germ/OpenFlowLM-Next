"""PrismML's Hadamard-rotated ternary GGUFs (Ternary Bonsai 2): the PQ2_0 type and the rotation.

A PQ2_0 block is 34 bytes for 128 values along the input (last) axis: an fp16 scale s, then
8 little-endian uint32 words, code lane i of a word at bits 2i (16 codes per word);
value = s*code - s, code in {0, 1, 2}. PrismML's runtime/codec.py is the reference. Stock
gguf-py does not know GGML type 142 and GGUFReader refuses the file; register_pq2_0() teaches
it the block geometry so the reader can map the tensors. Nothing here decodes through gguf-py.

Every listed PQ2_0 matmul expects its input rotated, y = W' @ fwht(x) with
fwht(x) = H(signs_w * x) / sqrt(B) per B-wide block of the input (normalized Sylvester
Walsh-Hadamard, B = prism.hadamard.block_size, one sign vector per input width w). The
embedding is the inverse, e = signs * H(row) / sqrt(B).

The open NPU kernels apply only a plain H / sqrt(B) to each projection input, plus explicit
signs on the attention / DeltaNet output (the 6144-wide inputs). Every other sign is folded
into the weights at conversion (PrismRotation.unpack), the lm head and the embedding are
un-rotated, and the 6144-wide signs travel in config.json for the engine to apply.
"""
from __future__ import annotations

import numpy as np
import torch
from gguf import GGMLQuantizationType, quantize
from gguf.constants import GGML_QUANT_SIZES

PQ2_0_ID = 142
QK = 128            # values per PQ2_0 block (one scale)
PQ2_BYTES = 34      # fp16 s + 32 bytes of 2-bit codes

# Tensors that read the 5120-wide hidden state through a norm whose gain absorbs s_hidden.
_NORMS_FOLDED = ("attn_norm.weight", "post_attention_norm.weight")
# Unrotated readers of that same norm output: their columns take s_hidden instead.
_COLS_FOLDED = ("ssm_alpha.weight", "ssm_beta.weight")


def register_pq2_0() -> GGMLQuantizationType:
    """Give gguf-py's type table a PQ2_0 member (id 142, 128 values in 34 bytes); idempotent."""
    try:
        t = GGMLQuantizationType(PQ2_0_ID)
    except ValueError:
        t = int.__new__(GGMLQuantizationType, PQ2_0_ID)
        t._name_, t._value_ = "PQ2_0", PQ2_0_ID
        GGMLQuantizationType._value2member_map_[PQ2_0_ID] = t
        GGMLQuantizationType._member_map_["PQ2_0"] = t
    if t.name != "PQ2_0":
        raise RuntimeError(f"gguf-py already defines type {PQ2_0_ID} as {t.name}, not PQ2_0")
    GGML_QUANT_SIZES.setdefault(t, (QK, PQ2_BYTES))
    return t


def decode_pq2(data: np.ndarray, rows: int, cols: int):
    """PQ2_0 bytes of an [rows, cols] matrix -> (codes uint8[rows, cols], s fp16[rows, cols//128])."""
    blocks = np.ascontiguousarray(data, dtype=np.uint8).reshape(rows, cols // QK, PQ2_BYTES)
    s = blocks[..., 0:2].copy().view("<f2").reshape(rows, cols // QK)
    words = blocks[..., 2:].copy().view("<u4").reshape(rows, cols // 16)
    codes = ((words[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3).astype(np.uint8)
    codes = codes.reshape(rows, cols)
    if (codes == 3).any():
        raise ValueError("PQ2_0 code 3 is not ternary")
    if not np.isfinite(s).all():
        raise ValueError("non-finite PQ2_0 scale")
    return codes, s


def dequant_pq2(codes: np.ndarray, s: np.ndarray, dtype=np.float64) -> np.ndarray:
    """value = s*code - s, in `dtype`."""
    sc = np.repeat(s.astype(dtype), QK, axis=1)
    return codes.astype(dtype) * sc - sc


def fwht(x: np.ndarray, block: int) -> np.ndarray:
    """Normalized Sylvester Walsh-Hadamard H / sqrt(block), blockwise along the last axis (float64)."""
    shape = x.shape
    y = np.asarray(x, dtype=np.float64).reshape(-1, block)
    h = 1
    while h < block:
        y = y.reshape(-1, block // (2 * h), 2, h)
        y = np.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], axis=2)
        h *= 2
    return y.reshape(shape) / np.sqrt(block)


def q8_0_tuple(w: np.ndarray):
    """A float [rows, cols] matrix -> unpack_q8_0's (d, d, qw), quantized by ggml's Q8_0 reference."""
    from .gguf_tensor import GGUFTensor
    w = np.ascontiguousarray(w, dtype=np.float32)
    return GGUFTensor.unpack_q8_0(quantize(w, GGMLQuantizationType.Q8_0), w.shape[1])


class PrismRotation:
    """The prism.hadamard.* contract of one GGUF, and the folds that put it into a q4nx container."""

    def __init__(self, reader):
        f = reader.fields

        def get(key, default=None):
            return f[key].contents() if key in f else default

        self.block = int(get("prism.hadamard.block_size"))
        transform = get("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
        if transform != "normalized-sylvester-walsh-hadamard":
            raise ValueError(f"unsupported prism.hadamard.transform {transform!r}")
        if get("prism.hadamard.sign_mode", "explicit") != "explicit":
            raise ValueError("prism.hadamard.sign_mode must be explicit")
        values = np.asarray(get("prism.hadamard.sign_values"), dtype=np.float32)
        self.signs, offset = {}, 0
        for width in (int(w) for w in get("prism.hadamard.sign_widths")):
            a = values[offset:offset + width]
            if len(a) != width or not np.isin(a, (-1.0, 1.0)).all() or width % self.block:
                raise ValueError(f"invalid prism.hadamard sign vector for width {width}")
            self.signs[width] = a
            offset += width
        if offset != len(values):
            raise ValueError("trailing prism.hadamard.sign_values")
        self.weight_names = set(get("prism.hadamard.weight_names", []))
        self.inverse_names = set(get("prism.hadamard.inverse_weight_names", []))
        if self.inverse_names != {"token_embd.weight"}:
            raise ValueError(f"unexpected prism.hadamard.inverse_weight_names {sorted(self.inverse_names)}")
        if "output.weight" not in self.weight_names:
            raise ValueError("expected a rotated output.weight (an untied lm head)")
        self.v_grouped = bool(get("prism.hadamard.gdn_v_grouped", False))
        self.hidden = int(get("qwen35.embedding_length"))
        self.ffn = int(get("qwen35.feed_forward_length"))
        self.s_hidden = self.signs[self.hidden]
        self.s_ffn = self.signs[self.ffn]
        rest = [w for w in self.signs if w not in (self.hidden, self.ffn)]
        if len(rest) != 1:
            raise ValueError(f"expected one attention-output sign width, got {rest}")
        self.out_width = rest[0]
        self.s_out = self.signs[self.out_width]

    @staticmethod
    def present(reader) -> bool:
        return "prism.hadamard.block_size" in reader.fields

    def config_entry(self) -> dict:
        """config.json's prism_hadamard: what the kernels still have to apply, and what was folded."""
        return {
            "block_size": self.block,
            "og_signs": [int(v) for v in self.s_out],
            "folded": "attn_norm,post_attention_norm,ssm_alpha_cols,ssm_beta_cols,ffn_up_rows",
            "lm_head": "unrotated",
            "embedding": "unrotated",
        }

    def unrotated_rows(self, t, rows_per_pass: int = 8192):
        """Yield (r0, r1, float64 rows) of a rotated [rows, hidden] PQ2_0 matrix in the plain
        basis: s_hidden * H(row) / sqrt(B) per block. That is the embedding's inverse transform,
        and equally the lm head as W' @ blockdiag(H / sqrt(B)) @ diag(s_hidden) (H is symmetric)."""
        cols = int(t.shape[0])
        rows = int(np.prod(t.shape[1:]))
        if cols != self.hidden:
            raise ValueError(f"{t.name}: input width {cols} is not the hidden size {self.hidden}")
        data = t.data.reshape(rows, -1)
        for r0 in range(0, rows, rows_per_pass):
            r1 = min(rows, r0 + rows_per_pass)
            codes, s = decode_pq2(data[r0:r1], r1 - r0, cols)
            yield r0, r1, fwht(dequant_pq2(codes, s), self.block) * self.s_hidden

    def embedding_rows(self, t):
        """token_embd, inverse-transformed, as row bands of the bf16 [vocab, hidden] the stock
        container stores."""
        for _, _, rows in self.unrotated_rows(t):
            yield torch.from_numpy(rows.astype(np.float32)).to(torch.bfloat16)

    def lm_head_q8_0_bands(self, t):
        """output.weight un-rotated to a dense matrix and ggml-Q8_0 quantized, as (d, d, qw) row
        bands. A band is 8192 rows, a multiple of the q4nx row block, so packing the bands one by
        one gives the packed tensor's own consecutive byte ranges."""
        for _, _, rows in self.unrotated_rows(t, rows_per_pass=8192):
            yield q8_0_tuple(rows)

    def stream_order(self, tensors: dict, out_name: dict):
        """(name, tensor) in the order a streamed container needs them: everything that is not a
        rotated matmul (small, and all bf16 / f32 / alpha-beta) first, then the embedding, then the
        packed matmuls by container name -- save_file's order for the big ones (q4nx/safetensors_stream.py)."""
        rot = [t for t in tensors.values() if t.tensor_type == PQ2_0_ID]
        small = [t for t in tensors.values() if t.tensor_type != PQ2_0_ID]
        embd = [t for t in rot if t.name in self.inverse_names]
        big = sorted((t for t in rot if t.name not in self.inverse_names), key=lambda t: out_name[t.name])
        return [(t.name, t) for t in small + embd + big]

    def unpack(self, t, target):
        """The (unpacked, tensor_type) the q4nx packer gets for one GGUF tensor, signs folded.

        PQ2_0 projections become q4_1 exactly (q = code, d = s, m = -s over each 128-group's four
        32-blocks), ffn_up with its rows flipped by s_ffn (code -> 2 - code). The two hidden-input
        norms take s_hidden in their gain, and alpha / beta, which read that norm output
        unrotated, take it in their columns. The lm head and the embedding are un-rotated row
        band by row band instead (lm_head_q8_0_bands, embedding_rows)."""
        name = t.name
        if t.tensor_type == PQ2_0_ID:
            if name not in self.weight_names:
                raise ValueError(f"{name} is PQ2_0 but not rotated; the kernels would rotate its input")
            if name in self.inverse_names or name == "output.weight":
                raise ValueError(f"{name} is un-rotated band by band, not unpacked whole")
            cols = int(t.shape[0])
            rows = int(np.prod(t.shape[1:]))
            codes, s = decode_pq2(t.data, rows, cols)
            if name.endswith("ffn_up.weight"):
                if rows != self.ffn:
                    raise ValueError(f"{name}: {rows} rows, expected {self.ffn}")
                codes = np.where((self.s_ffn < 0)[:, None], np.uint8(2) - codes, codes)
            d = torch.from_numpy(np.repeat(s.astype(np.float32), QK // 32, axis=1))
            return (d, -d, torch.from_numpy(codes.astype(np.float32))), GGMLQuantizationType.Q4_1
        if name.endswith(_NORMS_FOLDED):
            (w,) = t.unpack(target)
            return [w * torch.from_numpy(self.s_hidden)], target
        if name.endswith(_COLS_FOLDED):
            return q8_0_tuple(self.fold_cols(t.dequantize()).float().numpy()), GGMLQuantizationType.Q8_0
        return t.unpack(target), target

    def fold_cols(self, w: torch.Tensor) -> torch.Tensor:
        """Columns of an alpha / beta weight times s_hidden (exact: a sign flip)."""
        return w * torch.from_numpy(self.s_hidden).to(w.dtype)
