"""The two things the LFM references need that this tree's GGUF reader and dense
reference do not provide: the NPU's bf16 rounding, attention with learned sinks
and a sliding window, and a whole-tensor accessor over `gguf_ref.Gguf`.

`gguf_ref.Gguf` hands out one row range at a time (`dequant(name, row0, rows)`);
the LFM references want a tensor whole (`array`), an embedding lookup by id
(`rows`), and one routed expert of a stacked tensor (`expert`) -- and, for the
`--npu-like` control, the block scales rounded to bf16 the way the pool holds
them. The math (softmax with sinks, the sliding mask, the bf16 round) is the
published model's, not this tree's recipe.
"""
from __future__ import annotations

import numpy as np

from gguf_ref import (BF16, F16, F32, Q4_0, Q4_1, Q6_K, Q8_0, Gguf)

# block byte counts (per 32 values; Q6_K is per 256) for the narrow re-read
_NARROW_BYTES = {Q8_0: 34, Q4_0: 18, Q4_1: 20}


def bf(x):
    """Round to bf16 (round to nearest even) and back, in f32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def _decode_narrow(raw: np.ndarray, ty: int) -> np.ndarray:
    """Q8_0 / Q4_0 / Q4_1 blocks with the block scale (and min) rounded to bf16
    first -- what the NPU's q4_1 pool chunks hold (pools.cpp std_perm_gguf)."""
    b = np.asarray(raw).reshape(-1, _NARROW_BYTES[ty])
    d = bf(b[:, 0:2].copy().view(np.float16).astype(np.float32))
    if ty == Q8_0:
        return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)
    if ty == Q4_1:
        m = bf(b[:, 2:4].copy().view(np.float16).astype(np.float32))
        qs = b[:, 4:]
    else:                                        # Q4_0: the min is -8 d
        m = -8.0 * d
        qs = b[:, 2:]
    q = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32)   # value j | j + 16
    return (q * d + m).reshape(-1)


class LfmGguf(Gguf):
    """`gguf_ref.Gguf` with the whole-tensor accessors the LFM references use."""

    def array(self, name: str, narrow: bool = False) -> np.ndarray:
        """The whole tensor as f32, row-major (dims reversed)."""
        ty, dims, _ = self.tensor(name)
        shape = tuple(reversed(dims))
        rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        if narrow and ty in _NARROW_BYTES:
            _, ne0, raw, _ = self.raw(name, 0, rows)
            return _decode_narrow(raw, ty).reshape(shape)
        return self.dequant(name, 0, rows).reshape(shape)

    def rows(self, name: str, ids) -> np.ndarray:
        """Rows `ids` of a [rows, cols] tensor (the embedding lookup)."""
        ids = list(ids)
        ty, dims, _ = self.tensor(name)
        cols = int(dims[0])
        out = np.empty((len(ids), cols), np.float32)
        for i, r in enumerate(ids):
            out[i] = self.dequant(name, int(r), 1)[0]
        return out

    def expert(self, name: str, e: int, narrow: bool = False) -> np.ndarray:
        """Expert `e` of a [experts, rows, cols] tensor as f32 [rows, cols]."""
        ty, dims, _ = self.tensor(name)
        cols, rows = int(dims[0]), int(dims[1])
        if narrow and ty in _NARROW_BYTES:
            _, ne0, raw, _ = self.raw(name, e * rows, rows)
            return _decode_narrow(raw, ty).reshape(rows, cols)
        return self.dequant(name, e * rows, rows)


def rope(x: np.ndarray, pos: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Half-split (rotate_half) rotation per head: x [T, H, hd]. `inv` is the
    progression over the full head dim (hd / 2 angles); a zero entry leaves that
    dimension unrotated (LFM3's partial-rotary zero tail)."""
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    cos = np.cos(ang).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def attention(q, k, v, sinks=None, window: int = 0):
    """q [T, H, hd], k / v [T, KVH, hd]: causal; `window` rows (0 = all) -- query t
    sees keys (t - window, t], so 513 means itself and the 512 before it; learned
    sinks (None = none) join each head's softmax with no value vector."""
    T, H, hd = q.shape
    kvh = k.shape[1]
    g = H // kvh
    out = np.empty((T, H, hd), np.float32)
    scale = hd ** -0.5
    for t in range(T):
        s0 = max(0, t - window + 1) if window else 0
        kk, vv = k[s0:t + 1], v[s0:t + 1]                       # [L, KVH, hd]
        qh = q[t].reshape(kvh, g, hd)
        sc = np.einsum("kgd,lkd->kgl", qh.astype(np.float64), kk.astype(np.float64)) * scale
        if sinks is None:
            m = sc.max(axis=-1, keepdims=True)
            p = np.exp(sc - m)
            den = p.sum(axis=-1, keepdims=True)
        else:
            sk = sinks.reshape(kvh, g, 1).astype(np.float64)
            m = np.maximum(sc.max(axis=-1, keepdims=True), sk)
            p = np.exp(sc - m)
            den = p.sum(axis=-1, keepdims=True) + np.exp(sk - m)
        out[t] = (np.einsum("kgl,lkd->kgd", p / den, vv.astype(np.float64))).reshape(H, hd)
    return out


def silu(x):
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


__all__ = ["LfmGguf", "attention", "bf", "rope", "silu"]
