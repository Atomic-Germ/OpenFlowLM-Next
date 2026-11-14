"""The fp64 oracle for the attention gate.

Independent of the kernel: it recomputes what the attention core must produce
from the same inputs, in float64, with the kernel's roundings written down
where they matter (the q/k norm, the rotation, the bf16 cache rows, the
softmax). It is the yardstick compare.py measures the device against, so it
must not share code with the kernel -- only the layout.
"""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


def rms_norm(x: np.ndarray, eps: float) -> np.ndarray:
    """The q/k RMSNorm: one RMS over the whole head, then the model's eps."""
    x = np.asarray(x, np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def norm_rope(x: np.ndarray, w: np.ndarray, cs: np.ndarray, eps: float,
              qk_norm: bool, post: bool, interleaved: bool = False,
              pair: int | None = None) -> np.ndarray:
    """rms(q) * w, then the rotation over the first rotary_dim dims.

    `cs` is [cos(ROT/2) | sin(ROT/2)] as the host sends it -- pair-ordered under
    both pairing laws. Half-split (the default): the pair (j, j + ROT/2) turns
    by the angle at j. Interleaved (llama.cpp's NORM archs, ATTN_ROPE_IL): the
    pair (2j, 2j + 1) turns by the angle at j. `pair` overrides the partner
    distance ONLY for the half-split law: Gemma 4's proportional RoPE turns the
    first ROT/2 pairs, but pairs dim j with j + pair (ATTN_ROPE_PAIR), because
    HF rotates the whole head with inv_freq zero past the first ROT/2 (identity
    there) while the record holds only ROT/2 cos / sin.
    """
    y = np.asarray(x, np.float64).copy()
    if qk_norm and not post:
        y = rms_norm(y, eps) * np.asarray(w, np.float64)
    rot = len(cs)                     # cs is [cos(ROT/2) | sin(ROT/2)]: ROT entries
    half = rot // 2                   # pair j turns by the angle at j, either law
    step = pair if (pair is not None and pair > 0) else half
    c = cs[:half].astype(np.float64)
    s = cs[half:].astype(np.float64)
    if interleaved:
        a = y[..., 0:rot:2].copy()
        b = y[..., 1:rot:2].copy()
        y[..., 0:rot:2] = a * c - b * s
        y[..., 1:rot:2] = b * c + a * s
    else:
        for j in range(half):
            a, b = j, j + step
            ta, tb = y[..., a].copy(), y[..., b].copy()
            y[..., a] = ta * c[j] - tb * s[j]
            y[..., b] = tb * c[j] + ta * s[j]
    if qk_norm and post:
        y = y * np.asarray(w, np.float64)
    return y


def cos_sin(pos: int, rot: int, theta: float) -> np.ndarray:
    """The host's position record: [cos(ROT/2) | sin(ROT/2)] at one position."""
    half = rot // 2
    ang = pos * np.asarray(theta, np.float64) ** (-np.arange(half) / half)
    return np.concatenate([np.cos(ang), np.sin(ang)]).astype(np.float32)


def decode(q: np.ndarray, k: np.ndarray, v: np.ndarray, qn: np.ndarray, kn: np.ndarray,
           cs: np.ndarray, cache_k: np.ndarray, cache_v: np.ndarray, *,
           nh: int, kvh: int, hd: int, eps: float, qk_norm: bool, post: bool,
           interleaved: bool = False, pair: int | None = None,
           v_norm: bool = False, attn_scale: float | None = None):
    """The attention step: the new position's rows plus the cached window.

    `cache_k` / `cache_v` are the window's bf16 rows ([rows, KVH, HD]). The
    result is (the new k' | v' rows, the attention output [NH, HD]).

    The new k' row is what the kernel writes into the cache, so it is computed
    here the same way and rounded to bf16 -- the kernel's own row is the input
    of the score, not a separate value.
    """
    qp = norm_rope(q, qn, cs, eps, qk_norm, post, interleaved, pair)     # [NH, HD]
    kp = norm_rope(k, kn, cs, eps, qk_norm, post, interleaved, pair).astype(bfloat16)  # [KVH, HD]
    if v_norm:
        # Gemma 4's v_norm: an RMSNorm over the head with NO weight (HF
        # Gemma4RMSNorm(with_scale=False)), per head, eps shared with q/k.
        vp = rms_norm(v, eps).astype(bfloat16)                           # [KVH, HD]
    else:
        vp = np.asarray(v, np.float32).astype(bfloat16)                  # [KVH, HD]
    rows_k = np.concatenate([np.asarray(cache_k, np.float64), kp[None].astype(np.float64)], 0)
    rows_v = np.concatenate([np.asarray(cache_v, np.float64), vp[None].astype(np.float64)], 0)
    grp = nh // kvh
    og = np.zeros((nh, hd), np.float64)
    scale = attn_scale if attn_scale is not None else 1.0 / np.sqrt(hd)
    for h in range(nh):
        kv = h // grp
        s = rows_k[:, kv] @ qp[h] * scale
        a = np.exp(s - s.max())
        p = a / a.sum()
        og[h] = p @ rows_v[:, kv]
    return np.stack([kp, vp]), og


def metric(got: np.ndarray, ref: np.ndarray, tol: float) -> dict:
    """The pass rule: relative error against the reference's own scale, and the
    cosine (a large outlier must not hide behind a small denominator)."""
    g = np.asarray(got, np.float64).ravel()
    r = np.asarray(ref, np.float64).ravel()
    if g.shape != r.shape:
        return {"passed": False, "error": "shape mismatch"}
    if not np.isfinite(g).all() or not np.isfinite(r).all():
        return {"passed": False, "error": "non-finite"}
    gs, rs = np.max(np.abs(g)), np.max(np.abs(r))
    err = float(np.max(np.abs(g - r)))
    rel = float(err / rs) if rs else (0.0 if err == 0 else None)
    if np.array_equal(g, r):
        cos = 1.0
    elif not gs or not rs:
        cos = 0.0
    else:
        a, b = g / gs, r / rs
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"passed": rel is not None and rel < tol and cos > 0.9999,
            "maxrel": rel, "cosine": cos}
