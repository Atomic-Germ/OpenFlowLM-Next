"""The host's bf16 -> bfp16ebs8 conversion: NumPy twin of block_host.cpp's, checked against the
NPU's own (bfp_cvt.cc: accum<accfloat, 64> -> to_v64bfp16ebs8 under conv_even) by check.py.

A 64-value block vector is eight blocks of eight values. Each block keeps one 8-bit exponent E,
the largest biased exponent among its eight values, and one int8 mantissa per value: the value
divided by 2^(E - 127 - 6), rounded to nearest even, saturated to [-128, 127]. MODEL holds the
choices the hardware probe settles (the byte layout, the exponent field's offset, overflow)."""
from __future__ import annotations

import numpy as np

MODEL = {
    "layout": "exp_first",      # per block: [E][8 mantissas] (9 B); see check.py for the others
    "exp_offset": 0,            # stored exponent byte = E + exp_offset
    "overflow": "bump",         # a value rounding out of int8: the block's exponent + 1, re-rounded
}


def _mantissas(b: np.ndarray, e: np.ndarray, E: np.ndarray) -> np.ndarray:
    """signed mantissas (int64, unsaturated) of bf16 bits b with exponents e against block exponent E"""
    sig = np.where(e > 0, (b & 0x7F) | 0x80, 0)
    shift = np.minimum(E - e + 1, 9)
    q = sig >> shift
    rem = sig & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    q = q + ((rem > half) | ((rem == half) & ((q & 1) == 1))).astype(np.int64)
    return np.where(b & 0x8000, -q, q)


def bf16_to_bfp16_v64(lanes: np.ndarray, model: dict | None = None) -> np.ndarray:
    """lanes [n, 64] uint16 bf16 bits (block b = lanes 8b..8b+7) -> [n, 72] uint8."""
    m = MODEL if model is None else model
    b = lanes.reshape(-1, 8, 8).astype(np.int64)
    e = (b >> 7) & 0xFF
    E = e.max(axis=2, keepdims=True)
    q = _mantissas(b, e, E)
    if m["overflow"] in ("bump", "bump_sym"):
        # a value that rounds out of int8 moves the whole block up one exponent and re-rounds
        lo = -128 if m["overflow"] == "bump" else -127
        over = ((q > 127) | (q < lo)).any(axis=2, keepdims=True)
        E = E + over
        q = np.where(over, _mantissas(b, e, E), q)
    elif m["overflow"] == "saturate":
        q = np.clip(q, -127, 127)
    q = np.clip(q, -128, 127).astype(np.int8).view(np.uint8)
    ex = (E[..., 0] + m["exp_offset"]).astype(np.uint8)                 # [n, 8]
    n = q.shape[0]
    if m["layout"] == "exp_first":
        return np.concatenate([ex[..., None], q], axis=2).reshape(n, 72)
    if m["layout"] == "exp_last":
        return np.concatenate([q, ex[..., None]], axis=2).reshape(n, 72)
    if m["layout"] == "mant_then_exp":
        return np.concatenate([q.reshape(n, 64), ex], axis=1)
    if m["layout"] == "exp_then_mant":
        return np.concatenate([ex, q.reshape(n, 64)], axis=1)
    raise ValueError(m["layout"])
