r"""Bit-exact model of wht_int.h and of wht.h + gemv_t2_prep_groups, down to the t2 table (OPEN-HADAMARD).

    python fwht_ref.py            # error report on synthetic inputs
"""
from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

BLOCK = 1024


def sylvester(n: int) -> np.ndarray:
    h = np.ones((1, 1), np.int64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


H32 = sylvester(32)
H1024 = sylvester(1024)


def rne_shift(v: np.ndarray, r: int) -> np.ndarray:
    """round-half-even of v / 2^r for int64 v (srs with conv_even)."""
    if r <= 0:
        return v << (-r)
    q = v >> r
    rem = v - (q << r)
    half = 1 << (r - 1)
    up = (rem > half) | ((rem == half) & ((q & 1) == 1))
    return q + up


def bf16(x) -> np.ndarray:
    return np.asarray(x, np.float32).astype(bfloat16)


def _exp_bits(x: np.ndarray) -> int:
    if x.dtype == bfloat16:
        return int((x.view(np.uint16) & 0x7FFF).max()) >> 7
    return int((x.astype(np.float32).view(np.uint32) & 0x7FFFFFFF).max()) >> 23


def _sum_hi_lo(s32: np.float32) -> tuple[np.ndarray, np.ndarray]:
    hi = bf16(s32)
    lo = bf16(np.float32(s32) - hi.astype(np.float32))
    return hi, lo


def int_block(x: np.ndarray):
    """One block (bf16 or fp32, 1024 values) -> (xi[1024] int16, sh[8], hi[8], lo[8]) as wht_int.h does."""
    a = 141 - _exp_bits(x)
    if x.dtype != bfloat16:             # an fp32 max that would round up to 2^15 takes a - 1
        bits = int((x.astype(np.float32).view(np.uint32) & 0x7FFFFFFF).max())
        a -= int((bits & 0x7FFFFF) >= 0x7FFF00)
    a = min(a, 126)
    xf = x.astype(np.float64) * np.float64(2.0) ** a
    X = np.clip(np.round(xf), -32768, 32767).astype(np.int64)        # np.round is half-even
    T = X.reshape(32, 32) @ H32                                       # pass A, exact in int32
    s1 = max(0, int(np.abs(T).max()).bit_length() - 15)
    T16 = np.clip(rne_shift(T, s1), -32768, 32767)
    Y = H32 @ T16                                                     # pass B, exact
    xi = np.empty((32, 32), np.int64)
    sh = np.empty(8, np.int32)
    hi = np.empty(8, bfloat16)
    lo = np.empty(8, bfloat16)
    for g in range(8):
        yg = Y[4 * g:4 * g + 4]
        r = max(0, int(np.abs(yg).max()).bit_length() - 15)
        q = np.clip(rne_shift(yg, r), -32768, 32767)
        xi[4 * g:4 * g + 4] = q
        sh[g] = a + 5 - s1 - r
        s = np.float32(np.float64(yg.sum()) * np.float64(2.0) ** (s1 - a - 5))   # sum of Y before the rounding
        hi[g], lo[g] = _sum_hi_lo(s)
    return xi.reshape(-1).astype(np.int16), sh, hi, lo


def int_tables(x: np.ndarray):
    parts = [int_block(x[i:i + BLOCK]) for i in range(0, x.size, BLOCK)]
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))


def fwht_f32_blocks(x: np.ndarray) -> np.ndarray:
    """wht.h's butterflies in fp32, stage order 16, 32, ..., 512 then 1, 2, 4, 8; then the 1/32 and bf16."""
    out = []
    for i in range(0, x.size, BLOCK):
        v = x[i:i + BLOCK].astype(np.float32).copy()
        for t in (16, 32, 64, 128, 256, 512, 1, 2, 4, 8):
            v = v.reshape(-1, 2, t)
            a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :], v[:, 1, :] = a + b, a - b
            v = v.reshape(-1)
        out.append(bf16(v).astype(np.float32) * np.float32(0.03125))
    return bf16(np.concatenate(out))


def prep_groups(xb: np.ndarray):
    """gemv_t2_prep_groups on a bf16 vector."""
    ng = xb.size // 128
    xi = np.empty(xb.size, np.int16)
    sh = np.empty(ng, np.int32)
    hi = np.empty(ng, bfloat16)
    lo = np.empty(ng, bfloat16)
    for g in range(ng):
        v = xb[128 * g:128 * g + 128]
        s = min(141 - (int((v.view(np.uint16) & 0x7FFF).max()) >> 7), 126)
        sh[g] = s
        xi[128 * g:128 * g + 128] = np.round(v.astype(np.float64) * 2.0 ** s).astype(np.int16)
        hi[g], lo[g] = _sum_hi_lo(np.float32(v.astype(np.float32).sum(dtype=np.float32)))
    return xi, sh, hi, lo


def fp_tables(x: np.ndarray):
    return prep_groups(fwht_f32_blocks(x))


def effective(tab) -> tuple[np.ndarray, np.ndarray]:
    xi, sh, hi, lo = tab
    v = xi.astype(np.float64) * np.repeat(2.0 ** (-sh.astype(np.float64)), 128)
    return v, hi.astype(np.float64) + lo.astype(np.float64)


def exact(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float64).reshape(-1, BLOCK) @ H1024.T).reshape(-1) / 32.0


def errors(x: np.ndarray, tab, w: np.ndarray | None = None) -> dict:
    """Errors against fp64 H x / 32; with ternary rows w, also the GEMV's relative RMS error."""
    z = exact(x)
    v, gs = effective(tab)
    d = {"rms": float(np.linalg.norm(v - z) / np.linalg.norm(z)),
         "max": float(np.abs(v - z).max() / np.abs(z).max())}
    if w is not None:
        yt = w @ z
        code = (w + 1).astype(np.float64)
        ye = code @ v - np.tile(gs.reshape(1, -1), (w.shape[0], 1)).sum(axis=1)
        d["gemv"] = float(np.linalg.norm(ye - yt) / np.linalg.norm(yt))
    return d


def _synth(rng, kind: str, k: int) -> np.ndarray:
    if kind == "gauss_bf16":
        return bf16(rng.standard_normal(k))
    if kind == "outlier_bf16":         # residual-like outlier channels
        x = rng.standard_normal(k)
        x[rng.choice(k, 8, replace=False)] *= 50
        x[rng.integers(k)] *= 1000
        return bf16(x)
    if kind == "silu_f32":             # silu(g) * u, the down projection's fp32 input
        g, u = rng.standard_normal(k) * 2, rng.standard_normal(k)
        return (g / (1 + np.exp(-g)) * u).astype(np.float32)
    if kind == "outlier_f32":
        x = rng.standard_normal(k)
        x[rng.integers(k)] *= 3000
        return x.astype(np.float32)
    raise ValueError(kind)


def main() -> int:
    rng = np.random.default_rng(0)
    k = 5120
    w = rng.integers(-1, 2, size=(256, k))
    print(f"{'input':14s} {'path':5s} {'rel rms':>10s} {'max/max':>10s} {'gemv rel':>10s}")
    for kind in ("gauss_bf16", "outlier_bf16", "silu_f32", "outlier_f32"):
        x = _synth(rng, kind, k)
        for name, tab in (("fp", fp_tables(x)), ("int", int_tables(x))):
            e = errors(x, tab, w)
            print(f"{kind:14s} {name:5s} {e['rms']:10.2e} {e['max']:10.2e} {e['gemv']:10.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
