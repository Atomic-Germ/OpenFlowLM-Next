r"""Q4_K-form weights and an exact reference for the GEMM_Q4K kernel.

gemv_q4's `w_qkv.bin` is q4_1-shaped: a bf16 scale and a bf16 min per 32-block
per row. Q4_K spends far less on that - one bf16 pair per 256 superblock per
row, plus a 6-bit sub-scale and 6-bit sub-min per 32-block - and a chunk here is
32 rows x 256 K, i.e. exactly one superblock per row. This script re-expresses
the same weights in that form:

    d[r]     = max_kb scale[kb][r] / 63      sc[kb][r] = round(scale / d)
    mcoef[r] = min_kb min_[kb][r] / 63       mn[kb][r] = round(min_ / mcoef)

and writes the 4736 B chunk the kernel reads (see the GEMM_Q4K note in
gemm_q4.h), in the same pool order as `w_qkv.bin`, plus the exact float64
product against `x_qkv.bin`.

The 6-bit rounding moves the weights slightly, so this is NOT ref_qkv.bin any
more - the reference is computed from the rounded values, which is what the
kernel is supposed to reproduce. The nibbles are copied through byte for byte.

    python make_q4k.py          # -> w_q4k_pool.bin, ref_q4k.bin
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"

N, K = 8192, 2048
BAND_ROWS, RS = 64, 2
BANDS = N // BAND_ROWS                     # 128
PER_BAND = RS * K // 256                   # 16 chunks per band
ROWS, KB, KT = 32, 8, 256
Q4NX_BYTES, Q4K_BYTES = 5120, 4736

# the mmul's lane order: evens 0,2..30 then odds 1,3..31 (see gemv_q4.h)
PERM = np.concatenate([np.arange(0, ROWS, 2), np.arange(1, ROWS, 2)])


def bf(a):
    return a.astype(bfloat16).astype(np.float64)


w = np.fromfile(GEMV / "w_qkv.bin", np.uint8)
assert w.size == BANDS * PER_BAND * Q4NX_BYTES, w.size
w = w.reshape(BANDS, PER_BAND, Q4NX_BYTES)
x = np.fromfile(GEMV / "x_qkv.bin", np.uint8).view(bfloat16).astype(np.float64)
assert x.size == K, x.size

out = np.zeros((BANDS, PER_BAND, Q4K_BYTES), np.uint8)
ref = np.zeros(N, np.float64)
n_pos_min = 0

for b in range(BANDS):
    blk = w[b]                                                   # [c][5120]
    d = np.ascontiguousarray(blk[:, :512]).view(bfloat16).astype(np.float64)
    m = np.ascontiguousarray(blk[:, 512:1024]).view(bfloat16).astype(np.float64)
    d = d.reshape(PER_BAND, KB, ROWS)                             # [c][kb][r]
    m = m.reshape(PER_BAND, KB, ROWS)

    # superblock scale / min coefficient, and the 6-bit sub-values off them
    dmax = d.max(axis=1)                                          # [c][r]
    mmin = m.min(axis=1)
    n_pos_min += int((m > 0).sum())
    dsup = np.where(dmax > 0, dmax / 63.0, 0.0)
    mco = np.where(mmin < 0, mmin / 63.0, 0.0)
    sc = np.where(dsup[:, None, :] != 0,
                  np.rint(d / np.where(dsup[:, None, :] != 0, dsup[:, None, :], 1.0)), 0.0)
    mn = np.where(mco[:, None, :] != 0,
                  np.rint(m / np.where(mco[:, None, :] != 0, mco[:, None, :], 1.0)), 0.0)
    sc = np.clip(sc, 0, 63).astype(np.uint8)
    mn = np.clip(mn, 0, 63).astype(np.uint8)
    dsup_b, mco_b = bf(dsup), bf(mco)                             # what the chunk stores

    # nibbles: index (r/16)*4096 + k*16 + (r%16), even index = low nibble
    nb = blk[:, 1024:]                                            # [c][4096]
    nib = np.stack([nb & 0x0F, nb >> 4], axis=-1).reshape(PER_BAND, 2, KT, 16)
    nib = nib.transpose(0, 1, 3, 2).reshape(PER_BAND, ROWS, KT)   # [c][r][k]

    # the values the kernel must reproduce; [c][kb][r] -> [c][r][k], kb = k/32
    scf = sc.transpose(0, 2, 1).repeat(32, axis=2).astype(np.float64)
    mnf = mn.transpose(0, 2, 1).repeat(32, axis=2).astype(np.float64)
    val = dsup_b[:, :, None] * scf * nib + mco_b[:, :, None] * mnf
    for c in range(PER_BAND):
        r0 = b * BAND_ROWS + (c % RS) * ROWS
        ref[r0:r0 + ROWS] += val[c] @ x[(c // RS) * KT:(c // RS) * KT + KT]

    o = out[b]
    o[:, 0:64] = np.ascontiguousarray(dsup_b[:, PERM].astype(bfloat16)).view(np.uint8)
    o[:, 64:128] = np.ascontiguousarray(mco_b[:, PERM].astype(bfloat16)).view(np.uint8)
    o[:, 128:384] = sc[:, :, PERM].reshape(PER_BAND, KB * ROWS)
    o[:, 384:640] = mn[:, :, PERM].reshape(PER_BAND, KB * ROWS)
    o[:, 640:] = nb

out.reshape(-1).tofile(HERE / "w_q4k_pool.bin")
ref.astype(np.float32).tofile(HERE / "ref_q4k.bin")
print(f"w_q4k_pool.bin {out.size} B ({Q4K_BYTES} B/chunk, "
      f"{100 * Q4K_BYTES / Q4NX_BYTES:.1f}% of q4_1's)")
print(f"ref_q4k.bin    {N} floats, |y| max {np.abs(ref).max():.4f}")
if n_pos_min:
    print(f"NOTE: {n_pos_min} blocks had a positive min, clipped to 0")
