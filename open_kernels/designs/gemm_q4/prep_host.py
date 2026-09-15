r"""Host-side mirror of gemm_q4_prep_il's GEMM_Q4K branch: produces the exact
byte table the core USED to compute on every (tile, k-tile, token) via
gemm_q4_prep_slice_m{M} -> gemm_q4_prep_il. With GEMM_PREQ=1,
gemm_q4_copy_group_m{M} just lands these bytes in `tab` - no arithmetic on the
core at all, only a DMA and a copy.

Reproduces gemm_q4_prep_il bit-for-bit except for the block-sum tree (folded
here as a plain float32 sum rather than the core's power-of-two fold order,
which the accuracy check absorbs - see check.py's q4k bar):

  - one shift per 256-wide k-tile slice (not per 32-block): the bf16 bit
    pattern with its sign cleared orders the same as the float magnitude, so
    max|x| over the slice is `reduce_max` of that, and s = 141 - (mx >> 7),
    clamped to [Q4K_SRS, 126] (Q4K_SRS = gemm_q4.h's kQ4KSrs, the int16 SRS
    depth the tile body narrows the mmul partial by).
  - xi = round_half_even(bf16(x) * 2^s), saturated to int16. Multiplying a
    bf16 value by an exact power of two is exact in bf16 (only the exponent
    moves), so the only rounding step left is the final int16 SRS, and
    numpy's `rint` is round-half-to-even like the core's `conv_even` mode -
    this matches the on-core arithmetic exactly, not just approximately.
  - block sums (still per 32-block - Q4_K's min term needs that granularity)
    as bf16 hi/lo, one pair per 32-block.

Table layout per 4-token group, one 256-wide k-tile (kILGroupBytes = 2304 B,
gemm_q4.h's "Token-interleaved activation table" note):
    int16 xi[32][4][8]   at    0   (octet = k/8, then token, then k)
    int32 s [4][8]       at 2048
    bf16  xs_hi[4][8]    at 2176
    bf16  xs_lo[4][8]    at 2240

Blob layout for the whole run: [n_kt][M/4][kILGroupBytes], the order
gemm_q4.py's core_body reads it in when GEMM_PREQ=1 (x_tap=None, so a straight
contiguous fill - the host is responsible for getting this order right).

    python prep_host.py --m 24     # -> x_preq_m24.bin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"

K = 2048
KT = 256          # k-tile width
KB = 8            # 32-wide blocks per k-tile
GROUP = 4         # tokens per IL group
Q4K_SRS = 9       # gemm_q4.h: kQ4KSrs


def quantise_slice(x256: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """One token's 256-wide k-tile slice (float64) -> (xi[8][32] int16, s (one
    shift for the whole slice), xsh[8] bf16, xsl[8] bf16)."""
    xb16 = x256.astype(bfloat16)
    bits = xb16.view(np.uint16).astype(np.int32)
    absbits = bits & 0x7FFF               # clear sign - positive floats order as their bits
    s = 141 - (int(absbits.max()) >> 7)
    s = min(126, s)
    s = max(Q4K_SRS, s)

    xf = xb16.astype(np.float64).reshape(KB, 32)
    q = np.clip(np.rint(xf * (2.0 ** s)), -32768, 32767).astype(np.int16)   # [8][32]

    xsh = np.zeros(KB, dtype=bfloat16)
    xsl = np.zeros(KB, dtype=bfloat16)
    for kb in range(KB):
        total = np.float32(xf[kb].astype(np.float32).sum())
        hi = total.astype(bfloat16)
        lo = bfloat16(total - np.float32(hi))
        xsh[kb] = hi
        xsl[kb] = lo
    return q, s, xsh, xsl


def pack_group(xg: np.ndarray) -> bytes:
    """xg: [4][256] float64, one 4-token group's 256-wide slice ->
    kILGroupBytes bytes in gemm_q4.h's documented layout."""
    xi = np.zeros((32, GROUP, 8), dtype=np.int16)      # [octet][t][8]
    sh = np.zeros((GROUP, KB), dtype=np.int32)
    xsh = np.zeros((GROUP, KB), dtype=bfloat16)
    xsl = np.zeros((GROUP, KB), dtype=bfloat16)
    for t in range(GROUP):
        q, s, h, l = quantise_slice(xg[t])
        for kb in range(KB):
            for j in range(4):
                xi[kb * 4 + j, t, :] = q[kb, j * 8:(j + 1) * 8]
        sh[t, :] = s
        xsh[t, :] = h
        xsl[t, :] = l
    return xi.tobytes() + sh.tobytes() + xsh.tobytes() + xsl.tobytes()


def build(m: int, xfile: Path | None = None) -> Path:
    assert m % GROUP == 0, m
    x1 = np.fromfile(xfile or GEMV / "x_qkv.bin", np.uint8).view(bfloat16).astype(np.float64)
    assert x1.size == K, x1.size
    # every token gets the SAME activation, exactly like make_test.py's x_m{M}.bin
    x = np.tile(x1, (m, 1))                             # [M][K]

    n_kt = K // KT
    out = bytearray()
    for kt in range(n_kt):
        xslice = x[:, kt * KT:(kt + 1) * KT]             # [M][256]
        for gi in range(m // GROUP):
            out += pack_group(xslice[gi * GROUP:(gi + 1) * GROUP])

    path = HERE / f"x_preq_m{m}.bin"
    path.write_bytes(bytes(out))
    print(f"wrote {path.name} ({len(out)} B)  [{n_kt} k-tiles x {m // GROUP} groups x "
          f"{len(out) // (n_kt * (m // GROUP))} B/group]")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--xfile", default=None, help="bf16 x[K] source (default: gemv_q4/x_qkv.bin)")
    a = ap.parse_args()
    build(a.m, Path(a.xfile) if a.xfile else None)
