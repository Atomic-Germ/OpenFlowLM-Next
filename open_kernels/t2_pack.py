r"""PrismML ternary blocks (PQ2_0, Ternary Bonsai 2) -> the pool-order chunks the
open 2-bit GEMV reads, the same weights as q4_1 chunks, and the fp64 reference.

Probe 0a of .claude/plans/bonsai-2-ternary.md. A ``PQ2_0`` block is 34 bytes for
128 values along K: ``fp16 s | 8 little-endian u32 words``, code lane i of a word
at bits 2i (16 codes per word); value = s*code - s, code in {0, 1, 2}. PrismML's
``runtime/codec.py`` (prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) is the reference.

The 2-bit kernel (designs/gemv_t2/gemv_t2.h) reads 2176-byte chunks, each a
32-row x 256-col tile:

    s    [2][32] bf16  [0 : 128]       group g = (k % 256) / 128, row r
    code [8 kb][4 oc][8 kk][8 p] B  [128 : 2176]
         byte (kb, oc, kk, p), bits 2j..2j+1 = code(row 8j + p, k = 32 kb + 8 oc + kk)
    value = s * code - s

Field j of a byte is one of four rows at one k, so the integer matrix unit's B
operand ([8 k][8 rows]) is a 64 B block masked to one field -- the 2-bit twin of
q4_1_pack's two-rows-per-byte transpose. Chunk order is q4_1_pack's band law.

The value form is q4_1's d*q + m with m = -d, so the same weights also pack as
q4_1 chunks with q = code, d = s, m = -s (`as_q4_1_blocks`): exact, apart from the
fp16 -> bf16 scale narrowing both layouts take.

``python t2_pack.py`` self-tests the packers against each other and the decode.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

from q4_1_pack import BLK, Q4_1_BYTES, chunk_geometry, pack_q4_1_pool

QK = 128            # values per PQ2_0 block (one scale)
PQ2_BYTES = 34      # fp16 s, 32 bytes of 2-bit codes
CH_T2 = 2176        # 2-bit pool chunk bytes (32 rows x 256 k)


def random_pq2_codes(n: int, k: int, rng: np.random.Generator, scale: float = 0.02,
                     p_zero: float = 0.4):
    """(codes uint8[n, k] in {0,1,2}, s fp16[n, k//128]) for an [n, k] ternary matrix."""
    assert k % QK == 0
    p = (1 - p_zero) / 2
    codes = rng.choice(3, size=(n, k), p=[p, p_zero, p]).astype(np.uint8)
    s = (np.abs(rng.standard_normal((n, k // QK))) * scale + 0.1 * scale).astype(np.float16)
    return codes, s


def encode_pq2(codes: np.ndarray, s: np.ndarray) -> np.ndarray:
    """codes uint8[n, k], s fp16[n, k//128] -> PQ2_0 blocks uint8[n, k//128, 34]."""
    n, k = codes.shape
    w = codes.astype(np.uint32).reshape(n, k // 16, 16)
    words = np.bitwise_or.reduce(w << (2 * np.arange(16, dtype=np.uint32)), axis=-1).astype("<u4")
    blocks = np.empty((n, k // QK, PQ2_BYTES), np.uint8)
    blocks[..., 0:2] = np.ascontiguousarray(s.astype("<f2")).view(np.uint8).reshape(n, k // QK, 2)
    blocks[..., 2:] = words.reshape(n, k // QK, 8).view(np.uint8).reshape(n, k // QK, 32)
    return blocks


def decode_pq2(blocks: np.ndarray):
    """PQ2_0 blocks uint8[n, nb, 34] -> (codes uint8[n, nb*128], s fp16[n, nb])."""
    n, nb, _ = blocks.shape
    s = np.ascontiguousarray(blocks[..., 0:2]).view("<f2").reshape(n, nb)
    words = np.ascontiguousarray(blocks[..., 2:]).view("<u4").reshape(n, nb * 8)
    codes = (words[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
    return codes.astype(np.uint8).reshape(n, nb * QK), s


def dequant(codes: np.ndarray, s: np.ndarray, scale_dtype=np.float32) -> np.ndarray:
    """f32[n, k] = s*code - s. scale_dtype=bfloat16 reproduces what the kernels see."""
    sc = np.repeat(s.astype(np.float32).astype(scale_dtype).astype(np.float32), QK, axis=1)
    return codes.astype(np.float32) * sc - sc


def pack_t2_pool(codes: np.ndarray, s: np.ndarray, rs: int) -> np.ndarray:
    """codes uint8[n, k], s fp16[n, k//128] -> 2-bit pool chunks uint8[nch * 2176]."""
    n, k = codes.shape
    s16 = s.astype(np.float32).astype(bfloat16).view(np.uint16)
    nch, _, rows0, cols0 = chunk_geometry(n, k, rs)
    out = np.empty((nch, CH_T2), np.uint8)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        g0 = c0 // QK
        out[c, :128] = np.ascontiguousarray(s16[r0:r0 + 32, g0:g0 + 2].T).reshape(-1).view(np.uint8)
        # [32 r, 256 k] -> [j, p, k8, kk] (r = 8j + p, k = 8 k8 + kk, k8 = 4 kb + oc) -> [k8, kk, p, j]
        q = codes[r0:r0 + 32, c0:c0 + 256].reshape(4, 8, 32, 8).transpose(2, 3, 1, 0)
        out[c, 128:] = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).reshape(-1)
    return out.reshape(-1)


def dequant_t2_chunk(b: np.ndarray) -> np.ndarray:
    """One 2176 B chunk -> f32[32 rows, 256 k], read the way the kernel reads it."""
    s = b[:128].view(np.uint16).view(bfloat16).astype(np.float32).reshape(2, 32)    # [g, r]
    byte = b[128:].reshape(32, 8, 8)                                                # [k8, kk, p]
    codes = np.stack([(byte >> (2 * j)) & 3 for j in range(4)], axis=-1)            # [k8, kk, p, j]
    q = codes.transpose(3, 2, 0, 1).reshape(32, 256).astype(np.float32)             # [r, k]
    sc = np.repeat(s.T, QK, axis=1)
    return q * sc - sc


def dequant_t2_pool(pool: np.ndarray, n: int, k: int, rs: int) -> np.ndarray:
    nch, _, rows0, cols0 = chunk_geometry(n, k, rs)
    w = np.empty((n, k), np.float32)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        w[r0:r0 + 32, c0:c0 + 256] = dequant_t2_chunk(pool[c * CH_T2:(c + 1) * CH_T2])
    return w


def as_q4_1_blocks(codes: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The same weights as GGUF Q4_1 blocks: q = code, d = s, m = -s (exact)."""
    n, k = codes.shape
    nb = k // BLK
    d = np.repeat(s.astype(np.float16), QK // BLK, axis=1)
    q = codes.reshape(n, nb, BLK)
    blocks = np.empty((n, nb, Q4_1_BYTES), np.uint8)
    blocks[..., 0:2] = np.ascontiguousarray(d).view(np.uint8).reshape(n, nb, 2)
    blocks[..., 2:4] = np.ascontiguousarray(-d).view(np.uint8).reshape(n, nb, 2)
    blocks[..., 4:20] = q[..., :16] | (q[..., 16:] << 4)
    return blocks


def pack_q4_1_from_ternary(codes: np.ndarray, s: np.ndarray, rs: int) -> np.ndarray:
    return pack_q4_1_pool(as_q4_1_blocks(codes, s), rs)


def reference(codes: np.ndarray, s: np.ndarray, x: np.ndarray, rows: int = 2048) -> np.ndarray:
    """y = W @ x in fp64 with W as the kernels see it (bf16 scales); f32[n]."""
    xf = x.astype(np.float64)
    y = np.empty(codes.shape[0], np.float64)
    for r0 in range(0, codes.shape[0], rows):
        y[r0:r0 + rows] = dequant(codes[r0:r0 + rows], s[r0:r0 + rows], bfloat16).astype(np.float64) @ xf
    return y.astype(np.float32)


def _selftest() -> int:
    from q4_1_pack import dequant_pool
    rng = np.random.default_rng(0)
    ok = True
    for n, k, rs in [(512, 1024, 2), (256, 2048, 2), (512, 512, 4)]:
        codes, s = random_pq2_codes(n, k, rng)
        c2, s2 = decode_pq2(encode_pq2(codes, s))
        rt = np.array_equal(c2, codes) and np.array_equal(s2.view(np.uint16), s.view(np.uint16))
        want = dequant(codes, s, bfloat16)
        t2 = np.array_equal(dequant_t2_pool(pack_t2_pool(codes, s, rs), n, k, rs), want)
        q4 = np.array_equal(dequant_pool(pack_q4_1_from_ternary(codes, s, rs), n, k, rs), want)
        print(f"{'PASS' if rt and t2 and q4 else 'FAIL'} n={n} k={k} rs={rs}  pq2 round trip {rt}  "
              f"t2 pool == decode {t2}  q4_1 pool == decode {q4}")
        ok &= rt and t2 and q4
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
