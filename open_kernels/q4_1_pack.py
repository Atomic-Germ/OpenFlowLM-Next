r"""GGUF Q4_1 blocks -> the pool-order chunk layout the open q4 GEMV reads.

This is the seed of the GGUF -> device-pool packer. A GGUF ``Q4_1`` block is
20 bytes for 32 values along K: ``fp16 d | fp16 m | 16 nibble bytes`` (byte j
holds value j in its low nibble and value j+16 in its high nibble). The kernel
reads 5120-byte chunks, each a 32-row x 256-col tile:

    d  [256] bf16   [0    : 512]     index j = kb*32 + r
    m  [256] bf16   [512  : 1024]    same index
    nib[4096] B     [1024 : 5120]    nibble p = (r/16)*4096 + k*16 + (r%16),
                                     even p = low nibble
    value = nib * d[j] + m[j]

Same nibbles, same per-32 scale-and-min, same arithmetic: the transform is a
byte permutation plus fp16 -> bf16 on d/m. Nothing is re-quantized. The one
thing that is not a no-op is the nibble transpose — GGUF packs two K values per
byte for one row, the kernel wants two ROWS per byte at one K, because the
integer matrix unit's B operand is [8 k][8 row pairs].

Chunk (pool) order for a [N, K] matrix with band row split RS (2 for the
standard layout, 4 for expert stripes; see gemv_q4.h):

    per_band = RS * K / 256 chunks; chunk c: band = c // per_band, ci = c % per_band
    rows0 = 32*RS*band + 32*(ci % RS),  cols0 = 256*(ci // RS)

``python q4_1_pack.py`` self-tests: random blocks -> pack -> dequant of the pool
bytes must equal dequant of the GGUF blocks (with d/m narrowed to bf16) exactly.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

CH = 5120           # pool chunk bytes (32 rows x 256 k)
BLK = 32            # values per Q4_1 block
Q4_1_BYTES = 20     # fp16 d, fp16 m, 16 nibble bytes


def random_q4_1_blocks(n: int, k: int, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    """GGUF-style Q4_1 blocks for an [n, k] matrix: uint8[n, k//32, 20]."""
    assert k % BLK == 0
    nb = k // BLK
    d = (rng.random((n, nb), np.float32) * scale + 1e-3).astype(np.float16)
    m = ((rng.random((n, nb), np.float32) - 0.5) * scale * 15).astype(np.float16)
    q = rng.integers(0, 16, (n, nb, BLK), dtype=np.uint8)
    blocks = np.empty((n, nb, Q4_1_BYTES), np.uint8)
    blocks[..., 0:2] = d.view(np.uint8).reshape(n, nb, 2)
    blocks[..., 2:4] = m.view(np.uint8).reshape(n, nb, 2)
    blocks[..., 4:20] = q[..., :16] | (q[..., 16:] << 4)
    return blocks


def unpack_q4_1_blocks(blocks: np.ndarray):
    """uint8[n, nb, 20] -> (d f32[n, nb], m f32[n, nb], q uint8[n, nb*32])."""
    n, nb, _ = blocks.shape
    d = np.ascontiguousarray(blocks[..., 0:2]).view(np.float16).reshape(n, nb).astype(np.float32)
    m = np.ascontiguousarray(blocks[..., 2:4]).view(np.float16).reshape(n, nb).astype(np.float32)
    qs = blocks[..., 4:20]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(n, nb * BLK)
    return d, m, q


def dequant_q4_1(blocks: np.ndarray, scale_dtype=np.float32) -> np.ndarray:
    """GGUF blocks -> f32[n, k]. scale_dtype=bfloat16 reproduces what the kernel sees."""
    d, m, q = unpack_q4_1_blocks(blocks)
    d = d.astype(scale_dtype).astype(np.float32)
    m = m.astype(scale_dtype).astype(np.float32)
    return q.astype(np.float32) * np.repeat(d, BLK, axis=1) + np.repeat(m, BLK, axis=1)


def chunk_geometry(n: int, k: int, rs: int):
    """(nch, per_band, rows0[nch], cols0[nch]) for the pool order."""
    assert k % 256 == 0 and n % (32 * rs) == 0
    per_band = rs * k // 256
    nch = n * k // (32 * 256)
    c = np.arange(nch)
    band, ci = np.divmod(c, per_band)
    rows0 = 32 * rs * band + 32 * (ci % rs)
    cols0 = 256 * (ci // rs)
    return nch, per_band, rows0, cols0


def pack_q4_1_pool(blocks: np.ndarray, rs: int) -> np.ndarray:
    """GGUF Q4_1 blocks uint8[n, k//32, 20] -> pool-order chunk bytes uint8[nch*5120]."""
    n, nb, _ = blocks.shape
    k = nb * BLK
    d, m, q = unpack_q4_1_blocks(blocks)
    d16 = d.astype(bfloat16).view(np.uint16)
    m16 = m.astype(bfloat16).view(np.uint16)
    nch, _, rows0, cols0 = chunk_geometry(n, k, rs)
    out = np.empty((nch, CH), np.uint8)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        kb0 = c0 // BLK
        # d/m: [32 r, 8 kb] -> index kb*32 + r  ->  [8 kb, 32 r] row-major
        out[c, 0:512] = d16[r0:r0 + 32, kb0:kb0 + 8].T.reshape(-1).view(np.uint8)
        out[c, 512:1024] = m16[r0:r0 + 32, kb0:kb0 + 8].T.reshape(-1).view(np.uint8)
        # nibbles: [32 r, 256 k] -> [rb, k, r16] with rb = r // 16, then pairs -> bytes
        qq = q[r0:r0 + 32, c0:c0 + 256].reshape(2, 16, 256).transpose(0, 2, 1).reshape(-1)
        out[c, 1024:] = qq[0::2] | (qq[1::2] << 4)
    return out.reshape(-1)


def dequant_chunk(b: np.ndarray) -> np.ndarray:
    """One 5120 B pool chunk -> f32[32 rows, 256 k] (mirror of q4nx.rs dequant_q4_bytes)."""
    d = b[0:512].view(bfloat16).astype(np.float32)        # index kb*32 + r
    m = b[512:1024].view(bfloat16).astype(np.float32)
    qb = b[1024:]
    nib = np.empty(8192, np.uint8)
    nib[0::2] = qb & 0xF
    nib[1::2] = qb >> 4
    n3 = nib.reshape(2, 256, 16)                          # [rb, k, r16]
    w = np.empty((32, 256), np.float32)
    kb = np.arange(256) // BLK
    for rb in range(2):
        r = rb * 16 + np.arange(16)
        codes = n3[rb].T.astype(np.float32)               # [16 rows, 256 k]
        w[r] = codes * d[kb[None, :] * 32 + r[:, None]] + m[kb[None, :] * 32 + r[:, None]]
    return w


def dequant_pool(pool: np.ndarray, n: int, k: int, rs: int) -> np.ndarray:
    """Pool-order chunk bytes -> f32[n, k]."""
    nch, _, rows0, cols0 = chunk_geometry(n, k, rs)
    w = np.empty((n, k), np.float32)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        w[r0:r0 + 32, c0:c0 + 256] = dequant_chunk(pool[c * CH:(c + 1) * CH])
    return w


def pool_reference(pool: np.ndarray, x: np.ndarray, n: int, k: int, rs: int) -> np.ndarray:
    """y = W @ x in fp64 from the SAME pool bytes the kernel streams (f32[n])."""
    nch, _, rows0, cols0 = chunk_geometry(n, k, rs)
    xf = x.astype(np.float64)
    y = np.zeros(n, np.float64)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        y[r0:r0 + 32] += dequant_chunk(pool[c * CH:(c + 1) * CH]).astype(np.float64) @ xf[c0:c0 + 256]
    return y.astype(np.float32)


def _selftest() -> int:
    rng = np.random.default_rng(0)
    ok = True
    for n, k, rs in [(512, 2048, 2), (2048, 512, 2), (512, 2048, 4)]:
        blocks = random_q4_1_blocks(n, k, rng)
        pool = pack_q4_1_pool(blocks, rs)
        want = dequant_q4_1(blocks, bfloat16)
        got = dequant_pool(pool, n, k, rs)
        exact = np.array_equal(got, want)
        # and the fp16 -> bf16 narrowing is the ONLY loss vs the GGUF bytes
        rel = np.abs(want - dequant_q4_1(blocks)).max() / np.abs(want).max()
        print(f"{'PASS' if exact else 'FAIL'} n={n} k={k} rs={rs} pool={len(pool)} B "
              f"pool==gguf(bf16 scales): {exact}  bf16-narrowing maxrel={rel:.2e}")
        ok &= exact
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
