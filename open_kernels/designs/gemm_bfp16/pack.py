r"""Host packing for gemm_bfp16: bfp16ebs8 operands in the order the cores stream them, and C untiled.

Derived from ggml-xdna's src/npu/bfp16_pack.{h,cpp} (Cyrus Attoun, MIT).

bfp16ebs8: eight values along K share one exponent byte (the largest of the eight float exponents) and
each is a signed 8-bit mantissa, q = round-half-even(x * 2^(133 - e)) saturated to [-128, 127]; nine bytes
a block. Decoded value q * 2^(e - 133).

A tile is `rows` x k blocks: the 8x8 sub-tiles go row pair (16 rows) by row pair, block column by block
column, the pair's first eight rows then its second. A: for each block of m rows, for each chunk of k: one
tile. B: for each array column c, for each of its n-row groups (rows (c + cols * t) * n onward), for each
chunk of k: one tile.
"""

from __future__ import annotations

import numpy as np

BLOCK_BYTES = 9
COLS, ROWS = 8, 4
TILE_M, TILE_K, TILE_N = 64, 64, 128    # what the shipped build compiles (gemm_bfp16.py defaults)


def encode_blocks(x: np.ndarray) -> np.ndarray:
    """float [..., 8] -> uint8 [..., 9]."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert x.shape[-1] == 8
    e = ((x.view(np.uint32) >> 23) & 0xFF).astype(np.int64)
    mx = e.max(axis=-1)
    q = np.rint(np.ldexp(x.astype(np.float64), (133 - mx)[..., None]))
    q = np.clip(q, -128, 127).astype(np.int8)
    out = np.empty(x.shape[:-1] + (BLOCK_BYTES,), dtype=np.uint8)
    out[..., 0] = mx.astype(np.uint8)
    out[..., 1:] = q.view(np.uint8)
    return out


def decode_blocks(b: np.ndarray) -> np.ndarray:
    """uint8 [..., 9] -> float32 [..., 8]."""
    e = b[..., 0].astype(np.int64)
    q = b[..., 1:].view(np.int8).astype(np.float64)
    return np.ldexp(q, (e - 133)[..., None]).astype(np.float32)


def pack_a(x: np.ndarray, m: int = 64, k: int = 64) -> np.ndarray:
    """[M, K] float -> bytes [M * K / 8 * 9]."""
    M, K = x.shape
    assert M % m == 0 and K % k == 0 and m % 16 == 0
    e = encode_blocks(x.reshape(M, K // 8, 8))                       # [M, K/8, 9]
    e = e.reshape(M // m, m // 16, 2, 8, K // k, k // 8, BLOCK_BYTES)
    return np.ascontiguousarray(e.transpose(0, 4, 1, 5, 2, 3, 6)).reshape(-1)


def pack_b(w: np.ndarray, n: int = 128, k: int = 64, cols: int = COLS) -> np.ndarray:
    """[N, K] float -> bytes [N * K / 8 * 9]."""
    N, K = w.shape
    assert N % (n * cols) == 0 and K % k == 0 and n % 16 == 0
    e = encode_blocks(w.reshape(N, K // 8, 8))
    e = e.reshape(N // (n * cols), cols, n // 16, 2, 8, K // k, k // 8, BLOCK_BYTES)    # [t, c, ...]
    return np.ascontiguousarray(e.transpose(1, 0, 5, 2, 6, 3, 4, 7)).reshape(-1)


def pack_b_group(w: np.ndarray, k: int = 64) -> np.ndarray:
    """One n-row group [n, K] float -> its K/k tiles, in stream order, as bytes."""
    n, K = w.shape
    e = encode_blocks(w.reshape(n, K // 8, 8)).reshape(n // 16, 2, 8, K // k, k // 8, BLOCK_BYTES)   # sp half i kk sx
    return np.ascontiguousarray(e.transpose(3, 0, 4, 1, 2, 5)).reshape(-1)


def unpack_a(buf: np.ndarray, M: int, K: int, m: int = 64, k: int = 64) -> np.ndarray:
    e = buf.reshape(M // m, K // k, m // 16, k // 8, 2, 8, BLOCK_BYTES).transpose(0, 2, 4, 5, 1, 3, 6)
    return decode_blocks(e.reshape(M, K // 8, BLOCK_BYTES)).reshape(M, K)


def unpack_b(buf: np.ndarray, N: int, K: int, n: int = 128, k: int = 64, cols: int = COLS) -> np.ndarray:
    e = buf.reshape(cols, N // (n * cols), K // k, n // 16, k // 8, 2, 8, BLOCK_BYTES).transpose(1, 0, 3, 5, 6, 2, 4, 7)
    return decode_blocks(e.reshape(N, K // 8, BLOCK_BYTES)).reshape(N, K)


def untile_c(c: np.ndarray, M: int, N: int, m: int = 64, n: int = 128, cols: int = COLS, rows: int = ROWS) -> np.ndarray:
    """The c-tiled output (fp32, flat) -> [M, N]."""
    rb = m * rows
    j = N // (n * cols)
    t = c.reshape(cols, M // rb, j, rb, n).transpose(1, 3, 2, 0, 4)     # [rbi, rr, t, c, j]
    return np.ascontiguousarray(t).reshape(M, N)
