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
H_SLOT_WORDS = 8199                     # gemm_bfp16.py's: a core tile's 8192 output words padded to a multiple of 9


def glu_row(f: np.ndarray | int, ff: int, n: int = TILE_N, cols: int = COLS):
    """B row of gate row f (up is 8 further); the n-tile order puts each array column's ff chunks on one run of H slots."""
    per_col = 2 * ff // n // cols
    kc = f // (n // 2)
    ct = (kc % per_col) * cols + kc // per_col
    return ct * n + 16 * ((f % (n // 2)) // 8) + f % 8


def pack_glu_b(gate: np.ndarray, up: np.ndarray, n: int = TILE_N, k: int = TILE_K) -> np.ndarray:
    """gate, up [ff, K] float -> the glu GEMM's B operand, bytes [2 ff * K / 8 * 9]."""
    ff, K = gate.shape
    w = np.empty((2 * ff, K), np.float32)
    rows = glu_row(np.arange(ff), ff, n)
    w[rows], w[rows + 8] = gate, up
    return pack_b(w, n, k)


def h_tile(h_words: np.ndarray, kc: int, row: int) -> np.ndarray:
    """The A tile (bytes, 4608) an H output holds for ff chunk kc and the row block `row` (64 tokens)."""
    off = (kc * ROWS + row) * H_SLOT_WORDS
    return h_words[off:off + TILE_M * TILE_K // 8 * BLOCK_BYTES // 4].view(np.uint8)


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


MX4_TILE = 4352
# the mx4 nibble's code (twice the e2m1 level), indexed by the nibble: gemv_q4/mx4_codes.h
MX4_CODE = np.array([0, 1, 2, 3, 4, 6, 8, 12, -12, 0, -8, -6, -4, -3, -2, -1], np.int8)


def pack_b_mx4(v: np.ndarray, e: np.ndarray, n: int = 128, k: int = 64, cols: int = COLS) -> np.ndarray:
    """MXFP4 [N, K] as mx4 nibble values v (int8, mx4_codes.h) and e8m0 block exponents e (uint8 [N, K/32]) ->
    GB_B=mx4's B, bytes [N/n * K/k * 4352], the tiles in pack_b's order. A tile (mx4_b.cc): x = e + 5 per
    (32-K block, row) as [kb32][sp][half][r], then per (sp, kb, half) 32 B of nibbles, value j = row 8 half + j/8 of
    the pair sp, k = 8 kb + j % 8 -- the bfp16 sub-tile mm_paired pops there, exactly (q = code, x = e + 5)."""
    N, K = v.shape
    assert n == 128 and k == 64 and N % (n * cols) == 0 and K % k == 0 and e.shape == (N, K // 32)
    nk, ng = K // k, N // n
    x = (e.astype(np.int64) + 5).astype(np.uint8).reshape(ng, 8, 2, 8, nk, 2)              # g sp half r kc kb32
    x = x.transpose(0, 4, 5, 1, 2, 3).reshape(ng, nk, 256)                                  # g kc | kb32 sp half r
    nb = (np.asarray(v, np.int8).astype(np.uint8) & 15).reshape(ng, 8, 2, 8, nk, 8, 8)     # g sp half r kc kb kk
    nb = nb.transpose(0, 4, 1, 5, 2, 3, 6).reshape(ng, nk, 4096, 2)                        # g kc | sp kb half r kk
    tiles = np.concatenate([x, nb[..., 0] | (nb[..., 1] << 4)], axis=2)                    # [g, kc, 4352]
    g = np.arange(ng)
    order = np.argsort((g % cols) * (ng // cols) + g // cols, kind="stable")              # column c, then its groups
    return np.ascontiguousarray(tiles[order]).reshape(-1)


def unpack_b_mx4(buf: np.ndarray, N: int, K: int, n: int = 128, k: int = 64, cols: int = COLS) -> np.ndarray:
    """GB_B=mx4's B bytes -> the float [N, K] the kernel multiplies (each value code * 2^(x - 133))."""
    nk, ng = K // k, N // n
    t = np.asarray(buf, np.uint8).reshape(cols, ng // cols, nk, MX4_TILE)
    t = t.transpose(1, 0, 2, 3).reshape(ng, nk, MX4_TILE)                                  # back to group order
    x = t[..., :256].reshape(ng, nk, 2, 8, 2, 8).astype(np.int64)                          # g kc kb32 sp half r
    b = t[..., 256:].reshape(ng, nk, 8, 8, 2, 8, 4)                                        # g kc sp kb half r byte
    q = MX4_CODE[np.stack([b & 15, b >> 4], axis=-1).reshape(ng, nk, 8, 8, 2, 8, 8)]       # .. kk
    sc = np.ldexp(1.0, x - 133).transpose(0, 1, 3, 4, 5, 2)                                # g kc sp half r kb32
    w = q.astype(np.float64) * np.repeat(sc, 4, axis=-1).transpose(0, 1, 2, 5, 3, 4)[..., None]   # g kc sp kb half r kk
    return w.transpose(0, 2, 4, 5, 1, 3, 6).reshape(N, K).astype(np.float32)


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
