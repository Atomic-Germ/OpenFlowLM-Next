r"""GGUF blocks -> pool-order chunks with F32 scales (the direct-GGUF layout).

Companion to ``q4_1_pack.py`` (which documents the bf16-scale chunk layout the
shipping q4nx containers use). The direct-GGUF path is lossless in values:
the pack is a byte permutation of the codes plus an EXACT fp16 -> f32 widening
of the scales (every fp16 value is representable in f32: mantissa << 13,
exponent + 112 -- no rounding). Q4_0's signed nibble law ``d * (q - 8)`` is
represented losslessly as Q4_1-style ``d * q + m`` with ``m = -8*d``.

Why f32 scales in the pool: XDNA2 has no fp16 vector type, so the kernel could
only read GGUF's fp16 scales through ~200 extra integer vector ops per chunk
(per-bit decode of f32) or a 3-mantissa-bit lossy pack-side narrowing (what
q4nx-build does today). f32 is the exact middle: the pack widens losslessly,
and the kernel converts f32 -> bf16 with the native one-op RNE convert the
epilogue already uses for its accumulators -- so a GGUF model computes with
bit-identical numerics to its q4nx-converted twin (both are RNE-bf16 scales
inside the MAC), while the file keeps the GGUF values exactly.

Chunk layouts (same tile geometry as the q4nx container: 32 rows x 256 K):

  q4 chunk, 6144 B:  256 f32 scales ``d`` at [0:1024] and 256 f32 mins ``m``
    at [1024:2048], index j = kb*32 + r -- the GGUF per-block (row, kblock)
    scale pair, permuted into the kernel's kb-major plane order.
  q8 chunk, 9216 B (lm_head): 256 f32 scales, then 8192 int8 codes.

  codes at [2048:6144] / [1024:9216], 16-lane interleaved:
    byte p = (r/16)*4096 + k*16 + (r%16) holds element (row r, col k),
    even p = low nibble (q4). GGUF packs two K of ONE row per byte; this
    packs two ROWS at one K (the IMU B operand is [8 k][8 row pairs]) --
    that transpose is the only non-trivial part, and it is lossless.

Pool (band) order: identical to ``q4_1_pack.chunk_geometry`` -- the band law
is a whole-chunk permutation and does not care about the chunk's internal
layout.

``python gguf_pool.py`` self-tests: random GGUF blocks (Q4_1 / Q4_0 / Q8_0)
-> pack -> dequant of the pool bytes must equal dequant of the GGUF blocks
bit-exactly (fp32 arithmetic, same expression, same operands).
"""

from __future__ import annotations

import struct
import sys

import numpy as np

CH_Q4 = 6144           # pool chunk bytes, 32 rows x 256 k (q4, f32 scales)
CH_Q8 = 9216           # pool chunk bytes, 32 rows x 256 k (q8 lm_head, f32 scales)
SCALE_BYTES = 1024     # 256 f32 scales per plane
BLK = 32               # values per GGUF block
Q4_1_BYTES = 20        # fp16 d, fp16 m, 16 nibble bytes
Q4_0_BYTES = 18        # fp16 d, 16 nibble bytes (no min)
Q8_0_BYTES = 34        # fp16 d, 32 int8 values


def random_q4_blocks(n: int, k: int, rng: np.random.Generator, scale: float = 0.02,
                     gguf_type: str = "Q4_1") -> np.ndarray:
    """GGUF-style raw block bytes for an [n, k] matrix: uint8[n, k//32, blk_bytes]."""
    assert k % BLK == 0
    nb = k // BLK
    d = (rng.random((n, nb), np.float32) * scale + 1e-3).astype(np.float16)
    q = rng.integers(0, 16, (n, nb, BLK), dtype=np.uint8)
    nbytes = Q4_1_BYTES if gguf_type == "Q4_1" else Q4_0_BYTES
    blocks = np.zeros((n, nb, nbytes), np.uint8)
    blocks[..., 0:2] = d.view(np.uint8).reshape(n, nb, 2)
    if gguf_type == "Q4_1":
        m = ((rng.random((n, nb), np.float32) - 0.5) * scale * 15).astype(np.float16)
        blocks[..., 2:4] = m.view(np.uint8).reshape(n, nb, 2)
        blocks[..., 4:20] = q[..., :16] | (q[..., 16:] << 4)
    else:
        blocks[..., 2:18] = q[..., :16] | (q[..., 16:] << 4)
    return blocks


def random_q8_0_blocks(n: int, k: int, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    """GGUF Q8_0 raw blocks uint8[n, k//32, 34] (fp16 d + 32 int8 values)."""
    assert k % BLK == 0
    nb = k // BLK
    d = (rng.random((n, nb), np.float32) * scale + 1e-3).astype(np.float16)
    v = rng.integers(-128, 128, (n, nb, BLK), dtype=np.int8)
    blocks = np.empty((n, nb, Q8_0_BYTES), np.uint8)
    blocks[..., 0:2] = d.view(np.uint8).reshape(n, nb, 2)
    blocks[..., 2:] = v.view(np.uint8)
    return blocks


def random_iq4_xs_blocks(n: int, k: int, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    """GGUF IQ4_XS raw blocks uint8[n, k//256, 136] (fp16 d, 2 B of 2-bit scale highs,
    4 B of 4-bit scale lows, 128 B of 4-bit codes)."""
    assert k % 256 == 0
    nb = k // 256
    d = (rng.random((n, nb), np.float32) * scale + 1e-3).astype(np.float16)
    # the six-bit scales are stored as the value + 32 (the reader's "(u) - 32")
    sc = rng.integers(0, 64, (n, nb, 8), dtype=np.int16)
    q = rng.integers(0, 16, (n, nb, 8, 32), dtype=np.uint8)
    blocks = np.zeros((n, nb, IQ4_XS_BYTES), np.uint8)
    blocks[..., 0:2] = d.view(np.uint8).reshape(n, nb, 2)
    lo = (sc & 0xF).astype(np.uint8)
    hi = ((sc >> 4) & 0x3).astype(np.uint8)
    for g in range(8):
        if g & 1:
            blocks[..., 4 + g // 2] |= lo[:, :, g] << 4
        else:
            blocks[..., 4 + g // 2] |= lo[:, :, g]
        blocks[..., 2] |= hi[:, :, g] << (2 * g)
    blocks[..., 8:136] = (q[..., :16] | (q[..., 16:] << 4)).reshape(n, nb, 128)
    return blocks


def random_q5_k_blocks(n: int, k: int, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    """GGUF Q5_K raw blocks uint8[n, k//256, 176] (fp16 d, fp16 dmin, 12 B of 6-bit
    scales, 32 B of the fifth bit, 128 B of 4-bit codes)."""
    assert k % 256 == 0
    nb = k // 256
    d = (rng.random((n, nb), np.float32) * scale + 1e-3).astype(np.float16)
    dm = (rng.random((n, nb), np.float32) * scale * 0.1).astype(np.float16)
    sc = rng.integers(0, 64, (n, nb, 8), dtype=np.int16)
    mn = rng.integers(0, 64, (n, nb, 8), dtype=np.int16)
    q = rng.integers(0, 16, (n, nb, 4, 64), dtype=np.uint8)
    fifth = rng.integers(0, 2, (n, nb, 256), dtype=np.uint8)
    blocks = np.zeros((n, nb, 176), np.uint8)
    blocks[..., 0:2] = d.view(np.uint8).reshape(n, nb, 2)
    blocks[..., 2:4] = dm.view(np.uint8).reshape(n, nb, 2)
    # the scales array (bytes 4..15): the first four scale/min pairs in the low six
    # bits of bytes 0..3 and 4..7, the last four split across the shared bytes 8..11
    s = np.zeros((n, nb, 12), np.uint8)
    s[..., 0:4] = (sc[..., 0:4] & 0x3F).astype(np.uint8)
    s[..., 4:8] = (mn[..., 0:4] & 0x3F).astype(np.uint8)
    s[..., 8:12] |= (sc[..., 4:8] & 0x0F).astype(np.uint8)
    s[..., 0:4] |= ((sc[..., 4:8] >> 4) & 0x03).astype(np.uint8) << 6
    s[..., 4:8] |= ((mn[..., 4:8] >> 4) & 0x03).astype(np.uint8) << 6
    s[..., 8:12] |= ((mn[..., 4:8] & 0x0F) << 4).astype(np.uint8)
    blocks[..., 4:16] = s
    # the fourth-bit codes: per 64-value half, the low nibbles of 32 bytes then the high
    for h in range(4):
        blocks[..., 48 + h * 32:48 + h * 32 + 32] = (q[..., h, 0:32] | (q[..., h, 32:64] << 4))
    # the fifth bit: byte (v % 32) holds the bits of the values v, v+32, v+64, ...
    for v in range(256):
        blocks[..., 16 + v % 32] |= (fifth[..., v] << (v // 32)).astype(np.uint8)
    return blocks


def unpack_q4(raw: np.ndarray, gguf_type: str = "Q4_1"):
    """Raw block bytes uint8[n, nb, B] -> (d f32[n, nb], m f32[n, nb], q uint8[n, nb*32])."""
    n, nb, b = raw.shape
    d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(n, nb).astype(np.float32)
    m = (np.ascontiguousarray(raw[..., 2:4]).view(np.float16).reshape(n, nb).astype(np.float32)
         if b == Q4_1_BYTES else -8.0 * d)
    qs = raw[..., 4:20] if b == Q4_1_BYTES else raw[..., 2:18]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(n, nb * BLK)
    return d, m, q


def unpack_q8_0(raw: np.ndarray):
    """Raw Q8_0 blocks uint8[n, nb, 34] -> (d f32[n, nb], v int8[n, nb*32])."""
    n, nb, _ = raw.shape
    d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(n, nb).astype(np.float32)
    v = np.ascontiguousarray(raw[..., 2:]).view(np.int8).reshape(n, nb * BLK)
    return d, v


def dequant_q4(raw: np.ndarray, gguf_type: str = "Q4_1") -> np.ndarray:
    """GGUF Q4 blocks -> f32[n, k]."""
    n, nb, _ = raw.shape
    d, m, q = unpack_q4(raw, gguf_type)
    return q.astype(np.float32) * np.repeat(d, BLK, axis=1) + np.repeat(m, BLK, axis=1)


def dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    """GGUF Q8_0 blocks -> f32[n, k]."""
    n, nb, _ = raw.shape
    d, v = unpack_q8_0(raw)
    return v.astype(np.float32) * np.repeat(d, BLK, axis=1)


# llama.cpp's iq4_nl codebook: the sixteen values a 4-bit code indexes. iq4_xs uses
# the same table (a finer-grained scale on top of it).
IQ4_NL_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                           1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)
IQ4_NL_BYTES = 18        # fp16 d, 16 nibble bytes
IQ4_XS_BYTES = 136      # fp16 d, 2 B of 2-bit scale highs, 4 B of 4-bit scale lows, 128 code bytes


def get_scale_min_k4(scales: np.ndarray):
    """The eight 6-bit scale/min pairs of a k-quant super-block: [..., 12] -> ([..., 8], [..., 8]).

    The same law as the reader's get_scale_min_k4: the first four pairs are the
    low six bits of bytes 0..3 (scale) and 4..7 (min); the last four split a
    byte's high two bits (scale) and high two bits (min) with the top bits of
    the earlier bytes.
    """
    q = scales.astype(np.uint16)
    out_d = np.empty(q.shape[:-1] + (8,), np.int32)
    out_m = np.empty_like(out_d)
    for j in range(8):
        if j < 4:
            out_d[..., j] = q[..., j] & 63
            out_m[..., j] = q[..., j + 4] & 63
        else:
            out_d[..., j] = (q[..., j + 4] & 0xF) | ((q[..., j - 4] >> 6) << 4)
            out_m[..., j] = (q[..., j + 4] >> 4) | ((q[..., j] >> 6) << 4)
    return out_d, out_m


def dequant_q5_k(raw: np.ndarray) -> np.ndarray:
    """GGUF Q5_K blocks uint8[n, nb, 176] -> f32[n, nb*256].

    A super-block of 256 values: fp16 d, fp16 dmin, 12 bytes of 6-bit scales
    (eight scale/min pairs over eight groups of 32 values), 32 bytes of the
    FIFTH bit (byte v%32 holds the bits of the values v, v+32, v+64, ...), then
    128 bytes of 4-bit codes. The value is d*scale*(nibble | fifth<<4) - dmin*min.
    """
    n, nb, _ = raw.shape
    d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(n, nb).astype(np.float32)
    dmin = np.ascontiguousarray(raw[..., 2:4]).view(np.float16).reshape(n, nb).astype(np.float32)
    sc, m = get_scale_min_k4(raw[..., 4:16])
    qh = raw[..., 16:48]
    qs = raw[..., 48:176]

    # the 4-bit codes: per 64-value half, the low nibbles of 32 bytes then the high ones
    ql = np.empty((n, nb, 256), np.uint8)
    for h in range(4):
        ql[..., h * 64:h * 64 + 32] = qs[..., h * 32:h * 32 + 32] & 0xF
        ql[..., h * 64 + 32:h * 64 + 64] = qs[..., h * 32:h * 32 + 32] >> 4
    # the fifth bit: byte (v % 32) holds the bits of the values v, v+32, v+64, ...
    fifth = ((qh[..., :, None] >> np.arange(8, dtype=np.uint8)) & 0x01)            # [n, nb, 32, 8]
    fifth = fifth.transpose(0, 1, 3, 2).reshape(n, nb, 256)                      # bit-major -> value order
    q = (ql | (fifth << 4)).astype(np.float32).reshape(n, nb, 8, 32)
    d = (d[..., None] * sc.astype(np.float32))[..., None]    # [n, nb, 8, 1]: a scale per group of 32
    dm = (dmin[..., None] * m.astype(np.float32))[..., None]
    return (d * q - dm).reshape(n, nb * 256)


def dequant_iq4_nl(raw: np.ndarray) -> np.ndarray:
    """GGUF IQ4_NL blocks uint8[n, nb, 18] -> f32[n, nb*32].

    Each 4-bit code indexes a sixteen-entry codebook (llama.cpp's iq4_nl), scaled by
    the block's fp16 d.
    """
    n, nb, _ = raw.shape
    d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(n, nb).astype(np.float32)
    qs = raw[..., 2:18]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(n, nb * 32)
    return d[:, :, None] * IQ4_NL_VALUES[q].astype(np.float32)


def dequant_iq4_xs(raw: np.ndarray) -> np.ndarray:
    """GGUF IQ4_XS blocks uint8[n, nb, 136] -> f32[n, nb*256].

    A super-block of 256 values: fp16 d, two bytes of 2-bit scale highs, four bytes
    of 4-bit scale lows, then 128 bytes of 4-bit codes. Eight groups of 32 values,
    each with a signed 6-bit scale (the low nibble | the high two bits << 4, minus
    32) times d; the codes index the iq4_nl codebook.
    """
    n, nb, _ = raw.shape
    d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(n, nb).astype(np.float32)
    sh = np.ascontiguousarray(raw[..., 2:4]).view(np.uint16).reshape(n, nb)
    sl = raw[..., 4:8]
    qs = raw[..., 8:136]

    # the eight group scales: the low nibble of scales_l (two per byte) | the high two
    # bits of scales_h << 4, as a signed 6-bit value
    # the low nibbles and the high ones INTERLEAVE (byte 0 low, byte 0 high, byte 1 low, ...)
    lo = np.stack([sl & 0xF, sl >> 4], axis=-1).reshape(n, nb, 8).astype(np.int16)
    hi = ((sh[:, :, None] >> np.arange(8, dtype=np.uint16) * 2) & 0x3).astype(np.int16)
    scales = (lo | (hi << 4)) - 32
    dl = (d[:, :, None] * scales.astype(np.float32))[:, :, :, None]     # [n, nb, 8, 1]

    # the codes: 16 bytes per group of 32 values -- the sixteen low nibbles first,
    # then the sixteen high ones
    q = qs.reshape(n, nb, 8, 16)
    q = np.concatenate([q & 0xF, q >> 4], axis=-1).reshape(n, nb, 8, 32)
    return (dl * IQ4_NL_VALUES[q].astype(np.float32)).reshape(n, nb * 256)


def chunk_geometry(n: int, k: int, rs: int):
    """(nch, per_band, rows0[nch], cols0[nch]) for the pool order (q4_1_pack)."""
    assert k % 256 == 0 and n % (32 * rs) == 0
    per_band = rs * k // 256
    nch = n * k // (32 * 256)
    c = np.arange(nch)
    band, ci = np.divmod(c, per_band)
    rows0 = 32 * rs * band + 32 * (ci % rs)
    cols0 = 256 * (ci // rs)
    return nch, per_band, rows0, cols0


def _widened_f32(raw, rows0, cols0, sl):
    """GGUF fp16 scale bytes for one chunk, widened EXACTLY to f32 and
    permuted to the kernel's j = kb*32 + r. sl = scale byte slice in a block."""
    nb = raw.shape[1]
    kb0 = cols0 // BLK
    f16 = raw[rows0:rows0 + 32, kb0:kb0 + 8, sl] \
        .reshape(32, 8, 2).view(np.float16)[..., 0]
    f32 = f16.astype(np.float32)                     # exact widening
    return np.ascontiguousarray(f32.T).reshape(-1).view(np.uint8)  # [8 kb, 32 r]


def _chunk_d32(raw, rows0, cols0):
    return _widened_f32(raw, rows0, cols0, slice(0, 2))


def _chunk_m32(raw, rows0, cols0):
    return _widened_f32(raw, rows0, cols0, slice(2, 4))


def _codes_to_nibbles(q: np.ndarray, rows0: int, cols0: int) -> np.ndarray:
    """[32 r, 256 k] codes -> the chunk's 4096 interleaved nibble bytes."""
    qq = q[rows0:rows0 + 32, cols0:cols0 + 256].reshape(2, 16, 256) \
        .transpose(0, 2, 1).reshape(-1)                      # [rb, k, r16], r16 fastest
    return qq[0::2] | (qq[1::2] << 4)


def _codes_to_int8(v: np.ndarray, rows0: int, cols0: int) -> np.ndarray:
    """[32 r, 256 k] int8 codes -> the chunk's 8192 interleaved bytes."""
    return v[rows0:rows0 + 32, cols0:cols0 + 256].reshape(2, 16, 256) \
        .transpose(0, 2, 1).reshape(-1)


def q4_chunk_bytes(raw: np.ndarray, r0: int, c0: int, gguf_type: str = "Q4_1") -> np.ndarray:
    """One 6144 B f32-scale chunk covering rows r0..r0+31, cols c0..c0+255."""
    q = unpack_q4(raw, gguf_type)[2]
    out = np.empty(CH_Q4, np.uint8)
    out[0:1024] = _chunk_d32(raw, r0, c0)
    if gguf_type == "Q4_1":
        out[1024:2048] = _chunk_m32(raw, r0, c0)
    else:
        # Q4_0's min is -8*d, NOT zero: its nibbles are offset binary, so zeroing
        # the min plane puts every weight in the block 8*d too high and a block
        # that should straddle zero lands entirely on one side of it. pack_q4_pool
        # has always written it; this helper had not, and nothing called it.
        out[1024:2048] = np.ascontiguousarray(-8.0 * out[0:1024].view(np.float32)).view(np.uint8)
    out[2048:] = _codes_to_nibbles(q, r0, c0)
    return out


def q8_chunk_bytes(raw: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """One 9216 B f32-scale q8 chunk covering rows r0..r0+31, cols c0..c0+255."""
    v = unpack_q8_0(raw)[1]
    out = np.empty(CH_Q8, np.uint8)
    out[0:1024] = _chunk_d32(raw, r0, c0)
    out[1024:] = _codes_to_int8(v, r0, c0)
    return out


def pack_q4_pool(raw: np.ndarray, rs: int, gguf_type: str = "Q4_1") -> np.ndarray:
    """GGUF Q4 raw blocks uint8[n, nb, B] -> pool chunk bytes uint8[nch*6144].

    Scales widen EXACTLY fp16 -> f32; Q4_0 gets the exact ``-8*d`` min.
    Codes are permuted only. No requantization.
    """
    assert raw.shape[-1] == (Q4_1_BYTES if gguf_type == "Q4_1" else Q4_0_BYTES)
    n, nb, _ = raw.shape
    k = nb * BLK
    _, _, rows0, cols0 = chunk_geometry(n, k, rs)
    nch = len(rows0)
    q = unpack_q4(raw, gguf_type)[2]
    out = np.empty((nch, CH_Q4), np.uint8)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        out[c, 0:1024] = _chunk_d32(raw, r0, c0)
        if gguf_type == "Q4_1":
            out[c, 1024:2048] = _chunk_m32(raw, r0, c0)
        else:
            d = out[c, 0:1024].view(np.float32)
            out[c, 1024:2048] = np.ascontiguousarray(-8.0 * d).view(np.uint8)
        out[c, 2048:] = _codes_to_nibbles(q, r0, c0)
    return out.reshape(-1)


def pack_q8_pool(raw: np.ndarray, rs: int) -> np.ndarray:
    """GGUF Q8_0 raw blocks uint8[n, nb, 34] -> pool chunk bytes uint8[nch*9216]
    in the q4 band law (expert / dense-GEMV shapes use this when a q8 tensor
    rides the standard band order)."""
    n, nb, _ = raw.shape
    k = nb * BLK
    _, _, rows0, cols0 = chunk_geometry(n, k, rs)
    nch = len(rows0)
    out = np.empty((nch, CH_Q8), np.uint8)
    for c in range(nch):
        out[c] = q8_chunk_bytes(raw, int(rows0[c]), int(cols0[c]))
    return out.reshape(-1)


def pack_q8_pool_lmhead(raw: np.ndarray) -> np.ndarray:
    """GGUF Q8_0 raw blocks uint8[n, nb, 34] -> the lm_head pool's supertile
    order (pools.cpp lmhead_q8): pool chunk k <- file chunk
    (4*(k//32) + (k%4))*8 + ((k%32)//4) -- a band is 32 pool chunks = 128 rows
    x 2048 K, quarter = k%4 (rows), k-tile = k//4."""
    n, nb, _ = raw.shape
    k = nb * BLK
    assert k % 256 == 0 and n % 128 == 0
    ncol = k // 256
    nch = n * k // (32 * 256)
    file_chunks = np.empty((nch, CH_Q8), np.uint8)
    for f in range(nch):
        file_chunks[f] = q8_chunk_bytes(raw, 32 * (f // ncol), 256 * (f % ncol))
    out = np.empty((nch, CH_Q8), np.uint8)
    for p in range(nch):
        s, r = divmod(p, 32)
        fch = (4 * s + r % 4) * 8 + r // 4
        out[p] = file_chunks[fch]
    return out.reshape(-1)


def dequant_pool_lmhead(pool: np.ndarray, n: int, k: int, scale_dtype=None) -> np.ndarray:
    """Supertile-order 9216 B chunks -> f32[n, k]. scale_dtype narrows the f32
    scales (bfloat16 = what the kernel's RNE convert sees)."""
    nch = len(pool) // CH_Q8
    chunks = np.frombuffer(pool, np.uint8).reshape(nch, CH_Q8)
    ncol = k // 256
    w = np.empty((n, k), np.float32)
    for p in range(nch):
        s, r = divmod(p, 32)
        fch = (4 * s + r % 4) * 8 + r // 4
        b = chunks[p].copy()          # pool chunk p holds file chunk fch
        if scale_dtype is not None:
            narrow = b[0:1024].view(np.float32).astype(scale_dtype).astype(np.float32)
            b[0:1024] = narrow.view(np.uint8).reshape(1024)
        w[32 * (fch // ncol):32 * (fch // ncol) + 32, 256 * (fch % ncol):256 * (fch % ncol) + 256] = dequant_q8_chunk(b)
    return w


def dequant_q4_chunk(b: np.ndarray) -> np.ndarray:
    """One 6144 B f32-scale pool chunk -> f32[32 rows, 256 k]."""
    d = b[0:1024].view(np.float32)                        # index kb*32 + r
    m = b[1024:2048].view(np.float32)
    qb = b[2048:]
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


def dequant_q8_chunk(b: np.ndarray) -> np.ndarray:
    """One 9216 B f32-scale pool chunk -> f32[32 rows, 256 k]."""
    d = b[0:1024].view(np.float32)
    c = b[1024:].view(np.int8)
    c3 = c.reshape(2, 256, 16)
    w = np.empty((32, 256), np.float32)
    kb = np.arange(256) // BLK
    for rb in range(2):
        r = rb * 16 + np.arange(16)
        codes = c3[rb].T.astype(np.float32)
        w[r] = codes * d[kb[None, :] * 32 + r[:, None]]
    return w


def dequant_pool(raw: np.ndarray, n: int, k: int, rs: int, gguf_type: str) -> np.ndarray:
    """Pool chunk bytes -> f32[n, k]."""
    ch = CH_Q4 if gguf_type.startswith("Q4") else CH_Q8
    dequant = dequant_q4_chunk if gguf_type.startswith("Q4") else dequant_q8_chunk
    nch, _, rows0, cols0 = chunk_geometry(n, k, rs)
    pool = np.frombuffer(raw, np.uint8).reshape(nch, ch)
    w = np.empty((n, k), np.float32)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        w[r0:r0 + 32, c0:c0 + 256] = dequant(pool[c])
    return w


# ---------------------------------------------------------------------------
# A minimal GGUF file reader (header + KV metadata + tensor index). Enough to
# enumerate tensors and find a quantized weight file's layout; not a loader.
# ---------------------------------------------------------------------------

_GGUF_MAGIC = 0x46554747  # "GGUF" little-endian


def read_gguf(path: str):
    """-> (metadata dict, {name: {type, dims, offset, nbytes}})."""
    GGUF_T = {0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2),
              4: ("u32", 4), 5: ("i32", 4), 6: ("f32", 4), 7: ("bool", 1),
              8: ("str", None), 9: ("arr", None), 10: ("u64", 8), 11: ("i64", 8),
              12: ("f64", 8)}
    # The ggml type numbers (gguf/constants.py, the same table
    # src/open_qwen36/gguf_file.hpp carries): type -> (values per block, bytes per block).
    # The k- and i-quants' BYTES are what the reader needs to walk a tensor; the older
    # table here had the "_S/_M" names of a since-merged ggml and no byte sizes for
    # the superblock types at all.
    GGUF_W = {0: 4, 1: 2, 2: 18, 3: 20, 6: 22, 7: 24, 8: 34, 9: 40,
              10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292,
              16: 66, 17: 74, 18: 98, 19: 50, 20: 18, 21: 110, 22: 82, 23: 136,
              24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 29: 56, 30: 2}
    GGUF_BLK = {2: 32, 3: 32, 6: 32, 7: 32, 8: 32, 9: 32, 10: 256, 11: 256, 12: 256,
                13: 256, 14: 256, 15: 256, 16: 256, 17: 256, 18: 256, 19: 256,
                20: 32, 21: 256, 22: 256, 23: 256, 29: 256}
    QNAME = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
             8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
             14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
             19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
             24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
             30: "BF16"}

    f = open(path, "rb")
    magic, ver, ntensors, nkv = struct.unpack("<IIQQ", f.read(24))
    if magic != _GGUF_MAGIC:
        raise ValueError(f"{path}: not a GGUF file")
    if ver < 2:
        raise ValueError(f"{path}: GGUF v{ver} unsupported (need v2+)")

    def rd_str():
        (ln,) = struct.unpack("<Q", f.read(8))
        return f.read(ln).decode("utf-8")

    def rd_val(t):
        name, size = GGUF_T[t]
        if name == "str":
            return rd_str()
        if name == "arr":
            et, ln = struct.unpack("<IQ", f.read(12))
            return [rd_val(et) for _ in range(ln)]
        if name == "bool":
            return bool(f.read(1)[0])
        fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I",
               "i32": "<i", "f32": "<f", "u64": "<Q", "i64": "<q", "f64": "<d"}
        return struct.unpack(fmt[name], f.read(size))[0]

    meta = {}
    for _ in range(nkv):
        key = rd_str()
        (t,) = struct.unpack("<I", f.read(4))
        meta[key] = rd_val(t)

    tensors = {}
    for _ in range(ntensors):
        name = rd_str()
        (ndims,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{ndims}Q", f.read(8 * ndims))
        (t,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        tensors[name] = {"type": QNAME.get(t, f"type{t}"), "gguf_type": t,
                         "dims": list(dims), "offset": off,
                         "nbytes": (int(np.prod(dims, dtype=np.int64)) // GGUF_BLK[t]
                                    * GGUF_W[t]) if t in GGUF_BLK else 0}
    if ver == 2:
        (align,) = struct.unpack("<I", f.read(4))    # v3 removed it; fixed at 32
    else:
        align = 32
    data_base = f.tell()
    data_base += (align - data_base % align) % align
    f.close()
    return meta, tensors, data_base


def _selftest() -> int:
    rng = np.random.default_rng(0)
    ok = True
    for n, k, rs in [(512, 2048, 2), (2048, 512, 2), (512, 2048, 4)]:
        for gtype in ("Q4_1", "Q4_0"):
            raw = random_q4_blocks(n, k, rng, gguf_type=gtype)
            pool = pack_q4_pool(raw, rs, gtype)
            want = dequant_q4(raw, gtype)
            got = dequant_pool(pool, n, k, rs, gtype)
            exact = np.array_equal(got, want)
            print(f"{'PASS' if exact else 'FAIL'} {gtype} n={n} k={k} rs={rs} "
                  f"pool={len(pool)} B  pool==gguf(bit-exact): {exact}")
            ok &= exact
        if k % 256 == 0:
            raw = random_q5_k_blocks(n, k, rng)
            want = dequant_q5_k(raw).reshape(n, -1)
            # the reference: the block law, spelled out for the first row
            blk = raw[0, 0]
            d = np.float32(np.frombuffer(blk[0:2].tobytes(), np.float16)[0])
            dmin = np.float32(np.frombuffer(blk[2:4].tobytes(), np.float16)[0])
            s = blk[4:16]
            sc = np.empty(8, np.int32)
            mn = np.empty(8, np.int32)
            for j in range(8):
                if j < 4:
                    sc[j] = s[j] & 63
                    mn[j] = s[4 + j] & 63
                else:
                    j4 = j - 4
                    sc[j] = (s[8 + j4] & 0x0F) | ((s[j4] >> 6) << 4)
                    mn[j] = ((s[8 + j4] >> 4) & 0x0F) | ((s[4 + j4] >> 6) << 4)
            got = np.empty(256, np.float32)
            for v in range(256):
                h = (v // 64) * 64
                byte = blk[48 + h // 2 + (v % 32)]
                nib = (byte & 0xF) if (v % 64) < 32 else (byte >> 4)
                bit = (blk[16 + v % 32] >> (v // 32)) & 1
                got[v] = d * sc[v // 32] * (nib | (bit << 4)) - dmin * mn[v // 32]
            exact = np.array_equal(got, want[0, :256])
            print(f"{'PASS' if exact else 'FAIL'} Q5_K n={n} k={k}  dequant==the block law: {exact}")
            ok &= exact
            raw = random_iq4_xs_blocks(n, k, rng)
            want = dequant_iq4_xs(raw).reshape(n, -1)
            # the reference: the same law, spelled out value by value
            nb = k // 256
            got = np.empty_like(want)
            for b in range(nb):
                blk = raw[0, b]
                d = np.float32(np.frombuffer(blk[0:2].tobytes(), np.float16)[0])
                for g in range(8):
                    lo = (blk[4 + g // 2] >> 4) if (g & 1) else (blk[4 + g // 2] & 0xF)
                    hi = (blk[2] >> (2 * g)) & 0x3
                    dl = d * (int(lo | (hi << 4)) - 32)
                    for j in range(32):
                        byte = blk[8 + g * 16 + (j % 16)]
                        code = (byte & 0xF) if j < 16 else (byte >> 4)
                        got[0, b * 256 + g * 32 + j] = dl * float(IQ4_NL_VALUES[code])
            exact = np.array_equal(got[0], want[0])       # the reference spells out row 0
            print(f"{'PASS' if exact else 'FAIL'} IQ4_XS n={n} k={k}  dequant==the block law: {exact}")
            ok &= exact
        raw = random_q8_0_blocks(n, k, rng)
        pool = pack_q8_pool(raw, rs)
        want = dequant_q8_0(raw)
        got = dequant_pool(pool, n, k, rs, "Q8_0")
        exact = np.array_equal(got, want)
        print(f"{'PASS' if exact else 'FAIL'} Q8_0 n={n} k={k} rs={rs} "
              f"pool={len(pool)} B  pool==gguf(bit-exact): {exact}")
        ok &= exact
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
