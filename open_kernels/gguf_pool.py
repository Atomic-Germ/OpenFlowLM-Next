r"""GGUF blocks -> pool-order chunks with F32 scales (the direct-GGUF layout).

Companion to ``q4_1_pack.py`` (which documents the bf16-scale chunk layout the
shipping q4nx containers use). The direct-GGUF path is lossless in values:
the pack is a byte permutation of the codes plus an EXACT fp16 -> f32 widening
of the scales (every fp16 value is representable in f32: mantissa << 13,
exponent + 112 -- no rounding), and Q4_0 sources get m = 0 (GGUF Q4_0 has no
min). Nothing is requantized or narrowed.

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


def unpack_q4(raw: np.ndarray, gguf_type: str = "Q4_1"):
    """Raw block bytes uint8[n, nb, B] -> (d f32[n, nb], m f32[n, nb], q uint8[n, nb*32])."""
    n, nb, b = raw.shape
    d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(n, nb).astype(np.float32)
    m = (np.ascontiguousarray(raw[..., 2:4]).view(np.float16).reshape(n, nb).astype(np.float32)
         if b == Q4_1_BYTES else np.zeros((n, nb), np.float32))
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
    out[1024:2048] = _chunk_m32(raw, r0, c0) if gguf_type == "Q4_1" else np.zeros(1024, np.uint8)
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

    Scales widened EXACTLY fp16 -> f32 (no rounding); Q4_0 gets zero mins.
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
        out[c, 1024:2048] = (_chunk_m32(raw, r0, c0) if gguf_type == "Q4_1"
                             else np.zeros(1024, np.uint8))
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
    GGUF_W = {0: 32, 1: 32, 2: 16, 3: 32, 4: 32, 5: 32, 6: 16, 7: 8, 8: 16,
              9: 32, 10: 64, 12: 32, 13: 8, 14: 4, 15: 8, 16: 20, 17: 18,
              18: 34, 19: 34, 20: 66, 21: 210, 22: 164, 23: 144, 30: 34,
              31: 36, 32: 84}  # superblock types (Q2/Q3/Q4/Q5/Q6_K, IQ) get 0
    QNAME = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
             8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
             13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
             17: "Q5_K_M", 18: "Q6_K", 24: "Q8_K", 28: "IQ4_NL", 30: "BF16"}

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
                         "nbytes": int(np.prod(dims, dtype=np.int64)) * GGUF_W.get(t, 0)
                         if t in GGUF_W else 0}
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
