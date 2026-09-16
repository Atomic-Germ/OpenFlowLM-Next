"""Interpreter of a recipe's packing plan over a weight container (NumPy).

The plan (qwen36moe.pack_plan) says which tensor lands at which byte offset in
which chunk order; the ops here are the chunk-permutation laws phlegm verified
byte-for-byte against pools captured from OFLM's own engine (they were
open_kernels/model/pools.py's build_layer_pool / build_side / build_pack; the
frozen originals live in specs/open-engine/tests/legacy_pools.py and the test
there checks this interpreter reproduces them). src/open_qwen36/pools.cpp is
the same interpreter in C++.

The container object needs one method: `raw(name) -> bytes-like` (the
tensor's bytes as stored, in the file's raster order). It may also offer
`chunk_bytes_of(name) -> int`, the quantized chunk size that tensor is stored in
(5120 = q4_1, 8704 = q8, 4736 = Q4_K); a container that cannot say is read as q4_1.

  std_perm        standard [out, in] matmul tensor -> pool band order (64-row bands, K/128 chunks)
  q8_perm         the same tensor kept at q8: 16-row half-tiles of its chunks, in the q8 band
                  order (64-row bands, K/64 half-tiles) -- twice the bytes, no arithmetic
  expert_stripes  routed experts' up / gate as interleaved [up_k | gate_k] stripes, each transposed
  expert_down     routed experts' down slices, the RS=4 down law
  put             bytes verbatim (small weights), capped
  conv_transpose  conv1d [taps, NCH] bf16 -> [groups][taps][width]
  lmhead_q8       the q8 lm_head's 128-row supertile order
  transpose       a small [rows, cols] tensor -> [cols, rows] (qwen35's alpha / beta)

**q8 projections** (OPEN-QUANT-Q8). Where the recipe's per-role quant map says q8, the
plan carries `q8_perm` instead of `std_perm` and the pool holds the container's q8 values
as 16-row half-tiles the main cores' `gemv_q8_half` consumes. `q8_perm` refuses a source
that is not q8, naming the tensor: that is the check that a container agrees with the
kernel set it is being packed for.

**Source forms.** The three chunk ops above (std_perm, expert_stripes, expert_down)
accept a q8 tensor transparently and re-quantize it to q4_1 chunk by chunk on the
way into the pool (`requant_q4_1`), and a Q4_K tensor -- what OFLM 1.0.3+ writes --
by transcoding it (`q4k_to_q4_1`). All three forms hold the SAME 32-row x
256-column tile, so no chunk index law changes and neither the plan, the manifest
nor the kernels know the difference. That is what lets the Qwen3.6-35B fine-tunes
-- q8 attention, linear-attention and shared experts, q4_1 routed experts -- and
Qwen3.5's q8 `ssm_out_proj` run on a q4_1-only GEMV, and a 1.0.3 container run at
all (OPEN-QUANT-Q4K). Any other chunk size is refused, naming the tensor
(OPEN-PACK-PLAN).
"""
from __future__ import annotations

import numpy as np

from .catalogue import OpRangeError

CH = 5120
CH_HALF = 2560       # GPT-OSS's chunk: the same q4_1 layout over 32 rows x 128 columns
Q8 = 8704            # a q8 chunk: 256 bf16 scales then 8192 int8 codes
Q4K = 4736           # a Q4_K chunk (OFLM 1.0.3+): uint8 scales/mins, nibbles, one bf16 (S, M) per row
BLOCK = 32           # values per quantisation block, along the input dim
NBLOCK = 8           # 32-blocks per chunk (8192 values = 32 rows x 256 K)


def _chunk_index():
    """(code index, meta index) for the 32 rows x 8 blocks x 32 lanes of one chunk.
    Both the q4_1 and the q8 chunk use this raster (q4nx.py documents it)."""
    r = np.arange(BLOCK)[:, None, None]
    bc = np.arange(NBLOCK)[None, :, None]
    i = np.arange(BLOCK)[None, None, :]
    return ((r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)).reshape(-1), (bc * BLOCK + r + 0 * i).reshape(-1)


_CODE_IDX, _META_IDX = _chunk_index()
# per-block meta slot for the 32 rows x 8 blocks of a chunk, in (row, block) order
_BLOCK_META_IDX = (np.arange(NBLOCK)[None, :] * BLOCK + np.arange(BLOCK)[:, None]).reshape(-1)


def _bf16_to_f32(u16) -> np.ndarray:
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _bf16_floor(x) -> np.ndarray:
    """f32 -> bf16 rounded toward -inf (truncate a positive magnitude, grow a negative one)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return np.where(u >> 31, (u + 0xFFFF) >> 16, u >> 16).astype(np.uint16)


def _bf16_ceil(x) -> np.ndarray:
    """f32 -> bf16 rounded toward +inf."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return np.where(u >> 31, u >> 16, (u + 0xFFFF) >> 16).astype(np.uint16)


def _bf16_rne(x) -> np.ndarray:
    """f32 -> bf16, round to nearest even. The directed roundings above exist to make a
    re-quantized block's range cover its source; a Q4_K scale is not a range end, it is a
    value being re-expressed, so it takes the nearest bf16."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def requant_q4_1(chunks) -> np.ndarray:
    """[n, 8704] q8 chunk bytes -> [n, 5120] q4_1 chunk bytes, block for block.

    Per 32-value block: `m` is the block minimum rounded DOWN in bf16, and `d` is
    (max - m) / 15 -- the range measured from the STORED m -- rounded UP. So
    [m, m + 15d] provably covers [min, max] whatever bf16 did to either end, every value
    lands within d/2 of its q4_1 reading (the criterion OPEN-FAMILY-QWEN35 states), and a
    constant block is exact. Plain round-to-nearest on d and m would push an extreme value
    outside the range and past d/2 after clipping. The nibble is chosen against the stored
    d and m, so what this bounds is the reader's error, not an idealised one.

    src/open_qwen36/pools.cpp does the same arithmetic in float and must agree byte for
    byte (specs/open-engine/tests/test_qwen35.py and pools_test.cpp check the same vectors).
    """
    src = _u8(chunks).reshape(-1, Q8)
    n = src.shape[0]
    sc = _bf16_to_f32(np.ascontiguousarray(src[:, :512]).view(np.uint16))            # [n, 256]
    code = np.ascontiguousarray(src[:, 512:]).view(np.int8)                          # [n, 8192]
    v = (code[:, _CODE_IDX].astype(np.float32)
         * sc[:, _META_IDX].astype(np.float32)).reshape(n, BLOCK, NBLOCK, BLOCK)
    mn = v.min(-1)
    mx = v.max(-1)
    m_u = _bf16_floor(mn)
    m = _bf16_to_f32(m_u).astype(np.float32)
    d_u = _bf16_ceil(((mx - m).astype(np.float32) / np.float32(15)).astype(np.float32))
    d = _bf16_to_f32(d_u).astype(np.float32)
    inv = np.where(d > np.float32(0), np.float32(1) / d, np.float32(0)).astype(np.float32)
    t = ((v - m[..., None]).astype(np.float32) * inv[..., None]).astype(np.float32) + np.float32(0.5)
    nib = np.clip(t.astype(np.int32), 0, 15).astype(np.uint8)
    out = np.zeros((n, CH), np.uint8)
    meta_d = np.zeros((n, 256), np.uint16)
    meta_m = np.zeros((n, 256), np.uint16)
    meta_d[:, _BLOCK_META_IDX] = d_u.reshape(n, -1)
    meta_m[:, _BLOCK_META_IDX] = m_u.reshape(n, -1)
    out[:, :512] = meta_d.view(np.uint8).reshape(n, 512)
    out[:, 512:1024] = meta_m.view(np.uint8).reshape(n, 512)
    flat = np.zeros((n, 8192), np.uint8)
    flat[:, _CODE_IDX] = nib.reshape(n, -1)
    out[:, 1024:] = flat[:, 0::2] | (flat[:, 1::2] << 4)
    return out


def q4k_to_q4_1(chunks) -> np.ndarray:
    """[n, 4736] Q4_K chunk bytes -> [n, 5120] q4_1 chunk bytes, in the same chunk order.

    Both formats hold a 32-row x 256-column tile with one (scale, min) pair per (row,
    32-column group), indexed `g*32 + r` in both, so nothing is re-quantized and no index
    law moves. The Q4_K chunk (`q4k_block_t` in the OFLM 1.0.3 decoding kernels) is

        scales[8][32] uint8 @ [0, 256)      mins[8][32] uint8 @ [256, 512)
        qs[256][16]         @ [512, 4608)   byte k*16 + r/2, even row in the low nibble
        S[32] bf16          @ [4608, 4672)  M[32] bf16 @ [4672, 4736), M already negated

    and reads as `S[r] * scales[g][r] * nib + M[r] * mins[g][r]`. That is the pool's
    `nib*d + m` with `d = S*scales` and `m = M*mins`, so the transcode is:

      * the two products, rounded to bf16 -- the one place values move. The exact product
        of a bf16 and a uint8 needs 16 significand bits and the pool holds 8, so each
        group's scale shifts by at most a half-ulp of bf16, 2^-8 relative. Nothing else
        is lost;
      * a byte de-interleave of the nibbles. Q4_K keeps a column's 32 rows in 16
        contiguous bytes; the pool splits rows 0-15 and 16-31 into two 2048-byte planes,
        so q4_1 byte `h*2048 + k*8 + j` is Q4_K byte `k*16 + h*8 + j`. The nibble values
        and their parity are unchanged.

    src/open_qwen36/pools.cpp does the same in float and must agree byte for byte
    (specs/open-engine/tests/test_quant_q4k.py and pools_test.cpp hash the same vectors).
    """
    src = _u8(chunks).reshape(-1, Q4K)
    n = src.shape[0]
    S = _bf16_to_f32(np.ascontiguousarray(src[:, 4608:4672]).view(np.uint16))     # [n, 32] per row
    M = _bf16_to_f32(np.ascontiguousarray(src[:, 4672:4736]).view(np.uint16))     # [n, 32] per row
    row = np.arange(256) % BLOCK                                                  # meta slot g*32 + r
    d = (S[:, row] * src[:, :256].astype(np.float32)).astype(np.float32)
    m = (M[:, row] * src[:, 256:512].astype(np.float32)).astype(np.float32)
    out = np.empty((n, CH), np.uint8)
    out[:, :512] = _bf16_rne(d).view(np.uint8).reshape(n, 512)
    out[:, 512:1024] = _bf16_rne(m).view(np.uint8).reshape(n, 512)
    qs = src[:, 512:4608].reshape(n, 256, 2, 8)                                   # [k, half, j]
    out[:, 1024:] = qs.transpose(0, 2, 1, 3).reshape(n, CH - 1024)
    return out


# ---------------------------------------------------------------- q8 half-tiles
Q8H_SCALES = 256          # 128 bf16 scales, index kb*16 + r
Q8H_CODES = 4096          # 4096 int8 codes, index k*16 + r
Q8H_ROWS = 16


def q8_half_tiles(chunks) -> np.ndarray:
    """[n, 8704] container q8 chunks -> [2n, 5120] pool half-tiles, half `h` of chunk `f`
    at index 2*f + h.

    A container chunk is 32 output rows x 256 K:
        scales[256] bf16 at [0 : 512]    index kb*32 + r     (r = 0..31)
        codes [8192] int8 at [512 : 8704] index (r/16)*4096 + k*16 + (r%16)
    The main cores' w-fifo element is 5120 bytes, so the chunk is split into two 16-row
    half-tiles that fit one:
        scales[128] bf16 at [0 : 256]     index kb*16 + r     (r = 0..15)
        codes [4096] int8 at [256 : 4352] index k*16 + r
        zero pad          at [4352 : 5120]
    The container's row-block stride is exactly 4096 codes, so half h's codes are the
    verbatim slice [h*4096, (h+1)*4096) and only the scales are gathered. A byte
    permutation, no arithmetic -- designs/gemv_q4/gemv_q8.h reads it back unchanged."""
    src = _u8(chunks).reshape(-1, Q8)
    n = src.shape[0]
    out = np.zeros((n, 2, CH), np.uint8)
    sc = src[:, :512].reshape(n, NBLOCK, 2 * Q8H_ROWS, 2)        # [n, kb, row, byte]
    out[:, 0, :Q8H_SCALES] = sc[:, :, :Q8H_ROWS, :].reshape(n, Q8H_SCALES)
    out[:, 1, :Q8H_SCALES] = sc[:, :, Q8H_ROWS:, :].reshape(n, Q8H_SCALES)
    out[:, 0, Q8H_SCALES:Q8H_SCALES + Q8H_CODES] = src[:, 512:512 + Q8H_CODES]
    out[:, 1, Q8H_SCALES:Q8H_SCALES + Q8H_CODES] = src[:, 512 + Q8H_CODES:512 + 2 * Q8H_CODES]
    return out.reshape(2 * n, CH)


def q8_per_band(K: int) -> int:
    """Half-tiles per 64-row band of a K-wide q8 matrix: four per k-tile."""
    return K // 64


def q8_band_bytes(K: int) -> int:
    return q8_per_band(K) * CH


def q8_perm(nch: int, in_dim: int):
    """pool half-tile index -> (file chunk index, half), the q8 twin of `std_perm`.

    A band is still 64 output rows x in_dim, now `4 * in_dim/256` half-tiles; half-tile c
    inside its band covers rows 16*(c%4) of the band and k-tile c//4. The source is the
    container's raster (file chunk f = rows 32*(f//ncol), cols 256*(f%ncol)), so the 16-row
    slice at band row 16*part lives in file chunk (2*band + part//2) at half part%2.

    `in_dim` has to tile the chunk here for the same reason it does in `std_perm`: `ncol`
    floors, and at a width like 2944 the k-tile index runs past it and the half-tiles alias
    onto the next row block's (OPEN-WIDTH-PAD)."""
    if in_dim % 256:
        raise OpRangeError(f"q8_perm: in_dim={in_dim} is not a whole number of 256-column "
                           f"k-tiles ({in_dim % 256} over); the file raster's column count "
                           f"would floor and the pool half-tiles would alias onto each other")
    ncol = in_dim // 256
    per_band = in_dim // 64
    c = np.arange(nch)
    band, cc = c // per_band, c % per_band
    part, kt = cc % 4, cc // 4
    return (2 * band + part // 2) * ncol + kt, part % 2


def band_rowblock_ktile(nch: int, in_dim: int):
    """pool chunk index -> (32-row block, 256-column k-tile).

    The band law on its own, with no source raster in it: a band is 64 output rows x
    in_dim, so `per_band = in_dim/128` chunks, and chunk i inside its band covers row half
    `i % 2` and k-tile `i // 2` (gemv_q4.h). `std_perm` looks the pair up in the file's
    plain raster and `std_fuse` in GPT-OSS's supertile one; keeping the law in one place is
    what stops the two rasters from drifting apart on the half they share."""
    if in_dim % 256:
        raise OpRangeError(f"band law: in_dim={in_dim} is not a whole number of 256-column "
                           f"k-tiles ({in_dim % 256} over); the file raster's column count "
                           f"would floor and the pool chunks would alias onto each other")
    per_band = in_dim // 128
    c = np.arange(nch)
    return 2 * (c // per_band) + (c % 2), (c % per_band) // 2


def supertile_perm(nrb: int, ncol128: int, rg: int) -> np.ndarray:
    """(32-row block, 128-column block) -> file chunk index, for the supertile raster.

    Every other converter writes a plain raster -- chunk (rb, q) at `rb * ncol + q`.
    `q4nx-build`'s GPT-OSS path groups `rg` consecutive row blocks into a supertile and
    rasters the supertiles instead: row block `rb` is supertile `P = rb // rg` at position
    `F = rb % rg`, and its column block `q` lands at `(P * ncol128 + q) * rg + F`. `rg` is
    4 for the attention projections and the experts, 2 for the `lm_head`.

    Returns [nrb, ncol128]. It is a permutation of `range(nrb * ncol128)` whenever `nrb` is
    a multiple of `rg`, which the caller checks -- a partial supertile would put two row
    blocks on one index."""
    if rg <= 0 or nrb % rg:
        raise OpRangeError(f"supertile_perm: {nrb} row blocks is not a whole number of "
                           f"{rg}-row-block supertiles")
    rb = np.arange(nrb)[:, None]
    q = np.arange(ncol128)[None, :]
    return ((rb // rg) * ncol128 + q) * rg + (rb % rg)


EXPERT_ROLES = ("gate", "up", "down")


def expert_slabs(nslab: int) -> np.ndarray:
    """[3, nslab] -- (role, the projection's own 128-row slab) -> the fused tensor's slab.

    `q4nx-build` fuses one layer's gate, up and down for all its experts into a single
    `ffn_gate_up_down_exps.weight`, shaped [E, 3*nslab, ncol128, rg, 2560]. The slabs are
    NOT the three projections one after another: gate and up ALTERNATE every 128 rows over
    the first 2*nslab, and only then does down follow as one contiguous block.

    Nothing in the container names the order, and the two plausible readings -- this one and
    three concatenated projections -- agree on down and differ on gate and up, so a packer
    that guesses wrong swaps the two halves of every SwiGLU and still produces finite
    activations. What settles it is that the converter writes each expert bias twice, into
    byte 128 of every column-block-0 chunk as well as a named tensor: the bias distinguishes
    gate from up from down BY VALUE, so the reading is measured rather than inferred
    (OPEN-PACK-EXPERT-ORDER)."""
    s = np.arange(nslab)
    return np.stack([2 * s, 2 * s + 1, 2 * nslab + s])


def expert_chunks(nslab: int, ncol128: int, rg: int = 4) -> np.ndarray:
    """[3, nslab*rg, ncol128] -- (role, the projection's 32-row block, column block) ->
    the file chunk index WITHIN one expert. Add `expert * 3 * nslab * ncol128 * rg` for the
    expert's own.

    Composes the slab order above with the supertile raster the same converter writes for
    the attention projections, so the two cannot drift: a projection's row block `j` is its
    slab `j // rg` at position `F = j % rg`, and that slab's place in the fused tensor is
    what `expert_slabs` returns. A logical output row therefore decomposes exactly as
    `(row // 128, (row % 128) // 32, row % 32)` -- slab, quarter, row in chunk."""
    slab = expert_slabs(nslab)                                   # [3, nslab]
    idx = supertile_perm(3 * nslab * rg, ncol128, rg)            # [rowblock, col] -> chunk
    rb = slab[:, :, None] * rg + np.arange(rg)[None, None, :]    # [3, nslab, rg]
    return idx[rb.reshape(3, nslab * rg)]


def fuse_chunks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[n, 2560] + [n, 2560] -> [n, 5120]: the k-tile's low and high 128 columns as one
    pool chunk. Eight byte-slice copies and no arithmetic.

    A 2560-byte chunk is q4_1 in the ordinary layout over 32 rows x 128 columns -- four
    32-column blocks instead of eight -- so the two halves interleave rather than
    concatenate. The meta index is `b * 32 + r`, which puts A's four blocks at metas 0..127
    and B's at 128..255; the nibble raster is `(r//16) * 512*nb + b * 512 + i * 16 + r%16`,
    whose leading term splits each chunk into two 1024-byte planes by row half, so the
    nibbles go A-plane-0, B-plane-0, A-plane-1, B-plane-1. Read back with the pool's own
    `nib * d + m` at (row, block, lane), blocks 0..3 are A's and 4..7 are B's."""
    a, b = _u8(a).reshape(-1, CH_HALF), _u8(b).reshape(-1, CH_HALF)
    if a.shape != b.shape:
        raise ValueError(f"fuse_chunks: {a.shape[0]} low-half chunks against {b.shape[0]} high")
    out = np.empty((a.shape[0], CH), dtype=np.uint8)
    out[:, 0:256], out[:, 256:512] = a[:, 0:256], b[:, 0:256]            # d
    out[:, 512:768], out[:, 768:1024] = a[:, 256:512], b[:, 256:512]     # m
    out[:, 1024:2048], out[:, 2048:3072] = a[:, 512:1536], b[:, 512:1536]      # rows 0..15
    out[:, 3072:4096], out[:, 4096:5120] = a[:, 1536:2560], b[:, 1536:2560]    # rows 16..31
    return out


def fuse_perm(nch: int, in_dim: int, src_dim: int, rg: int):
    """pool chunk index -> (low file chunk, high file chunk) in the supertile raster, with
    `nsrc` -- one past the last real chunk -- standing for a column block the container does
    not have.

    The pool is `in_dim` wide and the container `src_dim`; GPT-OSS ships 2944 against a
    padded 3072, which is 23 real 128-column blocks against the 24 the pool wants. The
    missing block is synthesised as 2560 zero bytes, so the fuse and the pad are one pass
    rather than a pack followed by a zero fill."""
    if src_dim % 128:
        raise OpRangeError(f"std_fuse: src_dim={src_dim} is not a whole number of "
                           f"128-column chunks ({src_dim % 128} over)")
    if src_dim > in_dim:
        raise OpRangeError(f"std_fuse: the container is {src_dim} wide and the pool only "
                           f"{in_dim}; a pool narrower than the container would drop columns")
    rb, kt = band_rowblock_ktile(nch, in_dim)
    ncol128 = src_dim // 128
    nrb = int(rb.max()) + 1 if nch else 0
    idx = supertile_perm(nrb, ncol128, rg)
    nsrc = nrb * ncol128

    def pick(q):
        out = np.full(q.shape, nsrc, dtype=np.int64)
        real = q < ncol128
        out[real] = idx[rb[real], q[real]]
        return out

    return pick(2 * kt), pick(2 * kt + 1), nsrc


def std_perm(nch: int, in_dim: int) -> np.ndarray:
    """pool chunk index -> file chunk index, for a standard [out, in] matmul tensor:
    a band is 64 rows x in_dim = per_band = in_dim/128 chunks; inside its band chunk i
    covers row half i % 2 and k-tile i // 2 (gemv_q4.h's band law, q4_1_pack.chunk_geometry);
    file chunk f covers rows 32*(f//ncol), cols 256*(f%ncol).

    The law phlegm verified against OFLM's captured pools was written as
    cols = 1024*((c//8) % (in//1024)) + 256*((c//2) % 4); for in_dim a multiple of 1024
    that is this same k-tile order (tests/test_pack_plan.py checks the two agree there);
    this form is the one that also holds for in_dim = 2560 or 9728.

    `in_dim` must be a whole number of 256-column k-tiles, and that is checked rather than
    assumed: `ncol` floors, so a width like GPT-OSS's shipped 2944 used to hand back an
    index array that ALIASES -- 1409 distinct file chunks selected for 1472 pool slots --
    and neither apply_op nor anything above it looked. A corrupt pool with nothing raised
    is the one outcome worth a guard (OPEN-WIDTH-PAD)."""
    rb, kt = band_rowblock_ktile(nch, in_dim)
    return rb * (in_dim // 256) + kt


def down_perm(nch: int = 128) -> np.ndarray:
    """One expert's down [HID, FF] slice: pool chunk c <- file chunk 2*rt + cg,
    rt = 4*(c//8) + c%4, cg = (c//4)%2 (validated for [2048, 512])."""
    c = np.arange(nch)
    rt = 4 * (c // 8) + (c % 4)
    return 2 * rt + (c // 4) % 2


def stripe_transpose(in_dim: int = 2048) -> np.ndarray:
    """Inside one 128-row stripe: pool chunk c <- file chunk ncol*(c%4) + c//4."""
    ncol = in_dim // 256
    c = np.arange(4 * ncol)
    return ncol * (c % 4) + c // 4


def _u8(b) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8) if not isinstance(b, np.ndarray) else b.view(np.uint8)


def _chunk_guess(ch: int) -> str:
    if ch == CH_HALF:
        return ("GPT-OSS's 32-row x 128-column chunk, which the `std_fuse` op reads -- the "
                "file raster is a supertile as well as half-width, so this tensor needs that "
                "op rather than this one (OPEN-PACK-CHUNK-FUSE)")
    if ch == 1280:
        return f"a smaller chunk geometry ({ch * 8192 // CH} values per chunk instead of 8192)"
    return "not a chunk format this packer knows"


def _chunk_bytes_of(m, name: str) -> int:
    """The quantized chunk size the container stores `name` in; a container that cannot say
    is read as q4_1, which is what every container predating `chunk_bytes_of` is."""
    get = getattr(m, "chunk_bytes_of", None)
    return (get(name) if callable(get) else 0) or CH


def q4_chunks_of(m, name: str, raw, c0: int = 0, n: int | None = None) -> np.ndarray:
    """Chunks [c0, c0 + n) of `raw` as [n, 5120] q4_1 bytes, whatever the container stores.

    q4_1 is a view; q8 (8704 B chunks) is re-quantized here and Q4_K (4736 B chunks) is
    transcoded, both in batches so a 2048-chunk projection does not build a 70 MB float
    array. Anything else is refused by name, byte count and what the count probably means
    -- the message someone reads when they point the engine at a container this packer
    cannot use."""
    b = _u8(raw)
    ch = _chunk_bytes_of(m, name)
    if ch not in (CH, Q8, Q4K):
        raise ValueError(f"{name}: {ch}-byte quant chunks; the packer reads {CH} (q4_1), {Q8} (q8) "
                         f"and {Q4K} (Q4_K) only -- {ch} is {_chunk_guess(ch)}")
    src = b.reshape(-1, ch)
    sel = src[c0:] if n is None else src[c0:c0 + n]
    if ch == CH:
        return _batched(q4_0_to_q4_1, sel) if is_signed_q4(m, name, sel) else sel
    conv = requant_q4_1 if ch == Q8 else q4k_to_q4_1
    return _batched(conv, sel)


def _batched(conv, sel: np.ndarray) -> np.ndarray:
    """In batches, so a 2048-chunk projection does not build a 70 MB float array."""
    out = np.empty((sel.shape[0], CH), np.uint8)
    for i in range(0, sel.shape[0], 256):
        out[i:i + 256] = conv(sel[i:i + 256])
    return out


def is_signed_q4(m, name: str, sel: np.ndarray) -> bool:
    """Is this 5120-byte tensor the SIGNED quantiser rather than q4_1?

    Two formats share the chunk. q4_1 stores (scale, min) per 32-value block and reads
    w = d * q + min with q an unsigned nibble, 0..15. Some containers -- Qwen2.5 is the one
    that found this -- use the same chunk for w = d * int4(q), a signed nibble -8..7 with
    no min, and write every min as zero. Read the wrong way every block comes out
    one-sided, sharing the sign of its scale, at about 2.7x the right spread; the replica
    misreads it identically, so it agrees with the kernels to the bit and the model answers
    with noise.

    A container that SAYS which it is wins: `quant_format_of` is the hook for that, and no
    container implements it yet. Failing that, 256 exactly-zero mins in a chunk is the
    signal -- a real q4_1 tensor does not manage that, because a min is a block's own
    minimum and every one of them being exactly 0.0 does not happen to real weights.
    """
    say = getattr(m, "quant_format_of", None)
    declared = say(name) if callable(say) else None
    if declared:
        return declared == "q4_0"
    return bool(sel.size) and bool((sel[0, 512:1024].view(np.uint16) == 0).all())


def q4_0_to_q4_1(chunks) -> np.ndarray:
    """[n, 5120] signed-nibble chunk bytes -> [n, 5120] q4_1 chunk bytes.

    int4(q) == (q ^ 8) - 8, so flipping bit 3 of every nibble turns two's complement into
    offset binary and w = d * int4(q) becomes w = d * q' + (-8 * d). Writing -8 * d into
    the min slot leaves the GEMV, the pool and the replica reading exactly the right
    values, and nothing downstream learns a new format. Both halves are exact: the bit flip
    is a relabelling, and -8 * d only moves a bf16 exponent.

    src/open_qwen36/pools.cpp does the same and must agree byte for byte.
    """
    src = _u8(chunks).reshape(-1, CH)
    out = src.copy()
    d = _bf16_to_f32(np.ascontiguousarray(src[:, :512]).view(np.uint16))
    out[:, 512:1024] = _bf16_rne(-8.0 * d).view(np.uint8).reshape(-1, 512)
    out[:, 1024:] = src[:, 1024:] ^ 0x88          # bit 3 of the low nibble and of the high
    return out


def q8_chunks_of(m, name: str, raw, c0: int = 0, n: int | None = None) -> np.ndarray:
    """Chunks [c0, c0 + n) of a q8 tensor as [n, 8704] bytes. A source that is not q8 is
    refused by name: the kernel set was built to stream this projection at q8, and packing
    the q4_1 the container actually holds would put the wrong bytes in front of a q8 GEMV
    (OPEN-QUANT-Q8)."""
    get = getattr(m, "chunk_bytes_of", None)
    ch = (get(name) if callable(get) else 0) or CH
    if ch != Q8:
        raise ValueError(f"{name}: the kernel set streams this projection at q8 ({Q8}-byte chunks) but "
                         f"the container stores it in {ch}-byte chunks -- re-export the kernels for this "
                         f"container, or force the q4_1 fallback (OPEN_KERNELS_FORCE_Q4_1=1)")
    src = _u8(raw).reshape(-1, Q8)
    return src[c0:] if n is None else src[c0:c0 + n]


def _name(op: dict, key: str, layer: int) -> str:
    return op[key].replace("{l}", str(layer))


def _raw(m, name: str):
    """The tensor's bytes, or a message naming the one the container lacks.

    This is where the head of a tied model is checked: a plan always names
    `lm_head.weight` (the recipes never fold the head into the embedding table),
    and every container we pack from materialises it -- OFLM's `.q4nx` even for
    Llama 3.2 and the small Qwen3 models, whose config.json says
    `tie_word_embeddings: true`. A container that really is tied fails here,
    naming the tensor, rather than producing a pool of zeros."""
    try:
        return m.raw(name)
    except KeyError:
        raise KeyError(f"the container has no tensor {name!r} "
                       f"(a tied head must be materialised as its own q4 tensor)") from None


def apply_op(op: dict, m, layer: int, dst: np.ndarray) -> None:
    kind = op["op"]
    if kind == "std_perm":
        name = _name(op, "tensor", layer)
        if not op.get("nch") or not op.get("in_dim"):
            raise ValueError(f"std_perm {name} without nch / in_dim")
        c0 = op.get("chunk0", 0)
        sel = q4_chunks_of(m, name, _raw(m, name), c0, op["nch"])
        if sel.shape[0] != op["nch"]:
            raise ValueError(f"{op['tensor']}: too few chunks, need {c0 + op['nch']}")
        n = op["nch"] * CH
        dst[op["dst"]:op["dst"] + n] = sel[std_perm(op["nch"], op["in_dim"])].reshape(-1)
    elif kind == "std_fuse":
        # OPEN-PACK-CHUNK-FUSE: a std_perm band whose source chunks are half-width. The
        # container holds 32 rows x 128 columns per chunk in the supertile raster
        # q4nx-build writes for GPT-OSS, so each pool chunk is the k-tile's two 128-column
        # halves fused -- eight byte-slice copies, no arithmetic -- and a column block past
        # the container's own width is synthesised as zeros, which is also the pad from the
        # container's 2944 to the pool's 3072.
        name = _name(op, "tensor", layer)
        if not op.get("nch") or not op.get("in_dim") or not op.get("src_dim"):
            raise ValueError(f"std_fuse {name} without nch / in_dim / src_dim")
        src_ch = _chunk_bytes_of(m, name)
        if src_ch != CH_HALF:
            raise ValueError(f"{name}: std_fuse reads {CH_HALF}-byte chunks (32 rows x 128 "
                             f"columns) and the container stores it in {src_ch}-byte ones")
        raw = _u8(_raw(m, name)).reshape(-1, CH_HALF)
        lo, hi, nsrc = fuse_perm(op["nch"], op["in_dim"], op["src_dim"], op.get("rg", 4))
        if raw.shape[0] != nsrc:
            raise ValueError(f"{name}: {raw.shape[0]} chunks of {CH_HALF} B, but a "
                             f"{op['src_dim']}-wide tensor covering {op['nch']} pool chunks "
                             f"needs exactly {nsrc}")
        src = np.concatenate([raw, np.zeros((1, CH_HALF), np.uint8)])   # the synthesised block
        n = op["nch"] * CH
        dst[op["dst"]:op["dst"] + n] = fuse_chunks(src[lo], src[hi]).reshape(-1)
    elif kind == "q8_perm":
        # the q8 twin of std_perm: `nch` is the count of POOL half-tiles (5120 B each, twice
        # the q4_1 bytes of the same tensor); `chunk0` is a SOURCE file-chunk offset, as it is
        # for std_perm, so the fused [q | gate] split reads the same way in both formats.
        name = _name(op, "tensor", layer)
        if not op.get("nch") or not op.get("in_dim"):
            raise ValueError(f"q8_perm {name} without nch / in_dim")
        nch = op["nch"]
        if nch % 2:
            raise ValueError(f"q8_perm {name}: {nch} half-tiles is not a whole number of chunks")
        sel = q8_chunks_of(m, name, _raw(m, name), op.get("chunk0", 0), nch // 2)
        if sel.shape[0] != nch // 2:
            raise ValueError(f"{op['tensor']}: too few chunks, need {op.get('chunk0', 0) + nch // 2}")
        files, halves = q8_perm(nch, op["in_dim"])
        dst[op["dst"]:op["dst"] + nch * CH] = q8_half_tiles(sel)[2 * files + halves].reshape(-1)
    elif kind == "expert_stripes":
        un, gn = _name(op, "up", layer), _name(op, "gate", layer)
        up = q4_chunks_of(m, un, _raw(m, un))
        gt = q4_chunks_of(m, gn, _raw(m, gn))
        S, ns, E = op["stripe_bytes"], op["stripes"], op["experts"]
        nchs = S // CH
        tp = stripe_transpose(op["in_dim"])
        base = op["dst"]
        for e in range(E):
            for k in range(ns):
                c = (ns * e + k) * nchs
                d = base + (2 * ns * e + 2 * k) * S
                dst[d:d + S] = up[c:c + nchs][tp].reshape(-1)
                dst[d + S:d + 2 * S] = gt[c:c + nchs][tp].reshape(-1)
    elif kind == "expert_down":
        name = _name(op, "tensor", layer)
        dn = q4_chunks_of(m, name, _raw(m, name))
        B, E = op["expert_bytes"], op["experts"]
        nchs = B // CH
        dp = down_perm(nchs)
        base = op["dst"]
        for e in range(E):
            dst[base + e * B:base + (e + 1) * B] = dn[e * nchs:(e + 1) * nchs][dp].reshape(-1)
    elif kind == "put":
        b = _u8(_raw(m, _name(op, "tensor", layer)))
        if len(b) > op["cap"]:
            raise ValueError(f"{op['tensor']}: {len(b)} B does not fit its {op['cap']} B slot")
        dst[op["dst"]:op["dst"] + len(b)] = b
    elif kind == "lmhead_q8":
        # The q8 lm_head's 128-row supertile order. The file holds chunk (rowblock32, ktile) at
        # rowblock32 * nk + ktile with nk = K / 256 k-tiles; a band is 128 rows = 4 row quarters
        # x nk k-tiles, and the kernel reads pool chunk c of a band as (quarter = c % 4,
        # ktile = c // 4). So  pool k <- file (4 * (k // per_band) + k % 4) * nk + (k % per_band) // 4,
        # per_band = 4 * nk. nk was hardcoded to 8 (K = 2048) until OPEN-FAMILY-QWEN35's 4B slice
        # (K = 2560) came back with correct residuals and garbage logits.
        ch = op["chunk_bytes"]
        in_dim = op.get("in_dim")
        if not in_dim:
            raise ValueError(f"lmhead_q8 {_name(op, 'tensor', layer)} without in_dim (the hidden width)")
        nk = in_dim // 256
        per_band = 4 * nk
        raw = _u8(_raw(m, _name(op, "tensor", layer))).reshape(-1, ch)
        k = np.arange(raw.shape[0])
        s, r = k // per_band, k % per_band
        perm = (4 * s + r % 4) * nk + r // 4
        n = raw.shape[0] * ch
        if op["dst"] + n > len(dst):
            raise ValueError("lm_head larger than its pool")
        dst[op["dst"]:op["dst"] + n] = raw[perm].reshape(-1)
    elif kind == "transpose":
        # a small [rows, cols] tensor of `elem`-byte values -> [cols, rows]. `dst_rows`, when
        # given, widens the destination row to that many values and zeroes the tail -- the
        # 16-head DeltaNet's alpha / beta go into a 32-lane accumulator (recipes/qwen35.py).
        rows, cols, elem = op.get("rows"), op.get("cols"), op.get("elem", 2)
        dr = op.get("dst_rows") or rows
        name = _name(op, "tensor", layer)
        if not rows or not cols:
            raise ValueError(f"transpose {name} without rows / cols")
        if dr < rows:
            raise ValueError(f"transpose {name}: dst_rows {dr} is narrower than rows {rows}")
        b = _u8(_raw(m, name))
        if len(b) != rows * cols * elem:
            raise ValueError(f"{op['tensor']}: {len(b)} B is not a [{rows}, {cols}] tensor of {elem}-byte values")
        w = np.zeros((cols, dr, elem), np.uint8)
        w[:, :rows] = b.reshape(rows, cols, elem).transpose(1, 0, 2)
        dst[op["dst"]:op["dst"] + w.size] = w.reshape(-1)
    elif kind == "conv_transpose":
        b = _u8(_raw(m, _name(op, "tensor", layer)))
        taps, groups, width = op["taps"], op["groups"], op["width"]
        if len(b) != taps * groups * width * 2:
            raise ValueError(f"{op['tensor']}: {len(b)} B is not bf16[{taps}, {groups * width}]")
        w = b.view(np.uint16).reshape(taps, groups, width).transpose(1, 0, 2).reshape(-1).view(np.uint8)
        dst[op["dst"]:op["dst"] + len(w)] = w
    else:
        raise ValueError(f"unknown pack op {kind!r}")


def build_layer_pool(plan: dict, layer_type: str, m, layer: int, out: np.ndarray | None = None) -> np.ndarray:
    pool = np.zeros(plan["pool_bytes"], np.uint8) if out is None else out
    if out is not None:
        pool[:] = 0
    for op in plan["layer_types"][layer_type]["pool"]:
        apply_op(op, m, layer, pool)
    return pool


def build_consts(plan: dict, layer_type: str, m, layer: int, nbytes: int) -> np.ndarray:
    c = np.zeros(nbytes, np.uint8)
    for op in plan["layer_types"][layer_type]["consts"]:
        apply_op(op, m, layer, c)
    return c


def build_lmhead_pool(plan: dict, m) -> np.ndarray:
    """The lm_head pool: the plan's ops (q8 supertiles for the 27B, a std_perm for a q4 head)."""
    lm = plan["lm_head"]
    out = np.zeros(lm["pool_bytes"], np.uint8)
    for op in lm["ops"]:
        apply_op(op, m, 0, out)
    return out


def window_rows(p, window: int):
    """(valid, nf) for position p: the cached rows the attention core sees are [s, p) with
    s = max(0, p - (window - 1)) (window 0: all of them); it streams nf = max(1, p - s) rows and
    masks the ones at index >= valid = p - s (the dummy row at position 0)."""
    p = np.asarray(p)
    s = np.maximum(0, p - (window - 1)) if window else np.zeros_like(p)
    valid = p - s
    return valid, np.maximum(valid, 1)


def ptab(rows: int, rotary_dim: int, theta: float, ptab_row: int = 1024, inv_freq=None, window: int = 0,
         scale: float = 1.0, long_inv_freq=None, switch_row: int | None = None) -> np.ndarray:
    """The position record table: row p = [i32 valid | i32 nf | cos f32[rot/2] @512 | sin f32[rot/2]
    right after the cos, @512 + 2*rot] for the RoPE over the first `rotary_dim` dims of a head (half-split
    pairs (i, i + rot/2)); attn.h reads the rot floats at +512 as [cos | sin]. `inv_freq` (rot/2 values,
    ModelSpec.rope_inv_freq -- Llama 3's scaling lives there) defaults to theta^(-2i/rot). `window`
    (rows, 0 = unbounded) makes the record count the sliding window's rows (window_rows). `scale`
    multiplies cos and sin (longrope's attention factor; 1.0 for every other family).

    `long_inv_freq` (Phi-3's longrope only): a second frequency table, used for row p once
    p >= switch_row instead of `inv_freq`. HF picks between the two per forward call from the
    running sequence length; a resident table cannot re-select per call, but this engine computes
    one row per token as the context grows, so row p carries whichever table a forward call at
    sequence length p + 1 would have picked, and that choice never moves once a row is written
    (an earlier row's cached K is not rewound when the context later crosses the threshold)."""
    half = rotary_dim // 2
    if 512 + 8 * half > ptab_row:
        raise ValueError(f"a rotary dim of {rotary_dim} does not fit a {ptab_row}-byte position record")
    if (long_inv_freq is None) != (switch_row is None):
        raise ValueError("ptab: long_inv_freq and switch_row must be given together")
    t = np.zeros((rows, ptab_row), np.uint8)
    p = np.arange(rows)
    valid, nf = window_rows(p, window)
    t[:, :8] = np.stack([valid, nf], 1).astype(np.int32).view(np.uint8)
    f = np.asarray(inv_freq, np.float64) if inv_freq is not None else theta ** (-np.arange(half) / half)
    if len(f) != half:
        raise ValueError(f"inv_freq has {len(f)} values, the rotary dim wants {half}")
    if long_inv_freq is not None:
        lf = np.asarray(long_inv_freq, np.float64)
        if len(lf) != half:
            raise ValueError(f"long_inv_freq has {len(lf)} values, the rotary dim wants {half}")
        f = np.where((p >= switch_row)[:, None], lf[None, :], f[None, :])
        ang = p[:, None] * f
    else:
        ang = p[:, None] * f[None, :]
    t[:, 512:512 + 4 * half] = (scale * np.cos(ang)).astype(np.float32).view(np.uint8)
    t[:, 512 + 4 * half:512 + 8 * half] = (scale * np.sin(ang)).astype(np.float32).view(np.uint8)
    return t.reshape(-1)
