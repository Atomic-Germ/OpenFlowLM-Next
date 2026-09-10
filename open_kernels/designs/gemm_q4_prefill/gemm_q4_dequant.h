#pragma once
//===- gemm_q4_dequant.h ------------------------------------*- C++ -*-===//
//
// task 0167 stage 14: dequantize ONE 64-wide K-slice of a 64-row q4_1 pool
// BAND (see ../../q4_1_pack.py / ../gemv_q4/gemv_q4.h for the byte layout --
// identical convention, reused verbatim here, not reproduced from memory) into
// a bf16 [64,64] scratch tile laid out in the EXACT block order aie2p's
// matmul_vectorized_2x2_mmul (mlir-aie aie_kernels/aie2p/mm.cc) reads its A
// operand in: contiguous (r=4)x(s=8) blocks, block_index = row_block*colA +
// col_block, colA = k_tile/s = 8. Confirmed by reading mm.cc directly:
//   pA1 = pA + (z*colA)*MMUL::size_A;               // z = row-block index
//   for (i = 0; i < colA; ++i) { A0 = load_v<size_A>(pA1); pA1 += size_A; }
// i.e. A is a flat run of (dim_m/r * dim_k/s) contiguous r*s=32-element
// blocks, block order row-block-major then col-block, each block itself
// row-major (r rows of s elements) -- exactly what
// NpuEmbeddings/experiments/m5-pretiled-gemm/gemm_pretiled.py's own A-side
// dims_to_stream = [(m//r, r*k), (k//s, s), (r, k), (s, 1)] produces for the
// RAW (non-quantised) A operand via DMA. Here there is no DMA reorder
// available (the source bytes are opaque q4 pool chunks, not a plain [m,k]
// float array) so the SAME permutation is produced by the kernel itself, in
// two passes:
//
//   Pass 1 (gqd_gather_nibbles): pure INTEGER scalar gather of raw nibble
//   VALUES (0..15) into a plain ROW-MAJOR uint8[64,64] scratch. Index math +
//   one byte load + mask/shift + one byte store per element -- no bf16/float
//   anywhere, so this does NOT hit CLAUDE.md trap 5 (scalar float lowers to a
//   1617x-slower __mulsf3 call); integer scalar ops are ordinary instructions.
//
//   Pass 2 (gqd_dequant_block): dequantise (nib*d+m, vectorised bf16
//   arithmetic, 8 lanes at a time) AND perform the block reorder in the SAME
//   pass, using only 8-wide CONTIGUOUS vector loads/stores -- the reorder is
//   expressed entirely through the destination-offset formula
//   (block_index*32 + li*8), never through a vector shuffle/permute. This is
//   the deliberate simplification named in the task brief ("correctness
//   matters far more than peak efficiency here"): it trades some throughput
//   for a permutation that is auditable by inspection rather than by getting
//   an aie::shuffle bit pattern right blind.
//
// Byte layout (q4_1_pack.py / gemv_q4.h, unchanged):
//   one CHUNK = 5120 B = 32 rows x 256 K:
//     d[256] bf16 at [0:512]     index kb*32+rl   (kb=k//32, rl=row-in-chunk)
//     m[256] bf16 at [512:1024]  same index
//     nib[4096] B at [1024:5120] nibble p=(rl/16)*4096+k*16+(rl%16)
//                                p even -> low nibble, odd -> high nibble
//   one BAND (RS=2 standard layout) = 2 chunks stacked: chunk 0 = rows 0..31,
//   chunk 1 = rows 32..63, i.e. band bytes = tile[0:5120] | tile[5120:10240] --
//   and these two chunks are ADJACENT in the pool byte stream for a fixed
//   (row-block, k-group) by construction of pack_q4_1_pool's chunk order
//   (verified against chunk_geometry in q4_1_pack.py: for RS=2, chunk c =
//   band*per_band + ci, and the two RS=2 chunks of one k-group are
//   consecutive chunk indices), so ONE 10240 B contiguous DMA read is exactly
//   one band-k-group -- no host-side repacking needed beyond the existing
//   pack_q4_1_pool(rs=2) pool bytes.
//
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned GQD_M = 64;    // rows per band = m (one AIE row's m-tile)
static constexpr unsigned GQD_K64 = 64;  // one matmul k-tile (= s * colA)
static constexpr unsigned GQD_R = 4;     // mac_dims r (aie2p bf16, non-emulated: (4,8,8))
static constexpr unsigned GQD_S = 8;     // mac_dims s
static constexpr unsigned GQD_COLA = GQD_K64 / GQD_S;  // 8 col-blocks per k64-tile
static constexpr unsigned GQD_CHUNK_BYTES = 5120;
static constexpr unsigned GQD_META_BYTES = 1024;
static constexpr unsigned GQD_D_OFF = 0;
static constexpr unsigned GQD_M_OFF = 512;
static constexpr unsigned GQD_BAND_BYTES = 2 * GQD_CHUNK_BYTES;  // 10240

// Pass 1: gather nibble VALUES for k-slice `ky` (0..3, each 64 of the band's
// 256 K) into row-major nib_scr[64*64] uint8.
//
// 0167 stage 16, lever 1: this was originally a pure-integer SCALAR gather
// (one byte load + mask/shift + one byte store per element, ~4096 iterations
// per ky call) and measured as the dominant cost of the whole design
// (TFLOPS/ms-per-token flat across T, the signature of a fixed-per-element
// bottleneck -- see 0167 Stage 14/15). Replaced with the SAME masking
// technique ../gemv_q4/gemv_q4.h's `gemv_q4_tile` already uses on this exact
// q4_1 pool byte layout (read there before writing a third form of this
// unpack, per the task brief): raw nibble bytes are packed
// [k_local][row_pair] with row_pair FASTEST (see the file banner's byte
// layout note -- p(r,k) = k*16+r16, so 16 consecutive nibbles = 16 rows at
// one k), so a 64-byte contiguous load covers an 8x8 block of (k_local,
// row_pair). Mask 0x0F/0xF0 in two vector ops (no shift -- exactly
// gemv_q4.h's `e`/`o` split; the high-nibble mask leaves the value as
// nib*16, uncorrected, and Pass 2 divides it back out via `to_float`'s shift
// argument, the SAME "mask + to_float shift" idiom gemv_q4.h's own header
// banner cites as the fix for "aie::downshift on uint8 is an error").
//
// The masked 8x8 block is [k_local][row_pair]-major (row_pair fastest); we
// need [row_pair][k_local]-major (one row's 8 K values contiguous) to store
// it as one vector per row. `aie::transpose(v, 8, 8)` (aie_api's general
// reshape-group transpose, used internally by aie2p's own bf16/fp32 mmul
// operand prep in mmul_bf16_bf16.hpp/mmul_fp32_fp32.hpp -- not a bespoke
// trick) does exactly that reorder in ONE hardware shuffle instruction for
// an 8-bit, 64-element vector at compile-time-constant row=8
// (detail/aie2/transpose.hpp's `transpose_bits_impl<8,T,64>` skips its
// emulated second shuffle when `row` is a compile-time constant not equal to
// 2 or Elems/2) -- so each 8x8 block costs 1 load + 2 masks + 2 transposes,
// then 16 stores of the resulting 8-wide per-row slices (8 even rows via
// `et.extract<8>(rp)`, 8 odd rows via `ot.extract<8>(rp)`). No shuffle or
// permute was needed for the (r,s) BLOCK reorder Pass 2 already does by
// destination-offset formula -- this transpose is a separate, smaller
// reorder (K vs. row-pair inside one raw byte octet) that the raw pool byte
// layout forces regardless of the mm.cc A-operand block order.
//
// GQD_SCALAR_GATHER reverts to the original scalar loop, kept for ablation
// (0167 stage 16 methodology: one lever at a time, before/after measured).
#ifdef GQD_SCALAR_GATHER
__attribute__((noinline)) inline void gqd_gather_nibbles(
    const uint8_t *__restrict band, unsigned ky, uint8_t *__restrict nib_scr) {
#pragma clang loop unroll(disable)
  for (unsigned row = 0; row < GQD_M; ++row) {
    const unsigned part = row / 32;   // which stacked chunk (0/1)
    const unsigned rl = row % 32;     // row within that chunk
    const uint8_t *__restrict chunk = band + part * GQD_CHUNK_BYTES + GQD_META_BYTES;
    const unsigned half = rl / 16;    // nib0 (0) / nib1 (1)
    const unsigned r16 = rl % 16;
    const uint8_t *__restrict nibp = chunk + half * 2048;
#pragma clang loop unroll(disable)
    for (unsigned kl = 0; kl < GQD_K64; ++kl) {
      const unsigned k_abs = ky * GQD_K64 + kl;  // 0..255 within the chunk
      const unsigned p = k_abs * 16 + r16;        // nibble index (half already folded into nibp)
      const uint8_t byte = nibp[p >> 1];
      const uint8_t val = (p & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
      nib_scr[row * GQD_K64 + kl] = val;
    }
  }
}
#else
__attribute__((noinline)) inline void gqd_gather_nibbles(
    const uint8_t *__restrict band, unsigned ky, uint8_t *__restrict nib_scr) {
  // band-relative byte offset of nib0 (rows 0..15 of a chunk); nib1 is +2048.
  //
  // Sub-block width is 16 K (not 8): `aie::vector<uint8_t,8>` does not
  // compile (Stage 14 problem 1 -- aie2p's uint8 vector_storage is only
  // defined at {16,32,64,128+}), so `et.extract<8>(rp)` below is the SAME
  // illegal-width instantiation, just reached via `aie::transpose` instead
  // of the old scalar loop. Loading 128 raw bytes (16 k_local x 8 row_pair)
  // and transposing as a 16x8 matrix gives 16-wide per-row slices instead
  // (a legal uint8 width), which also halves the sub-loop trip count and the
  // store count versus an 8-wide attempt, for the same total bytes moved.
#pragma clang loop unroll(disable)
  for (unsigned part = 0; part < 2; ++part) {
    const uint8_t *__restrict chunk = band + part * GQD_CHUNK_BYTES + GQD_META_BYTES;
#pragma clang loop unroll(disable)
    for (unsigned half = 0; half < 2; ++half) {
      const uint8_t *__restrict nibp = chunk + half * 2048;
#pragma clang loop unroll(disable)
      for (unsigned sub = 0; sub < GQD_K64 / 16; ++sub) {
        // 128 contiguous bytes = 16 k_local x 8 row_pair (row_pair fastest).
        const unsigned k_byte_off = ky * (GQD_K64 * 8) + sub * 128;
        const aie::vector<uint8_t, 128> q = aie::load_v<128>(nibp + k_byte_off);
        const aie::vector<uint8_t, 128> e = aie::bit_and((uint8_t)0x0F, q);  // rows 2*rp    (true value)
        const aie::vector<uint8_t, 128> o = aie::bit_and((uint8_t)0xF0, q);  // rows 2*rp+1  (value*16)
        const aie::vector<uint8_t, 128> et = aie::transpose(e, 16, 8);  // -> [row_pair][k_local], 16-wide groups
        const aie::vector<uint8_t, 128> ot = aie::transpose(o, 16, 8);
        // 0167 stage 16, lever 4: `rp` is a small (8) fixed trip count, but
        // its use as `et.extract<16>(rp)`'s runtime index compiled to an
        // indirect-jump dispatch (a `.LJTI2_0` jump table selecting among 8
        // shuffle variants -- confirmed in the .o disassembly, `mov dj5;
        // lda p5,[p2,dj5]; j p5`), i.e. exactly skill lever #4's "branch
        // inside the inner loop" -- unroll(full) makes every `rp` a compile-
        // time literal so `extract<16>` folds to one fixed shuffle per
        // instance and the jump table disappears (verified in the .o below).
#pragma clang loop unroll(full)
        for (unsigned rp = 0; rp < 8; ++rp) {
          const unsigned row_e = part * 32 + half * 16 + 2 * rp;
          aie::store_v(nib_scr + row_e * GQD_K64 + sub * 16, et.template extract<16>(rp));
          aie::store_v(nib_scr + (row_e + 1) * GQD_K64 + sub * 16, ot.template extract<16>(rp));
        }
      }
    }
  }
}
#endif

// Pass 2: dequantise + block-reorder into scratch_out (bf16, GQD_M*GQD_K64
// elements, in the (r,s)-blocked order mm.cc's A operand expects).
//
// aie2p's minimum native vector width is 16 for uint8 and 8 for bf16
// (aie_api/detail/aie2/vector_native_types.hpp) -- an 8-wide uint8 vector
// does not compile at all (confirmed the hard way: "implicit instantiation
// of undefined template vector_storage<unsigned char,8>"). So nibbles are
// loaded/converted GQD_W=16 at a time (two adjacent jb col-blocks' worth --
// legal since kb_local, hence d/m, is constant across 16 K: 16 divides the
// 32-wide d/m block), then the resulting 16-wide bf16 vector is split into
// two 8-wide halves via .extract<8>() for the two (non-contiguous)
// destination blocks -- bf16 8-wide IS a native width, so this extract/store
// is a real vector op, not a scalar one.
static constexpr unsigned GQD_W = 16;  // load/convert granularity (2 jb-groups)
__attribute__((noinline)) inline void gqd_dequant_block(
    const uint8_t *__restrict band, unsigned ky, const uint8_t *__restrict nib_scr,
    bfloat16 *__restrict scratch_out) {
#pragma clang loop unroll(disable)
  for (unsigned ib = 0; ib < GQD_M / GQD_R; ++ib) {  // 16 row-blocks
    // 0167 stage 16, lever 4: TRIED unroll(full) here too (jp is a small,
    // fixed 4-iteration loop) and REVERTED -- measured flat-to-worse
    // (2.446-2.492ms vs 2.414-2.437ms median warm at qkv/T=256, i.e. inside
    // noise at best), unlike the `rp` loop above. Unsurprising in hindsight:
    // jp's body has no runtime-indexed extract to fold to a compile-time
    // literal (jb0/kb_local are only address arithmetic, already cheap), so
    // there was no branch/dispatch to remove -- unroll(full) here only grew
    // code size. Left disabled; see the task report for the measurement.
#pragma clang loop unroll(disable)
    for (unsigned jp = 0; jp < GQD_COLA / 2; ++jp) {  // 4 pairs of col-blocks (16 K each)
      const unsigned jb0 = jp * 2;
      const unsigned block0 = ib * GQD_COLA + jb0;
      const unsigned block1 = block0 + 1;
      bfloat16 *__restrict dest0 = scratch_out + block0 * (GQD_R * GQD_S);
      bfloat16 *__restrict dest1 = scratch_out + block1 * (GQD_R * GQD_S);
      // kb_local: constant across this 16-wide K run (16 divides the 32-wide
      // d/m block).
      const unsigned kb_local = (ky * GQD_K64 + jb0 * GQD_S) / 32;
#pragma clang loop unroll(full)
      for (unsigned li = 0; li < GQD_R; ++li) {
        const unsigned row = ib * GQD_R + li;
        const unsigned part = row / 32;
        const unsigned rl = row % 32;
        const uint8_t *__restrict chunk = band + part * GQD_CHUNK_BYTES;
        const bfloat16 d_val = *(const bfloat16 *)(chunk + GQD_D_OFF + (kb_local * 32 + rl) * 2);
        const bfloat16 m_val = *(const bfloat16 *)(chunk + GQD_M_OFF + (kb_local * 32 + rl) * 2);

        const aie::vector<uint8_t, GQD_W> nibv =
            aie::load_v<GQD_W>(nib_scr + row * GQD_K64 + jb0 * GQD_S);
        // Gen2 (aie2p) to_float accepts uint8 input directly (values 0..15
        // are exact in bf16, shift=0).
#ifdef GQD_SCALAR_GATHER
        const aie::vector<bfloat16, GQD_W> nibbf = aie::to_float<bfloat16>(nibv, 0);
#else
        // The vectorized gqd_gather_nibbles (default) stores ODD rows as
        // nib*16 (the raw 0xF0-masked byte value, never shifted -- see its
        // own comment) rather than true 0..15 -- `to_float`'s shift argument
        // divides it back out here, the SAME "mask + to_float shift" fix
        // ../gemv_q4/gemv_q4.h's header banner names for this exact
        // aie::downshift-on-uint8 limitation (there folded into a `dsc` bf16
        // multiply instead; either form is equivalent). `row = ib*GQD_R+li`
        // and `ib*GQD_R` is always even (GQD_R=4), so row's parity equals
        // li's -- a compile-time constant under this loop's `unroll(full)`,
        // so the shift folds to a compile-time 0 or 4 per unrolled instance
        // with no runtime branch.
        const aie::vector<bfloat16, GQD_W> nibbf =
            aie::to_float<bfloat16>(nibv, (li & 1) ? 4 : 0);
#endif

        const aie::vector<bfloat16, GQD_W> dscale = aie::broadcast<bfloat16, GQD_W>(d_val);
        const aie::vector<bfloat16, GQD_W> mscale = aie::broadcast<bfloat16, GQD_W>(m_val);
        const aie::vector<bfloat16, GQD_W> prod =
            aie::mul(nibbf, dscale).template to_vector<bfloat16>();
        const aie::vector<bfloat16, GQD_W> val = aie::add(prod, mscale);

        aie::store_v(dest0 + li * GQD_S, val.template extract<GQD_S>(0));
        aie::store_v(dest1 + li * GQD_S, val.template extract<GQD_S>(1));
      }
    }
  }
}

// One entry point: dequantise k-slice `ky` (0..3) of `band` (10240 B, one
// pool-order band-k-group) into `scratch` (bf16[64*64], block-ordered A
// operand). `nib_scr` is a caller-provided uint8[64*64] scratch buffer
// (Pass 1's row-major intermediate) -- kept as a separate argument rather
// than a local array so its L1 footprint is visible at the call site.
// 0167 stage 16: GQD_NULL_{GATHER,DEQUANT} skip Pass 1 / Pass 2's compute for
// a TIMING-ONLY ablation (attribution, not correctness -- output is garbage
// with either defined; never build a correctness-gate run with these set).
__attribute__((noinline)) inline void gqd_dequant_ky(
    const uint8_t *__restrict band, unsigned ky, uint8_t *__restrict nib_scr,
    bfloat16 *__restrict scratch) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
#ifndef GQD_NULL_GATHER
  gqd_gather_nibbles(band, ky, nib_scr);
#endif
#ifndef GQD_NULL_DEQUANT
  gqd_dequant_block(band, ky, nib_scr, scratch);
#endif
  event1();
}
