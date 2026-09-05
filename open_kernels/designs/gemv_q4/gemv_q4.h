#pragma once
//===- gemv_q4.h -------------------------------------------*- C++ -*-===//
//
// q4_1 GEMV on the AIE core against phlegm's POOL-ORDER chunks:
//   y[N] = W[N, K] @ x[K],  W in FLM's q4 form, chunks in the layer-pool order
//   the closed kernel consumes (npu-engine/src/pools.rs std_perm), so the same
//   kernel later streams straight out of the resident 512 MB pool.
//
// Inner tile arithmetic ported from vegah/LLMNpuTest designs/granite_gemv
// (Apache-2.0, see ../../LICENSE.LLMNpuTest). The chunk layout is identical:
//
//   chunk = 32 output rows x 256 K, 5120 B:
//     d  [256] bf16 at [0    : 512]    index kb*32 + r        (r = 0..31)
//     m  [256] bf16 at [512  : 1024]   same index
//     nib[4096] B at [1024 : 5120]     nibble (r/16)*4096 + k*16 + (r%16)
//                                      = elem(r, k); even index = low nibble
//   value = nib*d + m
//
// (npu-engine/src/q4nx.rs: nibble[(r/16)*4096 + bc*512 + i*16 + r%16], with
// k = bc*32 + i -- the same thing.)
//
// POOL ORDER (pools.rs std_perm, K = in_dim):
//   band = K/128 consecutive chunks = 64 output rows x K.
//   chunk c inside its band: row half = c % 2 (rows 0..31 / 32..63),
//                            k-tile   = c / 2 (cols 256*(c/2) ..).
// So one band is consumed as `per_band` chunks with a 64-float accumulator,
// and the entry point for chunk group g just walks (half, kt) off the index.
//
// Row index is the FASTEST axis inside a chunk: 16 consecutive nibbles are 16
// different rows at one k, so the 16 scales/mins load once per (rb, kb) and the
// inner loop is mask, to_float with a shifted binary point, zip, mac. No gather.
//
// The Q4_1 minimum factors out of the block sum:
//   y[r] = sum_kb { d[kb][r] * sum_{k in kb} nib[k][r]*x[k]  +  m[kb][r] * sum_{k in kb} x[k] }
//
// TRAPS (LLMNpuTest CLAUDE.md): default rounding is floor -> set conv_even;
// AIE2P has NO fp32 vector multiply (returns zero silently) -> split fp32 into
// two bf16 halves; no scalar float; aie::downshift on uint8 is an error -> mask
// + to_float shift; entry points one per TU; body noinline+inline (COMDAT) so
// the 16 KB program memory holds one copy.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gemv_tab.h"   // the activation's int16 block table (shared with lm_head_q8)

static constexpr unsigned kRows = 32;         // output rows per chunk
static constexpr unsigned kBandRows = 64;     // output rows per pool band
static constexpr unsigned kKBlocks = 8;       // 32-wide K blocks per chunk
static constexpr unsigned kKInBlock = 32;
static constexpr unsigned kTileK = 256;       // K per chunk
static constexpr unsigned kDBytes = 512;
static constexpr unsigned kMetaBytes = 1024;
static constexpr unsigned kTileBytes = 5120;

// Chunks per entry-point call (one ObjectFifo element = kPerCall * 5120 B,
// double-buffered in L1), chunks per pool band (= ROWSPLIT * K/256), and the
// band's row split: 2 for the standard layout (std_perm: 64-row bands,
// half = c%2, kt = c/2), 4 for the expert stripes / down experts (128-row
// bands, quarter = c%4, kt = c/4 -- pools.rs stripe_transpose / down_perm).
#ifndef GEMV_PER_CALL
#define GEMV_PER_CALL 4
#endif
#ifndef GEMV_PER_BAND
#define GEMV_PER_BAND 16
#endif
#ifndef GEMV_ROWSPLIT
#define GEMV_ROWSPLIT 2
#endif
static constexpr unsigned kPerCall = GEMV_PER_CALL;
static constexpr unsigned kPerBand = GEMV_PER_BAND;
static constexpr unsigned kRowSplit = GEMV_ROWSPLIT;

// ---------------------------------------------------------------------------
// Phase 2 item 5 (2026-09-02): the integer matrix unit does the inner product.
//
// The bf16 form (vegah's: nibble -> bf16 through the accumulator path, then
// bf16 x scalar MACs) is bound by the vector ALU / accumulator slots, not by
// the MACs: every 32-lane nibble->bf16 conversion is 4 accumulator ops, and
// the compiler spilled ~30 accumulator quarters per 64 B. Measured: 1.4 GB/s
// per core, against a 4 GB/s DMA stream (the null probe streams qkv's 10.5 MB
// in 0.45 ms; the kernel took 0.9).
//
// This form uses `mmul<4, 8, 8, int16, uint8>`: B = one 64 B nibble block
// straight from the pool bytes ([8 k][8 row pairs], N fastest -- the pool's
// own order, no unpack), masked to the low nibbles (even rows, value nib) or
// the high nibbles (odd rows, value 16*nib; the 16 folds into d). A = the
// activation octet x[k..k+8] as int16, replicated over the 4 rows we do not
// use. The activation is block-quantised once per x per core (gemv_q4_prep):
// per 32-wide K block, s = 14 - floor(log2(max|x|)) and xi = round(x * 2^s),
// which is EXACT for every bf16 element within 2^7 of the block max (8
// significant bits land inside int16's 15); smaller elements round at
// 2^-15 of the block max. The int32 partial sums are exact (<= 2^24 for the
// even rows), and the epilogue per K block is the same as before:
//   y[r] += d[kb][r] * 2^-s * part[r]  (bf16 hi/lo split)  +  m[kb][r] * xs[kb]
// with xs (the block sum) precomputed as bf16 hi/lo in the table.
//
// Lane order: the mmul yields 8 even rows of a 16-row block per e-product and
// 8 odd rows per o-product, so the accumulator holds the chunk's 32 rows as
// [evens 0..30 | odds 1..31]; d/m are unzipped to match and y is zipped back
// into row order only on the band's last k-tile (`last`), staying permuted in
// between.
//
// Table layout (gemv_q4_tab_bytes(K) = 2.25 K):
//   int16 xi[K] | int32 s[K/32] | bf16 xs_hi[K/32] | bf16 xs_lo[K/32]
// ---------------------------------------------------------------------------

// One chunk against the k-tile `kt` of the table. `first` starts the band
// accumulator, `last` writes y back in row order (see the lane-order note).
// K (the table's) is a runtime argument: one body for every K in a design.
// GEMV_NULL: skip the arithmetic (DMA-only probe of a design's dataflow).
__attribute__((noinline)) inline void gemv_q4_tile(const uint8_t *__restrict tile,
                                                   const uint8_t *__restrict tab, unsigned K,
                                                   unsigned kt, bool first, bool last,
                                                   float *__restrict y) {
  event0();
#ifdef GEMV_NULL
  if (first) {
    aie::store_v(y, aie::zeros<float, kRows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NB = K / 32;

  const bfloat16 *__restrict dp = (const bfloat16 *)tile;
  const bfloat16 *__restrict mp = (const bfloat16 *)(tile + kDBytes);
  const uint8_t *__restrict nib0 = tile + kMetaBytes;          // rows 0..15
  const uint8_t *__restrict nib1 = tile + kMetaBytes + 2048;   // rows 16..31
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kTileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * K) + kt * kKBlocks;
  const bfloat16 *__restrict xsh = (const bfloat16 *)(tab + 2 * K + 4 * NB) + kt * kKBlocks;
  const bfloat16 *__restrict xsl = xsh + NB;

  aie::accum<accfloat, kRows> acc;
  if (first)
    acc = aie::zeros<accfloat, kRows>();
  else
    acc.from_vector(aie::load_v<kRows>(y));

  // [1 x16 | 1/16 x16]: the odd rows' high-nibble value is 16*nib
  const aie::vector<bfloat16, kRows> dsc =
      aie::concat(aie::broadcast<int16_t, 16>((int16_t)0x3F80),
                  aie::broadcast<int16_t, 16>((int16_t)0x3D80)).template cast_to<bfloat16>();

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<int32_t, 8> ve[2], vo[2];
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict src = (rb == 0 ? nib0 : nib1) + kb * 256;
      aie::mmul<4, 8, 8, int16_t, uint8_t> Ce, Co;
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const aie::vector<int16_t, 32> A =
            aie::load_v<8>(xi + kb * kKInBlock + oc * 8).template grow_replicate<32>();
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(src + oc * 64);
        const aie::vector<uint8_t, 64> e = aie::bit_and((uint8_t)0x0F, q);
        const aie::vector<uint8_t, 64> o = aie::bit_and((uint8_t)0xF0, q);
        if (oc == 0) {
          Ce.mul(A, e);
          Co.mul(A, o);
        } else {
          Ce.mac(A, e);
          Co.mac(A, o);
        }
      }
      ve[rb] = Ce.template to_vector<int32_t>().template extract<8>(0);
      vo[rb] = Co.template to_vector<int32_t>().template extract<8>(0);
    }
    // [rb0 evens | rb1 evens | rb0 odds | rb1 odds] = [evens 0..30 | odds 1..31]
    const aie::vector<int32_t, kRows> vi = aie::concat(ve[0], ve[1], vo[0], vo[1]);
    aie::accum<accfloat, kRows> part;
    part.from_vector(aie::to_float<float>(vi, sh[kb]));
    const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
    const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();

    const aie::vector<bfloat16, kRows> d32 = aie::load_v<kRows>(dp + kb * kRows);
    const aie::vector<bfloat16, kRows> m32 = aie::load_v<kRows>(mp + kb * kRows);
    auto [de, dod] = aie::interleave_unzip(d32, d32, 1);
    auto [me, mo] = aie::interleave_unzip(m32, m32, 1);
    const aie::vector<bfloat16, kRows> dperm = aie::concat(de.template extract<16>(0), dod.template extract<16>(0));
    const aie::vector<bfloat16, kRows> mperm = aie::concat(me.template extract<16>(0), mo.template extract<16>(0));
    const aie::vector<bfloat16, kRows> ds = aie::mul(dperm, dsc).template to_vector<bfloat16>();   // exact

    acc = aie::mac(acc, hi, ds);
    acc = aie::mac(acc, lo, ds);
    acc = aie::mac(acc, mperm, xsh[kb]);
    acc = aie::mac(acc, mperm, xsl[kb]);
  }

  const aie::vector<float, kRows> yv = acc.template to_vector<float>();
  if (last) {
    auto [r0, r1] = aie::interleave_zip(yv.template extract<16>(0), yv.template extract<16>(1), 1);
    aie::store_v(y, r0);
    aie::store_v(y + 16, r1);
  } else {
    aie::store_v(y, yv);
  }
  event1();
#endif
}

// One call = kPerCall consecutive POOL-ORDER chunks of one band, group `group`
// (chunks group*kPerCall .. +kPerCall-1 of the band). y is the band's 64-float
// accumulator; it is only complete after the last group of the band.
static constexpr unsigned kK = kTileK * kPerBand / kRowSplit;
static inline void gemv_q4_pool_group(const uint8_t *__restrict chunks,
                                      const uint8_t *__restrict tab,
                                      unsigned group, float *__restrict y) {
  constexpr unsigned kKt = kPerBand / kRowSplit;    // k-tiles per band
#ifdef GEMV_TABDUMP
  // debug probe (every band): y quarter N (64 B) = 64 B of the table at
  // {0, 2K, 2K + K/8, 2K + K/8 + K/16}[N] = xi[0:32], s[0:16], xs_hi[0:32], xs_lo[0:32]
  {
    const unsigned offs[4] = {0, 2 * kK, 2 * kK + kK / 8, 2 * kK + kK / 8 + kK / 16};
    aie::store_v((uint8_t *)y + 64 * group, aie::load_v<64>(tab + offs[group]));
  }
  return;
#endif
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kPerCall; ++i) {
    const unsigned c = group * kPerCall + i;   // index within the band
    const unsigned part = c % kRowSplit;       // which 32-row slice of the band
    const unsigned kt = c / kRowSplit;
    gemv_q4_tile(chunks + i * kTileBytes, tab, kK, kt, kt == 0, kt == kKt - 1, y + part * kRows);
  }
}

// The same with the band law (chunks per band, row split) as RUNTIME arguments:
// one entry point serves every shape of a design (16 KB program memory).
static inline void gemv_q4_pool_group_rt(const uint8_t *__restrict chunks,
                                         const uint8_t *__restrict tab,
                                         unsigned group, float *__restrict y,
                                         unsigned per_band, unsigned rs) {
  const unsigned K = kTileK * per_band / rs;
  const unsigned kt_last = per_band / rs - 1;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kPerCall; ++i) {
    const unsigned c = group * kPerCall + i;
    const unsigned part = c % rs;
    const unsigned kt = c / rs;
    gemv_q4_tile(chunks + i * kTileBytes, tab, K, kt, kt == 0, kt == kt_last, y + part * kRows);
  }
}

#define GEMV_Q4_ENTRY__(P, B, R, N)                                         \
  void gemv_q4_p##P##b##B##r##R##_k##N(const uint8_t *__restrict t,         \
                                       const uint8_t *__restrict tab,       \
                                       float *__restrict y) {               \
    gemv_q4_pool_group(t, tab, N, y);                                       \
  }

#define GEMV_Q4_ENTRY_(P, B, R, N) GEMV_Q4_ENTRY__(P, B, R, N)
#define GEMV_Q4_ENTRY(N) GEMV_Q4_ENTRY_(GEMV_PER_CALL, GEMV_PER_BAND, GEMV_ROWSPLIT, N)
// One entry point per (PER_CALL, PER_BAND, ROWSPLIT) with the group AND the
// band (y + band * band_rows) as RUNTIME arguments (from range_ loops):
// gemv_q4_p<P>b<B>r<R>_g. Keeps the core program compact (16 KB).
#define GEMV_Q4_GROUP_ENTRY__(P, B, R)                                       \
  void gemv_q4_p##P##b##B##r##R##_g(const uint8_t *__restrict t,            \
                                    const uint8_t *__restrict tab,          \
                                    float *__restrict y, int32_t group,     \
                                    int32_t band) {                         \
    gemv_q4_pool_group(t, tab, (unsigned)group, y + band * kBandRows * kRowSplit / 2); \
  }
#define GEMV_Q4_GROUP_ENTRY_(P, B, R) GEMV_Q4_GROUP_ENTRY__(P, B, R)
#define GEMV_Q4_GROUP_ENTRY() GEMV_Q4_GROUP_ENTRY_(GEMV_PER_CALL, GEMV_PER_BAND, GEMV_ROWSPLIT)
