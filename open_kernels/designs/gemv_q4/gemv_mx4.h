#pragma once
//===- gemv_mx4.h ------------------------------------------*- C++ -*-===//
//
// MXFP4 GEMV on the main cores, against POOL-ORDER mx4 HALF-TILES: GPT-OSS's experts as the
// GGUF stores them (OPEN-QUANT-MXFP4), y[N] += W[N, K] @ x[K], no re-quantisation.
//
// Sibling of gemv_q8.h: the same 16-row x 256-K half-tile, band law (rs = 4), activation table,
// mmul and epilogue, so every product is the q8 half-tile's -- an MXFP4 code is twice an e2m1
// level, an integer in [-12, 12], which is exactly the int8 code the q8 tile held, and the block
// scale 2^(e-128) is the same exact bf16. Only the bytes change: 2368 a half-tile instead of
// 5120, the codes as the GGUF's own nibbles, decoded on the core.
//
//   half-tile, 2368 B: four 64-K groups g of 576 B, then the bias
//     [576 g       : +64 )  bf16 scale[kb' * 16 + r]  for blocks 2g + kb' (kb' = 0, 1), rows r = 0..15
//     [576 g + 64  : +256)  block 2g's nibbles: four 8-K octets of 64 B
//     [576 g + 320 : +256)  block 2g + 1's
//     [2304 : 2368)         f32 bias[16] on k-tile 0 (GEMV_Q8_PAD_BIAS's law); zero elsewhere
//   An octet is 32 B for rows 0..7 then 32 B for rows 8..15. Nibble j of a half (byte j / 2, the
//   low nibble first) is k = j / 8, row = j % 8: unpack's order, so each half widens straight into
//   one mmul<4, 8, 8, int16, int8> B operand ([8 k][8 rows], rows fastest). Every 64-B code run
//   stays 64-byte aligned, and a 64-K group is one contiguous 576-B run (a GEMM's k-chunk).
//
// The nibble and its decode: mx4_codes.h (shared with gemm_bfp16's MXFP4 B operand).
//
// TRAPS (gemv_q8.h's, all of them): conv_even rounding; no fp32 vector multiply (hi/lo bf16);
// entry points one per TU; the body noinline+inline (COMDAT) for the 16 KB program memory.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gemv_tab.h"   // the activation's int16 block table (shared with gemv_q8.h / gemv_q4.h)
#include "mx4_codes.h"

static constexpr unsigned kMx4Rows = 16;
static constexpr unsigned kMx4KBlocks = 8;
static constexpr unsigned kMx4KInBlock = 32;
static constexpr unsigned kMx4TileK = 256;
static constexpr unsigned kMx4GroupBytes = 576;      // 64 B of scales + two blocks of 256 B of nibbles
static constexpr unsigned kMx4TileBytes = 2368;
static constexpr unsigned kMx4PadBias = 2304;
static constexpr unsigned kMx4RowSplit = 4;

#ifndef GEMV_Q8_PAD_BIAS
#define GEMV_Q8_PAD_BIAS 0
#endif
#ifndef GEMV_PER_CALL
#define GEMV_PER_CALL 4
#endif
static constexpr unsigned kMx4PerCall = GEMV_PER_CALL;

// One half-tile against the k-tile `kt` of the table, into 16 floats of the band's accumulator;
// `first` starts it (from the pad's bias under GEMV_Q8_PAD_BIAS). gemv_q8_half_tile's walk.
__attribute__((noinline)) inline void gemv_mx4_half_tile(const uint8_t *__restrict tile,
                                                         const uint8_t *__restrict tab, unsigned K,
                                                         unsigned kt, bool first,
                                                         float *__restrict y) {
  event0();
#ifdef GEMV_NULL
  if (first) {
    aie::store_v(y, aie::zeros<float, kMx4Rows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);

  const int16_t *__restrict xi = (const int16_t *)tab + kt * kMx4TileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * K) + kt * kMx4KBlocks;

  aie::accum<accfloat, kMx4Rows> acc;
  if (first)
#if GEMV_Q8_PAD_BIAS
    acc.from_vector(aie::load_v<kMx4Rows>((const float *)(tile + kMx4PadBias)));
#else
    acc = aie::zeros<accfloat, kMx4Rows>();
#endif
  else
    acc.from_vector(aie::load_v<kMx4Rows>(y));

  // Two blocks a turn let one block's epilogue (to_float, hi / lo, two macs: ~28 bundles of
  // latency alone) run under the next block's decode: Peano 2026-10, 108 -> 89.5 bundles a block
  // (four a turn: 116, the decodes' registers spill).
#pragma clang loop unroll_count(2)
  for (unsigned kb = 0; kb < kMx4KBlocks; ++kb) {
    const uint8_t *__restrict grp = tile + (kb / 2) * kMx4GroupBytes;
    const uint8_t *__restrict src = grp + 64 + (kb % 2) * 256;
    aie::mmul<4, 8, 8, int16_t, int8_t> Clo, Chi;
#pragma clang loop unroll(full)
    for (unsigned oc = 0; oc < 4; ++oc) {
      const aie::vector<int16_t, 32> A =
          aie::load_v<8>(xi + kb * kMx4KInBlock + oc * 8).template grow_replicate<32>();
      const aie::vector<int8_t, 64> lo = mx4_codes(src + oc * 64);        // rows 0..7 at k 0..7
      const aie::vector<int8_t, 64> hi = mx4_codes(src + oc * 64 + 32);   // rows 8..15
      if (oc == 0) {
        Clo.mul(A, lo);
        Chi.mul(A, hi);
      } else {
        Clo.mac(A, lo);
        Chi.mac(A, hi);
      }
    }
    const aie::vector<int32_t, kMx4Rows> vi =
        aie::concat(Clo.template to_vector<int32_t>().template extract<8>(0),
                    Chi.template to_vector<int32_t>().template extract<8>(0));   // rows 0..15
    aie::accum<accfloat, kMx4Rows> part;
    part.from_vector(aie::to_float<float>(vi, sh[kb]));
    const aie::vector<bfloat16, kMx4Rows> hi = part.template to_vector<bfloat16>();
    const aie::vector<bfloat16, kMx4Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
    const aie::vector<bfloat16, kMx4Rows> sv = aie::load_v<kMx4Rows>((const bfloat16 *)grp + (kb % 2) * kMx4Rows);
    acc = aie::mac(acc, hi, sv);
    acc = aie::mac(acc, lo, sv);
  }

  aie::store_v(y, acc.template to_vector<float>());
  event1();
#endif
}

// One call = kMx4PerCall consecutive POOL-ORDER half-tiles of one band, group `group`, into the
// band's 64-float accumulator (gemv_q8_pool_group_rt's walk; per_band = K/64, rs = 4).
static inline void gemv_mx4_pool_group_rt(const uint8_t *__restrict chunks,
                                          const uint8_t *__restrict tab,
                                          unsigned group, float *__restrict y,
                                          unsigned per_band) {
  const unsigned K = kMx4TileK * per_band / kMx4RowSplit;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kMx4PerCall; ++i) {
    const unsigned c = group * kMx4PerCall + i;
    const unsigned part = c % kMx4RowSplit;
    const unsigned kt = c / kMx4RowSplit;
    gemv_mx4_half_tile(chunks + i * kMx4TileBytes, tab, K, kt, kt == 0, y + part * kMx4Rows);
  }
}

// The same walk, starting afresh on the band's first k-tile only when `fresh`, else adding onto
// y (gemv_q8_pool_group_acc: GPT-OSS's down sums its four experts in one accumulator).
static inline void gemv_mx4_pool_group_acc(const uint8_t *__restrict chunks,
                                           const uint8_t *__restrict tab,
                                           unsigned group, float *__restrict y,
                                           unsigned per_band, bool fresh) {
  const unsigned K = kMx4TileK * per_band / kMx4RowSplit;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kMx4PerCall; ++i) {
    const unsigned c = group * kMx4PerCall + i;
    const unsigned part = c % kMx4RowSplit;
    const unsigned kt = c / kMx4RowSplit;
    gemv_mx4_half_tile(chunks + i * kMx4TileBytes, tab, K, kt, fresh && kt == 0, y + part * kMx4Rows);
  }
}
