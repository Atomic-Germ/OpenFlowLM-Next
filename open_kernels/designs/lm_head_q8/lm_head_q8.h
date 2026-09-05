#pragma once
//===- lm_head_q8.h ----------------------------------------*- C++ -*-===//
//
// W8A16 GEMV for the lm_head from phlegm's POOL-ORDER q8 chunks:
//   logits[248320] = W[248320, 2048] @ x[2048]
//
// Inner arithmetic ported from vegah/LLMNpuTest designs/lm_head (Apache-2.0,
// ../../LICENSE.LLMNpuTest); the chunk layout is FLM's q8 (q4nx.rs):
//
//   chunk = 32 output rows x 256 K, 8704 B:
//     scales[256] bf16 at [0   : 512]     index kb*32 + r          (r = 0..31)
//     codes [8192] int8 at [512 : 8704]   index (r/16)*4096 + k*16 + (r%16)
//   value = code * scale
//
// so `load_v<32>(s + kb*32)` is the 32 output rows in order, and two 16-lane
// code loads 4096 apart concatenate to those rows at one k: one 32-lane MAC
// per activation element, no gather.
//
// POOL ORDER (pools.rs build_lmhead_pool, "128-row supertile transpose"):
//   pool chunk k <- file chunk (4*(k/32) + (k%4))*8 + ((k%32)/4), i.e.
//   band = 32 consecutive chunks = 128 output rows x 2048 K;
//   chunk c inside its band: row quarter = c % 4, k-tile = c / 4.
// One band is consumed with a 128-float accumulator; the runtime `group`
// argument walks (quarter, kt) off the chunk index.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gemv_tab.h"

static constexpr unsigned kRows = 32;        // output rows per chunk
static constexpr unsigned kKBlocks = 8;      // 32-wide K blocks per chunk
static constexpr unsigned kKInBlock = 32;
static constexpr unsigned kTileK = 256;      // K per chunk
static constexpr unsigned kScaleBytes = 512;
static constexpr unsigned kRowBlockStride = 4096;  // in codes
static constexpr unsigned kTileBytes = 8704;
static constexpr unsigned kRowSplit = 4;     // 32-row quarters per 128-row band

#ifndef LMHEAD_PER_CALL
#define LMHEAD_PER_CALL 2
#endif
static constexpr unsigned kPerCall = LMHEAD_PER_CALL;

// Phase 2 item 5 (2026-09-02): the inner product on the integer matrix unit,
// like gemv_q4.h. vegah's form (int8 -> bf16 through the accumulator, bf16 x
// scalar MACs) ran at ~3 GB/s per core against a ~4 GB/s stream; this one is
// DMA-bound. Per 128 B (8 k x 16 rows, 16 B per k): an unzip at step 8 gives
// [8 k][rows 0..7] and [8 k][rows 8..15] -- two mmul<4,8,8,int16,int8> B
// operands -- against the x octet from the int16 table (gemv_tab.h). The
// four 8-row results assemble into the chunk's 32 rows IN ORDER (no
// permutation, unlike the q4 kernel), and per K block:
//   y[r] += scale[kb][r] * 2^-s[kb] * part[r]   (bf16 hi/lo split)
// Runtime kt/first, noinline + inline (COMDAT): one body in program memory.
static constexpr unsigned kK = 2048;
__attribute__((noinline)) inline void gemv_q8_tile(const uint8_t *__restrict tile,
                                                   const uint8_t *__restrict tab,
                                                   unsigned kt, bool first,
                                                   float *__restrict y) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict s = (const bfloat16 *)tile;
  const uint8_t *__restrict c0 = tile + kScaleBytes;              // rows 0..15: byte k*16 + r
  const uint8_t *__restrict c1 = c0 + kRowBlockStride;            // rows 16..31
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kTileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * kK) + kt * kKBlocks;

  aie::accum<accfloat, kRows> acc;
  if (first)
    acc = aie::zeros<accfloat, kRows>();
  else
    acc.from_vector(aie::load_v<kRows>(y));

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<int32_t, 8> v[4];
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict src = (rb == 0 ? c0 : c1) + kb * kKInBlock * 16;
      aie::mmul<4, 8, 8, int16_t, int8_t> Clo, Chi;
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const aie::vector<int16_t, 32> A =
            aie::load_v<8>(xi + kb * kKInBlock + oc * 8).template grow_replicate<32>();
        const aie::vector<int8_t, 64> q0 = aie::load_v<64>((const int8_t *)src + oc * 128);
        const aie::vector<int8_t, 64> q1 = aie::load_v<64>((const int8_t *)src + oc * 128 + 64);
        auto [lo, hi] = aie::interleave_unzip(q0, q1, 8);   // rows 0..7 / 8..15 at k 0..7
        if (oc == 0) {
          Clo.mul(A, lo);
          Chi.mul(A, hi);
        } else {
          Clo.mac(A, lo);
          Chi.mac(A, hi);
        }
      }
      v[2 * rb] = Clo.template to_vector<int32_t>().template extract<8>(0);
      v[2 * rb + 1] = Chi.template to_vector<int32_t>().template extract<8>(0);
    }
    const aie::vector<int32_t, kRows> vi = aie::concat(v[0], v[1], v[2], v[3]);   // rows 0..31
    aie::accum<accfloat, kRows> part;
    part.from_vector(aie::to_float<float>(vi, sh[kb]));
    const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
    const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
    const aie::vector<bfloat16, kRows> sv = aie::load_v<kRows>(s + kb * kRows);
    acc = aie::mac(acc, hi, sv);
    acc = aie::mac(acc, lo, sv);
  }

  aie::store_v(y, acc.template to_vector<float>());
  event1();
}
