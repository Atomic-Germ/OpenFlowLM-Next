#pragma once
//===- gemv_q8.h -------------------------------------------*- C++ -*-===//
//
// W8A16 GEMV on the main cores, against POOL-ORDER q8 HALF-TILES:
//   y[N] += W[N, K] @ x[K],  W as FLM stores it at q8, no re-quantization.
//
// Sibling of gemv_q4.h (same activation table, same band, same y element); the
// arithmetic is designs/lm_head_q8/lm_head_q8.h's `gemv_q8_tile` cut down to ONE
// 16-row block and given a runtime K. See OPEN-QUANT-Q8 in specs/open-engine/spec.md
// and specs/open-engine/plans/q8-gemv.md.
//
// WHY HALF-TILES. A container q8 chunk is 8704 B (32 rows x 256 K) and the main
// cores' w-fifo element is PER_CALL x 5120 B, which 8704 fits neither way. The host
// splits each chunk into two 16-row half-tiles that each fill one 5120 B element
// (recipes/pack.py q8_half_tiles):
//
//   half-tile, 5120 B:  scales[128] bf16 at [0    : 256]    index kb*16 + r  (r = 0..15)
//                       codes [4096] int8 at [256  : 4352]  index k*16 + r
//                       zero pad          at [4352 : 5120]
//   value = code * scale
//
// The container's row-block stride is exactly 4096 codes, so half h's codes are the
// verbatim slice [512 + h*4096, 512 + (h+1)*4096) of the chunk and only the scales are
// gathered: a byte permutation on the host, no arithmetic. The 768 B of pad is 15 % of
// the stream on q8 tensors only; a 4352-byte element type would change every design's
// fifo element and is not worth it.
//
// POOL ORDER (recipes/pack.py q8_perm, K = in_dim):
//   band = K/64 consecutive half-tiles = 64 output rows x K.
//   half-tile c inside its band: row quarter = c % 4 (rows 16*(c%4) ..),
//                                k-tile      = c / 4 (cols 256*(c/4) ..).
// So a band is consumed as `per_band` half-tiles with the SAME 64-float accumulator the
// q4 GEMV uses, and the entry point walks (part, kt) off the index -- the q4 law with
// rs = 4 and 16-row parts instead of rs = 2 and 32-row halves. Nothing downstream of the
// y element changes.
//
// Rows are IN ORDER inside a half-tile (16 consecutive codes at one k are 16 different
// rows), so unlike the q4 kernel there is no lane permutation and no `last` argument.
//
// TRAPS (the ones gemv_q4.h lists, all of them still apply): default rounding is floor
// -> set conv_even; AIE2P has NO fp32 vector multiply (it returns zero silently) -> the
// accumulator is split into two bf16 halves; no scalar float; entry points one per TU;
// the body is noinline+inline (COMDAT) so 16 KB of program memory holds one copy.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gemv_tab.h"   // the activation's int16 block table (shared with gemv_q4.h)

static constexpr unsigned kQ8Rows = 16;        // output rows per half-tile
static constexpr unsigned kQ8KBlocks = 8;      // 32-wide K blocks per half-tile
static constexpr unsigned kQ8KInBlock = 32;
static constexpr unsigned kQ8TileK = 256;      // K per half-tile
static constexpr unsigned kQ8ScaleBytes = 256;
static constexpr unsigned kQ8TileBytes = 5120;
static constexpr unsigned kQ8RowSplit = 4;     // 16-row parts per 64-row band

// Half-tiles per entry-point call (one ObjectFifo element = kQ8PerCall * 5120 B), the
// same knob the q4 GEMV takes so a design's w element is one size for both formats.
#ifndef GEMV_PER_CALL
#define GEMV_PER_CALL 2
#endif
static constexpr unsigned kQ8PerCall = GEMV_PER_CALL;

// One half-tile against the k-tile `kt` of the table, into 16 floats of the band's
// accumulator. `first` starts the accumulation; there is no `last` (rows are in order,
// so y is never held permuted). K (the table's) is a runtime argument: one body for
// every K in a design.
//
// Per 128 B of codes (8 k x 16 rows, 16 B per k): an unzip at step 8 gives [8 k][rows
// 0..7] and [8 k][rows 8..15] -- two mmul<4,8,8,int16,int8> B operands -- against the x
// octet from the int16 table. The two 8-row results concatenate to the tile's 16 rows in
// order, and per K block:
//   y[r] += scale[kb][r] * 2^-s[kb] * part[r]   (bf16 hi/lo split)
// There is no min term: q8 is symmetric (value = code * scale), so the `m * sum(x)`
// half of the q4_1 epilogue simply does not exist.
__attribute__((noinline)) inline void gemv_q8_half_tile(const uint8_t *__restrict tile,
                                                        const uint8_t *__restrict tab, unsigned K,
                                                        unsigned kt, bool first,
                                                        float *__restrict y) {
  event0();
#ifdef GEMV_NULL
  if (first) {
    aie::store_v(y, aie::zeros<float, kQ8Rows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict s = (const bfloat16 *)tile;             // scale[kb*16 + r]
  const uint8_t *__restrict c0 = tile + kQ8ScaleBytes;               // code[k*16 + r]
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kQ8TileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * K) + kt * kQ8KBlocks;

  aie::accum<accfloat, kQ8Rows> acc;
  if (first)
    acc = aie::zeros<accfloat, kQ8Rows>();
  else
    acc.from_vector(aie::load_v<kQ8Rows>(y));

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kQ8KBlocks; ++kb) {
    const uint8_t *__restrict src = c0 + kb * kQ8KInBlock * kQ8Rows;
    aie::mmul<4, 8, 8, int16_t, int8_t> Clo, Chi;
#pragma clang loop unroll(full)
    for (unsigned oc = 0; oc < 4; ++oc) {
      const aie::vector<int16_t, 32> A =
          aie::load_v<8>(xi + kb * kQ8KInBlock + oc * 8).template grow_replicate<32>();
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
    const aie::vector<int32_t, kQ8Rows> vi =
        aie::concat(Clo.template to_vector<int32_t>().template extract<8>(0),
                    Chi.template to_vector<int32_t>().template extract<8>(0));   // rows 0..15
    aie::accum<accfloat, kQ8Rows> part;
    part.from_vector(aie::to_float<float>(vi, sh[kb]));
    const aie::vector<bfloat16, kQ8Rows> hi = part.template to_vector<bfloat16>();
    const aie::vector<bfloat16, kQ8Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
    const aie::vector<bfloat16, kQ8Rows> sv = aie::load_v<kQ8Rows>(s + kb * kQ8Rows);
    acc = aie::mac(acc, hi, sv);
    acc = aie::mac(acc, lo, sv);
  }

  aie::store_v(y, acc.template to_vector<float>());
  event1();
#endif
}

// One call = kQ8PerCall consecutive POOL-ORDER half-tiles of one band, group `group`.
// `y` is the band's 64-float accumulator; it is complete after the band's last group.
// The band law is RUNTIME (per_band half-tiles, row split rs = 4), exactly as
// gemv_q4_pool_group_rt takes it, so one entry point serves every shape of a design.
static inline void gemv_q8_pool_group_rt(const uint8_t *__restrict chunks,
                                         const uint8_t *__restrict tab,
                                         unsigned group, float *__restrict y,
                                         unsigned per_band, unsigned rs) {
  const unsigned K = kQ8TileK * per_band / rs;      // per_band = K/64, rs = 4
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kQ8PerCall; ++i) {
    const unsigned c = group * kQ8PerCall + i;      // index within the band
    const unsigned part = c % rs;                   // which 16-row slice of the band
    const unsigned kt = c / rs;
    gemv_q8_half_tile(chunks + i * kQ8TileBytes, tab, K, kt, kt == 0, y + part * kQ8Rows);
  }
}
