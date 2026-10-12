//===- gemv_t2n.cc ----------------------------------------*- C++ -*-===//
//
// Ternary (2-bit) GEMV over eight rows at once (the DFlash 2 drafter's block, OPEN-DRAFT-DFLASH2):
//   y[t, r] = sum_k W[r, k] x[t, k],   W[r, k] = s[r, k/128] * (code - 1)
// on gemv_t2.h's chunks (OPEN-QUANT-T2: 32 rows x 256 K, bf16 scales then 2-bit codes, 2176 B).
//
// The mmul<4, 8, 8, int16, uint8>'s four A rows are four tokens (a quad) instead of one token
// repeated, so one masked code block feeds four rows' products. Field j (rows 8j..8j+7) carries
// code * 4^j, so C_j >> 2j is sum(x * code) exactly and D_j = (C_j >> 2j) - sum(x) is the
// product with (code - 1): the min term is one integer subtract instead of a float epilogue.
//
// The table is the host's (dflash2.cpp), one record per (k-tile, 128-group, quad):
//   int16 x[16 octets][4 tok][8]   at [0, 1024)     the quad's values at one shift
//   int32 sum[4 tok][8]            at [1024, 1152)  each token's group sum, repeated over p
//   int32 shift                    at 1152          x = round(xr * 2^shift)
// y, per 64-row band: f32 [half 2][quad 2][field 4][tok 4][p 8] (row 32 half + 8 field + p,
// token 4 quad + tok), accumulated over the item's k-tiles.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kChunk = 2176;
static constexpr unsigned kRec = 1216;
static constexpr unsigned kHdr = 64;

__attribute__((noinline)) static void t2n_tile(const uint8_t *__restrict chunk, const uint8_t *__restrict rec0,
                                               const uint8_t *__restrict rec1, bool first, float *__restrict yq) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::accum<accfloat, 32> acc0, acc1, acc2, acc3;
  if (first) {
    acc0 = aie::zeros<accfloat, 32>();
    acc1 = aie::zeros<accfloat, 32>();
    acc2 = aie::zeros<accfloat, 32>();
    acc3 = aie::zeros<accfloat, 32>();
  } else {
    acc0.from_vector(aie::load_v<32>(yq));
    acc1.from_vector(aie::load_v<32>(yq + 32));
    acc2.from_vector(aie::load_v<32>(yq + 64));
    acc3.from_vector(aie::load_v<32>(yq + 96));
  }
#pragma clang loop unroll(disable)
  for (unsigned g = 0; g < 2; ++g) {
    const uint8_t *__restrict rec = g ? rec1 : rec0;
    const int16_t *__restrict xi = (const int16_t *)rec;
    const aie::vector<int32_t, 32> os = aie::load_v<32>((const int32_t *)(rec + 1024));
    const int sh = *(const int32_t *)(rec + 1152);
    const int16_t *__restrict sc = (const int16_t *)chunk + g * 32;
    const uint8_t *__restrict code = chunk + 128 + g * 1024;
    aie::vector<int32_t, 32> cv[4];
#pragma clang loop unroll(full)
    for (unsigned jp = 0; jp < 2; ++jp) {
      const uint8_t me = jp == 0 ? (uint8_t)0x03 : (uint8_t)0x30;
      const uint8_t mo = jp == 0 ? (uint8_t)0x0C : (uint8_t)0xC0;
      aie::mmul<4, 8, 8, int16_t, uint8_t> Ce, Co;
#pragma clang loop unroll(full)
      for (unsigned o = 0; o < 16; ++o) {
        const aie::vector<int16_t, 32> A = aie::load_v<32>(xi + o * 32);
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(code + o * 64);
        if (o == 0) {
          Ce.mul(A, aie::bit_and(me, q));
          Co.mul(A, aie::bit_and(mo, q));
        } else {
          Ce.mac(A, aie::bit_and(me, q));
          Co.mac(A, aie::bit_and(mo, q));
        }
      }
      cv[2 * jp] = Ce.template to_vector<int32_t>(4 * jp);          // fields 0 / 2: >> 0 / >> 4
      cv[2 * jp + 1] = Co.template to_vector<int32_t>(4 * jp + 2);  // fields 1 / 3: >> 2 / >> 6
    }
#pragma clang loop unroll(full)
    for (unsigned j = 0; j < 4; ++j) {
      aie::accum<accfloat, 32> part;
      part.from_vector(aie::to_float<float>(aie::sub(cv[j], os), sh));
      const aie::vector<bfloat16, 32> hi = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, 32> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      const aie::vector<bfloat16, 32> sw =
          aie::load_v<8>(sc + 8 * j).template grow_replicate<32>().template cast_to<bfloat16>();
      aie::accum<accfloat, 32> &a = j == 0 ? acc0 : j == 1 ? acc1 : j == 2 ? acc2 : acc3;
      a = aie::mac(a, hi, sw);
      a = aie::mac(a, lo, sw);
    }
  }
  aie::store_v(yq, acc0.template to_vector<float>());
  aie::store_v(yq + 32, acc1.template to_vector<float>());
  aie::store_v(yq + 64, acc2.template to_vector<float>());
  aie::store_v(yq + 96, acc3.template to_vector<float>());
}

extern "C" {
// One w element = k-tile `kt` (of the item) of one band: chunk 0 is rows 0..31, chunk 1 rows 32..63.
void gemv_t2n_kt(const uint8_t *__restrict w, const int32_t *__restrict tab, float *__restrict y, int32_t kt) {
  event0();
#ifdef GEMV_NULL
  if (kt == 0)
    for (unsigned i = 0; i < 512; i += 32)
      aie::store_v(y + i, aie::zeros<float, 32>());
#else
  const uint8_t *__restrict tb = (const uint8_t *)tab + kHdr + (unsigned)kt * 4u * kRec;
#pragma clang loop unroll(disable)
  for (unsigned hq = 0; hq < 4; ++hq) {
    const unsigned half = hq >> 1, quad = hq & 1;
    t2n_tile(w + half * kChunk, tb + quad * kRec, tb + (2 + quad) * kRec, kt == 0, y + hq * 128);
  }
#endif
  event1();
}
}
