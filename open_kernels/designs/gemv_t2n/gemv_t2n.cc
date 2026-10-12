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

// One 32-row chunk against both token quads: the masked code block feeds four products, so each
// code load and mask serves eight tokens. yh = this half's [quad 2][field 4][tok 4][p 8] floats, read
// and written in place so that only the four mmul accumulators stay live.
__attribute__((noinline)) static void t2n_half(const uint8_t *__restrict chunk, const uint8_t *__restrict tb,
                                               float *__restrict yh) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned g = 0; g < 2; ++g) {
    const uint8_t *__restrict rec0 = tb + (g * 2) * kRec, *__restrict rec1 = tb + (g * 2 + 1) * kRec;
    const int16_t *__restrict x0 = (const int16_t *)rec0, *__restrict x1 = (const int16_t *)rec1;
    const int16_t *__restrict sc = (const int16_t *)chunk + g * 32;
    const uint8_t *__restrict code = chunk + 128 + g * 1024;
    const aie::vector<int32_t, 32> os0 = aie::load_v<32>((const int32_t *)(rec0 + 1024));
    const aie::vector<int32_t, 32> os1 = aie::load_v<32>((const int32_t *)(rec1 + 1024));
    const int sh0 = *(const int32_t *)(rec0 + 1152), sh1 = *(const int32_t *)(rec1 + 1152);
#pragma clang loop unroll(full)
    for (unsigned jp = 0; jp < 2; ++jp) {
      const uint8_t me = jp == 0 ? (uint8_t)0x03 : (uint8_t)0x30;
      const uint8_t mo = jp == 0 ? (uint8_t)0x0C : (uint8_t)0xC0;
      aie::mmul<4, 8, 8, int16_t, uint8_t> Ce0, Co0, Ce1, Co1;
#pragma clang loop unroll(full)
      for (unsigned o = 0; o < 16; ++o) {
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(code + o * 64);
        const aie::vector<uint8_t, 64> e = aie::bit_and(me, q), od = aie::bit_and(mo, q);
        const aie::vector<int16_t, 32> A0 = aie::load_v<32>(x0 + o * 32), A1 = aie::load_v<32>(x1 + o * 32);
        if (o == 0) {
          Ce0.mul(A0, e);
          Co0.mul(A0, od);
          Ce1.mul(A1, e);
          Co1.mul(A1, od);
        } else {
          Ce0.mac(A0, e);
          Co0.mac(A0, od);
          Ce1.mac(A1, e);
          Co1.mac(A1, od);
        }
      }
      // fields 2jp / 2jp+1 sit at bits 4jp / 4jp+2
      const aie::vector<int32_t, 32> c[4] = {Ce0.template to_vector<int32_t>(4 * jp), Co0.template to_vector<int32_t>(4 * jp + 2),
                                             Ce1.template to_vector<int32_t>(4 * jp), Co1.template to_vector<int32_t>(4 * jp + 2)};
#pragma clang loop unroll(full)
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned qd = i >> 1, j = 2 * jp + (i & 1);
        float *yp = yh + qd * 128 + j * 32;
        aie::accum<accfloat, 32> part, acc;
        part.from_vector(aie::to_float<float>(aie::sub(c[i], qd ? os1 : os0), qd ? sh1 : sh0));
        const aie::vector<bfloat16, 32> hi = part.template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> lo = aie::sub(part, hi).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> sw =
            aie::load_v<8>(sc + 8 * j).template grow_replicate<32>().template cast_to<bfloat16>();
        acc.from_vector(aie::load_v<32>(yp));
        acc = aie::mac(acc, hi, sw);
        acc = aie::mac(acc, lo, sw);
        aie::store_v(yp, acc.template to_vector<float>());
      }
    }
  }
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
  if (kt == 0)
    for (unsigned i = 0; i < 512; i += 32)
      aie::store_v(y + i, aie::zeros<float, 32>());
  const uint8_t *__restrict tb = (const uint8_t *)tab + kHdr + (unsigned)kt * 4u * kRec;
  for (unsigned half = 0; half < 2; ++half)
    t2n_half(w + half * kChunk, tb, y + half * 256);
#endif
  event1();
}
}
