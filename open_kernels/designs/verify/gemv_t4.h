#pragma once
// Up to four tokens per weight pass: the mmul's four A rows are four tokens, read as one 64 B load from an
// interleaved table, so each token's int32 row is exactly the 1-wide product and its epilogue is the 1-wide
// epilogue: per token bit-identical to gemv_t2_tile (OPEN-DECODE-SPECULATIVE-VERIFY).
//
//   table: per K octet o, [t0 | t1 | t2 | t3] int16[8] each (64 B at 64 o); then per token j the 1-wide
//   scale block [int32 s[NG] | bf16 xs_hi[NG] | bf16 xs_lo[NG]] at 8 K + 8 NG j.

#include "gemv_t2.h"

#ifndef GEMV_T4_GUNROLL
#define GEMV_T4_GUNROLL disable
#endif
static constexpr unsigned kT4Tok = 4;
alignas(64) inline const float gemv_t4_zero[32] = {};
static constexpr unsigned gemv_t4_tab_bytes(unsigned K) { return kT4Tok * 2 * K + kT4Tok * (K / 16); }

// token J's row of the four products, with gemv_t2_tile's epilogue (scale block sb, group gi)
template <unsigned J>
__attribute__((always_inline)) inline void gemv_t4_epi(const aie::vector<int32_t, 32> &ce0, const aie::vector<int32_t, 32> &co0,
                                                      const aie::vector<int32_t, 32> &ce1, const aie::vector<int32_t, 32> &co1,
                                                      const uint8_t *__restrict scales, unsigned NG, unsigned gi, bool z,
                                                      const aie::vector<bfloat16, kT2Rows> &s32,
                                                      const aie::vector<bfloat16, kT2Rows> &ds, float *__restrict y) {
  const uint8_t *__restrict sb = scales + 8 * NG * J;
  const int32_t sh = ((const int32_t *)sb)[gi];
  const bfloat16 xsh = ((const bfloat16 *)(sb + 4 * NG))[gi];
  const bfloat16 xsl = ((const bfloat16 *)(sb + 6 * NG))[gi];
  aie::accum<accfloat, kT2Rows> acc;
  acc.from_vector(aie::load_v<kT2Rows>(z ? gemv_t4_zero : y));   // a branch per token would serialise the four
  const aie::vector<int32_t, kT2Rows> vi = aie::concat(ce0.template extract<8>(J), co0.template extract<8>(J),
                                                       ce1.template extract<8>(J), co1.template extract<8>(J));
  aie::accum<accfloat, kT2Rows> part;
  part.from_vector(aie::to_float<float>(vi, sh));
  const aie::vector<bfloat16, kT2Rows> hi = part.template to_vector<bfloat16>();
  const aie::vector<bfloat16, kT2Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
  acc = aie::mac(acc, hi, ds);
  acc = aie::mac(acc, lo, ds);
  acc = aie::msc(acc, s32, xsh);
  acc = aie::msc(acc, s32, xsl);
  aie::store_v(y, acc.template to_vector<float>());
}

__attribute__((noinline)) inline void gemv_t4_tile(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                                   unsigned K, unsigned kt, bool first, float *__restrict y0,
                                                   float *__restrict y1, float *__restrict y2, float *__restrict y3,
                                                   unsigned nt) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NG = K / 128;
  const bfloat16 *__restrict sp = (const bfloat16 *)tile;
  const uint8_t *__restrict code = tile + kT2ScaleBytes;
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kT2TileK * kT4Tok;
  const aie::vector<bfloat16, kT2Rows> fsc = gemv_t2_field_scale();
#pragma clang loop unroll(GEMV_T4_GUNROLL)
  for (unsigned g = 0; g < 2; ++g) {
    const aie::vector<bfloat16, kT2Rows> s32 = aie::load_v<kT2Rows>(sp + g * kT2Rows);
    const aie::vector<bfloat16, kT2Rows> ds = aie::mul(s32, fsc).template to_vector<bfloat16>();   // exact
    const int16_t *__restrict xg = xi + g * 128 * kT4Tok;
    const uint8_t *__restrict cg = code + g * 1024;
    aie::mmul<4, 8, 8, int16_t, uint8_t> Ce0, Co0, Ce1, Co1;
#pragma clang loop unroll(full)
    for (unsigned o = 0; o < 16; ++o) {
      const aie::vector<int16_t, 32> A = aie::load_v<32>(xg + o * 32);
      const aie::vector<uint8_t, 64> q = aie::load_v<64>(cg + o * 64);
      const aie::vector<uint8_t, 64> e0 = aie::bit_and((uint8_t)0x03, q), o0 = aie::bit_and((uint8_t)0x0C, q);
      const aie::vector<uint8_t, 64> e1 = aie::bit_and((uint8_t)0x30, q), o1 = aie::bit_and((uint8_t)0xC0, q);
      if (o == 0) {
        Ce0.mul(A, e0);
        Co0.mul(A, o0);
        Ce1.mul(A, e1);
        Co1.mul(A, o1);
      } else {
        Ce0.mac(A, e0);
        Co0.mac(A, o0);
        Ce1.mac(A, e1);
        Co1.mac(A, o1);
      }
    }
    const aie::vector<int32_t, 32> ce0 = Ce0.template to_vector<int32_t>(), co0 = Co0.template to_vector<int32_t>();
    const aie::vector<int32_t, 32> ce1 = Ce1.template to_vector<int32_t>(), co1 = Co1.template to_vector<int32_t>();
    const unsigned gi = kt * 2 + g;
    const bool z = first && g == 0;
    gemv_t4_epi<0>(ce0, co0, ce1, co1, tab + kT4Tok * 2 * K, NG, gi, z, s32, ds, y0);
    gemv_t4_epi<1>(ce0, co0, ce1, co1, tab + kT4Tok * 2 * K, NG, gi, z, s32, ds, y1);
    gemv_t4_epi<2>(ce0, co0, ce1, co1, tab + kT4Tok * 2 * K, NG, gi, z, s32, ds, y2);
    gemv_t4_epi<3>(ce0, co0, ce1, co1, tab + kT4Tok * 2 * K, NG, gi, z, s32, ds, y3);
  }
}

// gemv_t2_chunk_rt's band law: chunk c of a band is k-tile c / rs, rows (c % rs) * 32 of the band
__attribute__((noinline)) inline void gemv_t4_chunk_rt(const uint8_t *__restrict elem, const uint8_t *__restrict tab,
                                                       unsigned group, unsigned per_band, unsigned kt0,
                                                       float *__restrict y, unsigned ystride, unsigned nt) {
  const unsigned K = (kT2TileK * per_band) >> 1;
  const unsigned kt = group >> 1;
  float *__restrict yb = y + (group & 1) * kT2Rows;
  gemv_t4_tile(elem, tab, K, kt, kt0 == 0 && kt == 0, yb, yb + ystride, yb + 2 * ystride, yb + 3 * ystride, nt);
}
