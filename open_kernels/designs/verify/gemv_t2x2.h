#pragma once
// A rows [t0, t1, t0, t1] keep each token's int32 sums exact; with gemv_t2_tile's epilogue order that is bit-identical per token.

#include "gemv_t2.h"

// GEMV_X2_HOIST 1 (vux.py's) shares one A/code load across all four field accumulators: same exact sums, faster and smaller.
#ifndef GEMV_X2_HOIST
#define GEMV_X2_HOIST 0
#endif

__attribute__((noinline)) inline void gemv_t2x2_tile(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                                     unsigned ts, unsigned K, unsigned kt, bool first,
                                                     float *__restrict y0, float *__restrict y1) {
  event0();
#ifdef GEMV_NULL
  if (first) {
    aie::store_v(y0, aie::zeros<float, kT2Rows>());
    aie::store_v(y1, aie::zeros<float, kT2Rows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NG = K / 128;
  const bfloat16 *__restrict sp = (const bfloat16 *)tile;
  const uint8_t *__restrict code = tile + kT2ScaleBytes;
  const uint8_t *__restrict tab1 = tab + ts;
  const int16_t *__restrict xi0 = (const int16_t *)tab + kt * kT2TileK;
  const int16_t *__restrict xi1 = (const int16_t *)tab1 + kt * kT2TileK;
  const int32_t *__restrict sh0 = (const int32_t *)(tab + 2 * K) + kt * 2;
  const int32_t *__restrict sh1 = (const int32_t *)(tab1 + 2 * K) + kt * 2;
  const bfloat16 *__restrict xsh0 = (const bfloat16 *)(tab + 2 * K + 4 * NG) + kt * 2;
  const bfloat16 *__restrict xsh1 = (const bfloat16 *)(tab1 + 2 * K + 4 * NG) + kt * 2;
  const bfloat16 *__restrict xsl0 = xsh0 + NG;
  const bfloat16 *__restrict xsl1 = xsh1 + NG;

  aie::accum<accfloat, kT2Rows> acc0, acc1;
  if (first) {
    acc0 = aie::zeros<accfloat, kT2Rows>();
    acc1 = aie::zeros<accfloat, kT2Rows>();
  } else {
    acc0.from_vector(aie::load_v<kT2Rows>(y0));
    acc1.from_vector(aie::load_v<kT2Rows>(y1));
  }
  const aie::vector<bfloat16, kT2Rows> fsc = gemv_t2_field_scale();

#pragma clang loop unroll(disable)
  for (unsigned g = 0; g < 2; ++g) {
    const aie::vector<bfloat16, kT2Rows> s32 = aie::load_v<kT2Rows>(sp + g * kT2Rows);
    const aie::vector<bfloat16, kT2Rows> ds = aie::mul(s32, fsc).template to_vector<bfloat16>();   // exact
    const int16_t *__restrict xg0 = xi0 + g * 128;
    const int16_t *__restrict xg1 = xi1 + g * 128;
    const uint8_t *__restrict cg = code + g * 1024;
    aie::vector<int32_t, 8> ve0[2], vo0[2], ve1[2], vo1[2];
#if GEMV_X2_HOIST
    {
      aie::mmul<4, 8, 8, int16_t, uint8_t> Ce0, Co0, Ce1, Co1;
#pragma clang loop unroll(full)
      for (unsigned o = 0; o < 16; ++o) {
        const aie::vector<int16_t, 32> A =
            aie::concat(aie::load_v<8>(xg0 + o * 8), aie::load_v<8>(xg1 + o * 8)).template grow_replicate<32>();
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
      ve0[0] = ce0.template extract<8>(0);
      ve1[0] = ce0.template extract<8>(1);
      vo0[0] = co0.template extract<8>(0);
      vo1[0] = co0.template extract<8>(1);
      ve0[1] = ce1.template extract<8>(0);
      ve1[1] = ce1.template extract<8>(1);
      vo0[1] = co1.template extract<8>(0);
      vo1[1] = co1.template extract<8>(1);
    }
#else
#pragma clang loop unroll(full)
    for (unsigned jp = 0; jp < 2; ++jp) {
      const uint8_t me = jp == 0 ? (uint8_t)0x03 : (uint8_t)0x30;   // rows 0..7  / 16..23
      const uint8_t mo = jp == 0 ? (uint8_t)0x0C : (uint8_t)0xC0;   // rows 8..15 / 24..31
      aie::mmul<4, 8, 8, int16_t, uint8_t> Ce, Co;
#if GEMV_T2_OUNROLL >= 15
#pragma clang loop unroll(full)
      for (unsigned o = 0; o < 16; ++o) {
        const aie::vector<int16_t, 32> A =
            aie::concat(aie::load_v<8>(xg0 + o * 8), aie::load_v<8>(xg1 + o * 8)).template grow_replicate<32>();
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(cg + o * 64);
        const aie::vector<uint8_t, 64> e = aie::bit_and(me, q);
        const aie::vector<uint8_t, 64> od = aie::bit_and(mo, q);
        if (o == 0) {
          Ce.mul(A, e);
          Co.mul(A, od);
        } else {
          Ce.mac(A, e);
          Co.mac(A, od);
        }
      }
#else
      {
        const aie::vector<int16_t, 32> A =
            aie::concat(aie::load_v<8>(xg0), aie::load_v<8>(xg1)).template grow_replicate<32>();
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(cg);
        Ce.mul(A, aie::bit_and(me, q));
        Co.mul(A, aie::bit_and(mo, q));
      }
      GEMV_T2_UNROLL_(GEMV_T2_OUNROLL)
      for (unsigned o = 1; o < 16; ++o) {
        const aie::vector<int16_t, 32> A =
            aie::concat(aie::load_v<8>(xg0 + o * 8), aie::load_v<8>(xg1 + o * 8)).template grow_replicate<32>();
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(cg + o * 64);
        Ce.mac(A, aie::bit_and(me, q));
        Co.mac(A, aie::bit_and(mo, q));
      }
#endif
      const aie::vector<int32_t, 32> ce = Ce.template to_vector<int32_t>();
      const aie::vector<int32_t, 32> co = Co.template to_vector<int32_t>();
      ve0[jp] = ce.template extract<8>(0);
      ve1[jp] = ce.template extract<8>(1);
      vo0[jp] = co.template extract<8>(0);
      vo1[jp] = co.template extract<8>(1);
    }
#endif
    {
      const aie::vector<int32_t, kT2Rows> vi = aie::concat(ve0[0], vo0[0], ve0[1], vo0[1]);
      aie::accum<accfloat, kT2Rows> part;
      part.from_vector(aie::to_float<float>(vi, sh0[g]));
      const aie::vector<bfloat16, kT2Rows> hi = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, kT2Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc0 = aie::mac(acc0, hi, ds);
      acc0 = aie::mac(acc0, lo, ds);
      acc0 = aie::msc(acc0, s32, xsh0[g]);
      acc0 = aie::msc(acc0, s32, xsl0[g]);
    }
    {
      const aie::vector<int32_t, kT2Rows> vi = aie::concat(ve1[0], vo1[0], ve1[1], vo1[1]);
      aie::accum<accfloat, kT2Rows> part;
      part.from_vector(aie::to_float<float>(vi, sh1[g]));
      const aie::vector<bfloat16, kT2Rows> hi = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, kT2Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc1 = aie::mac(acc1, hi, ds);
      acc1 = aie::mac(acc1, lo, ds);
      acc1 = aie::msc(acc1, s32, xsh1[g]);
      acc1 = aie::msc(acc1, s32, xsl1[g]);
    }
  }
  aie::store_v(y0, acc0.template to_vector<float>());
  aie::store_v(y1, acc1.template to_vector<float>());
  event1();
#endif
}

// Must keep gemv_t2_chunk_rt's band law so both tokens address the 1-wide tables unchanged.
__attribute__((noinline)) inline void gemv_t2x2_chunk_rt(const uint8_t *__restrict elem, const uint8_t *__restrict tab,
                                                         unsigned ts, unsigned group, unsigned pbs,
                                                         float *__restrict y0, float *__restrict y1, unsigned rs) {
  const unsigned sub = pbs >> 16, per_band = pbs & 0xFFFFu;
  const unsigned sh = (rs == 4) ? 2u : 1u;
  const unsigned K = (kT2TileK * per_band) >> sh;
  const unsigned c = group * kT2PerCall + sub;
  const unsigned kt = c >> sh;
  const unsigned off = (c & (rs - 1)) * kT2Rows;
  gemv_t2x2_tile(elem + sub * kT2Chunk, tab, ts, K, kt, kt == 0, y0 + off, y1 + off);
}
