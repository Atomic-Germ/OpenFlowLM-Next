#pragma once
// OPEN-HADAMARD in integers as H32 X H32 on the mmul unit; not bit-exact vs wht.h, fwht_int/fwht_ref.py is its bit-exact model.

#include <aie_api/aie.hpp>
#include <stdint.h>

// inline: one copy per core however many of its translation units include this.
inline int32_t xh_int_scratch[1024] __attribute__((aligned(64)));

alignas(64) inline const int8_t xh_int_h8[2][64] = {
    {1, 1, 1, 1, 1, 1, 1, 1,  1, -1, 1, -1, 1, -1, 1, -1,  1, 1, -1, -1, 1, 1, -1, -1,  1, -1, -1, 1, 1, -1, -1, 1,
     1, 1, 1, 1, -1, -1, -1, -1,  1, -1, 1, -1, -1, 1, -1, 1,  1, 1, -1, -1, -1, -1, 1, 1,  1, -1, -1, 1, -1, 1, 1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1,  -1, 1, -1, 1, -1, 1, -1, 1,  -1, -1, 1, 1, -1, -1, 1, 1,  -1, 1, 1, -1, -1, 1, 1, -1,
     -1, -1, -1, -1, 1, 1, 1, 1,  -1, 1, -1, 1, 1, -1, 1, -1,  -1, -1, 1, 1, 1, 1, -1, -1,  -1, 1, 1, -1, 1, -1, -1, 1}};
alignas(32) inline const int16_t xh_int_h4[2][16] = {
    {1, 1, 1, 1,  1, -1, 1, -1,  1, 1, -1, -1,  1, -1, -1, 1},
    {-1, -1, -1, -1,  -1, 1, -1, 1,  -1, -1, 1, 1,  -1, 1, 1, -1}};
// Row p of H8 as sign bits: bit m set where H8[p][m] = -1.
inline const uint8_t xh_int_h8row[8] = {0x00, 0xAA, 0xCC, 0x66, 0xF0, 0x5A, 0x3C, 0x96};

// The right shift that brings max |v| = m under 2^15.
static inline int xh_int_shift(int m) {
  const int r = 17 - __builtin_clz((unsigned)(m | 1));
  return r < 0 ? 0 : r;
}

// No __restrict: X is x's own storage, safe because each 128-value row is read before it is written.
__attribute__((noinline)) inline int xh_int_q16(const void *x, int16_t *X, bool b16) {
  int A;
  if (b16) {
    const int16_t *xs = (const int16_t *)x;          // |bf16| bit patterns order like the values
    const aie::vector<int16_t, 32> mk = aie::broadcast<int16_t, 32>((int16_t)0x7FFF);
    aie::vector<int16_t, 32> m0 = aie::zeros<int16_t, 32>(), m1 = m0, m2 = m0, m3 = m0;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 1024; j += 128) {
      m0 = aie::max(m0, aie::bit_and(aie::load_v<32>(xs + j), mk));
      m1 = aie::max(m1, aie::bit_and(aie::load_v<32>(xs + j + 32), mk));
      m2 = aie::max(m2, aie::bit_and(aie::load_v<32>(xs + j + 64), mk));
      m3 = aie::max(m3, aie::bit_and(aie::load_v<32>(xs + j + 96), mk));
    }
    A = 141 - (aie::reduce_max(aie::max(aie::max(m0, m1), aie::max(m2, m3))) >> 7);
  } else {
    const int32_t *xs = (const int32_t *)x;
    const aie::vector<int32_t, 32> mk = aie::broadcast<int32_t, 32>(0x7FFFFFFF);
    aie::vector<int32_t, 32> m0 = aie::zeros<int32_t, 32>(), m1 = m0;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 1024; j += 64) {
      m0 = aie::max(m0, aie::bit_and(aie::load_v<32>(xs + j), mk));
      m1 = aie::max(m1, aie::bit_and(aie::load_v<32>(xs + j + 32), mk));
    }
    const int bits = aie::reduce_max(aie::max(m0, m1));
    A = 141 - (bits >> 23) - ((bits & 0x7FFFFF) >= 0x7FFF00 ? 1 : 0);
  }
  if (A > 126) A = 126;                       // zero / tiny block: any shift works
#pragma clang loop unroll(disable)
  for (unsigned p = 0; p < 8; ++p) {
    aie::vector<int16_t, 32> r0, r1, r2, r3;
    if (b16) {
      const bfloat16 *xb = (const bfloat16 *)x + 128 * p;
      r0 = aie::to_fixed<int16_t>(aie::load_v<32>(xb), A);
      r1 = aie::to_fixed<int16_t>(aie::load_v<32>(xb + 32), A);
      r2 = aie::to_fixed<int16_t>(aie::load_v<32>(xb + 64), A);
      r3 = aie::to_fixed<int16_t>(aie::load_v<32>(xb + 96), A);
    } else {
      const float *xf = (const float *)x + 128 * p;
      r0 = aie::to_fixed<int16_t>(aie::load_v<32>(xf), A);
      r1 = aie::to_fixed<int16_t>(aie::load_v<32>(xf + 32), A);
      r2 = aie::to_fixed<int16_t>(aie::load_v<32>(xf + 64), A);
      r3 = aie::to_fixed<int16_t>(aie::load_v<32>(xf + 96), A);
    }
    int16_t *xo = X + 128 * p;
#pragma clang loop unroll(full)
    for (unsigned kb = 0; kb < 4; ++kb) {
      aie::store_v(xo + 32 * kb, r0.template extract<8>(kb));
      aie::store_v(xo + 32 * kb + 8, r1.template extract<8>(kb));
      aie::store_v(xo + 32 * kb + 16, r2.template extract<8>(kb));
      aie::store_v(xo + 32 * kb + 24, r3.template extract<8>(kb));
    }
  }
  return A;
}

__attribute__((noinline)) inline int xh_int_pass_a(int16_t *__restrict X, int32_t *__restrict T) {
  const aie::vector<int8_t, 64> hp = aie::load_v<64>(xh_int_h8[0]), hn = aie::load_v<64>(xh_int_h8[1]);
  aie::vector<int32_t, 32> mx = aie::zeros<int32_t, 32>();
#pragma clang loop unroll(disable)
  for (unsigned p = 0; p < 8; ++p) {
    const aie::vector<int16_t, 32> A0 = aie::load_v<32>(X + 128 * p), A1 = aie::load_v<32>(X + 128 * p + 32);
    const aie::vector<int16_t, 32> A2 = aie::load_v<32>(X + 128 * p + 64), A3 = aie::load_v<32>(X + 128 * p + 96);
    int32_t *__restrict tp = T + 128 * p;
#pragma clang loop unroll(full)
    for (unsigned q = 0; q < 4; ++q) {                      // H4[kb][q] = (-1)^popcount(kb & q)
      aie::mmul<4, 8, 8, int16_t, int8_t> C;
      C.mul(A0, hp);
      C.mac(A1, (q & 1) ? hn : hp);
      C.mac(A2, (q & 2) ? hn : hp);
      C.mac(A3, ((q ^ (q >> 1)) & 1) ? hn : hp);
      const aie::vector<int32_t, 32> t = C.template to_vector<int32_t>();
      mx = aie::max(mx, aie::abs(t));
      aie::store_v(tp + 32 * q, t);
    }
  }
  const int s1 = xh_int_shift(aie::reduce_max(mx));
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 1024; j += 128) {               // four vectors a trip: -Oz makes no hardware loops
#pragma clang loop unroll(full)
    for (unsigned u = 0; u < 128; u += 32) {
      aie::accum<acc32, 32> ta;
      ta.from_vector(aie::load_v<32>(T + j + u));
      aie::store_v(X + j + u, ta.template to_vector<int16_t>(s1));
    }
  }
  return s1;
}

// `sc` = A + 5 - s1 is Y's own shift; the 5 is the 1/32 normalisation.
__attribute__((noinline)) inline void xh_int_pass_b(const int16_t *__restrict Ts, uint8_t *__restrict tab,
                                                    unsigned K, unsigned g, int sc) {
  const unsigned NG = K / 128;
  int16_t *__restrict xi = (int16_t *)tab + 128 * g;
  const aie::vector<int16_t, 16> h4p = aie::load_v<16>(xh_int_h4[0]), h4n = aie::load_v<16>(xh_int_h4[1]);
  aie::vector<int32_t, 16> shv = aie::zeros<int32_t, 16>(), sums = aie::zeros<int32_t, 16>();
#pragma clang loop unroll(disable)
  for (unsigned p = 0; p < 8; ++p) {
    const unsigned row = xh_int_h8row[p];
    aie::mmul<4, 4, 8, int16_t, int16_t> C[4];
    aie::vector<int32_t, 32> mx = aie::zeros<int32_t, 32>(), sy = aie::zeros<int32_t, 32>();
#pragma clang loop unroll(full)
    for (unsigned q = 0; q < 4; ++q) {
      aie::mmul<4, 4, 8, int16_t, int16_t> &M = C[q];
#pragma clang loop unroll(full)
      for (unsigned m = 0; m < 8; ++m) {
        const aie::vector<int16_t, 16> h = ((row >> m) & 1u) ? h4n : h4p;
        const aie::vector<int16_t, 32> t = aie::load_v<32>(Ts + 32 * (4 * m + q));
        if (m == 0)
          M.mul(h, t);
        else
          M.mac(h, t);
      }
      const aie::vector<int32_t, 32> y = M.template to_vector<int32_t>();
      mx = aie::max(mx, aie::abs(y));
      sy = aie::add(sy, y);
    }
    const int r = xh_int_shift(aie::reduce_max(mx));
    int16_t *__restrict xo = xi + 128 * p;
#pragma clang loop unroll(full)
    for (unsigned q = 0; q < 4; ++q) {
      const aie::vector<int16_t, 32> v = C[q].template to_vector<int16_t>(r);
#pragma clang loop unroll(full)
      for (unsigned i = 0; i < 4; ++i)                      // tile row i -> group offset 32 i + 8 q
        aie::store_v(xo + 32 * i + 8 * q, v.template extract<8>(i));
    }
    shv[p] = sc - r;
    sums[p] = aie::reduce_add(sy);                          // exact: |sum| < 2^27
  }
  aie::store_v((int32_t *)(tab + 2 * K) + g, shv.template extract<8>(0));
  aie::accum<accfloat, 16> f;
  f.from_vector(aie::to_float<float>(sums, sc));          // the group sum is taken before Y's per-group rounding
  const aie::vector<bfloat16, 16> hi = f.template to_vector<bfloat16>();
  const aie::vector<bfloat16, 16> lo = aie::sub(f, hi).template to_vector<bfloat16>();
  bfloat16 *__restrict xsh = (bfloat16 *)(tab + 2 * K + 4 * NG) + g;
  aie::store_v(xsh, hi.template extract<8>(0));
  aie::store_v(xsh + NG, lo.template extract<8>(0));
}

__attribute__((noinline)) inline void xh_int_rest(int16_t *X, uint8_t *__restrict tab, unsigned K, unsigned g, int A) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::saturation_mode sat = aie::get_saturation();
  aie::set_saturation(aie::saturation_mode::saturate);
  const int s1 = xh_int_pass_a(X, xh_int_scratch);
  xh_int_pass_b(X, tab, K, g, A + 5 - s1);
  aie::set_saturation(sat);
}

template <typename T>
__attribute__((always_inline)) inline void xh_int_wht(T *x, uint8_t *__restrict tab, unsigned K, unsigned g) {
  int16_t *X = (int16_t *)x;
  xh_int_rest(X, tab, K, g, xh_int_q16(x, X, sizeof(T) == 2));
}
