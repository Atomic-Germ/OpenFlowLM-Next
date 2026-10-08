// Layer RMSNorm without a residual add (the layer-entry norm: add == 0):
//   xn = bf16( x * rsqrt(mean(x^2) + 1e-6) * w )
// Elements are 4 KB: x as two fp32[1024] halves; w, xn as bf16[2048].
// Same arithmetic as designs/ln/ln.cc (bit-identical xn for add == 0).
//
// LN_GROUPS=G: K2's GroupRMSNorm, an RMS per contiguous group (eps inside rsqrt) but a channel-wise weight; 1 is bit-identical.
#include "vecmath.h"

// LN_N: the width (designs/ln/ln.py and the whole-layer designs pass it from the ModelSpec);
// elements are LN_N*2 bytes: x / add / y as two f32 halves, w / xn as one bf16 element.
#ifndef LN_N
#define LN_N 2048
#endif
#ifndef LN_EPS
#define LN_EPS 1e-6f
#endif
#ifndef LN_GROUPS
#define LN_GROUPS 1
#endif
static constexpr unsigned kN = LN_N;
static constexpr unsigned kHalf = LN_N / 2;
static constexpr unsigned kV = 32;
static_assert(kHalf % kV == 0, "LN_N must be a multiple of 64");
#if LN_GROUPS > 1
static constexpr unsigned kGrp = LN_N / LN_GROUPS;
static_assert(LN_N % LN_GROUPS == 0 && kGrp % kV == 0, "LN_GROUPS needs each group a multiple of the vector width");
static_assert(kHalf % kGrp == 0, "a group may not straddle the two fp32 halves");
#endif

extern "C" {
void ln_nr(const float *__restrict x0, const float *__restrict x1, const bfloat16 *__restrict w,
           bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#if LN_GROUPS > 1
  for (unsigned g = 0; g < LN_GROUPS; ++g) {
    accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
    for (unsigned j = g * kGrp; j < (g + 1) * kGrp; j += kV) {
      const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
      const v32f y = aie::load_v<kV>(xp);
      v32b h, l;
      split32(y, h, l);
      ss = aie::mac(ss, h, h);
      ss = aie::mac(ss, h, l);
      ss = aie::mac(ss, h, l);
    }
    const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kGrp) + LN_EPS);
    const bfloat16 ih = (bfloat16)inv;
    const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
    for (unsigned j = g * kGrp; j < (g + 1) * kGrp; j += kV) {
      const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
      accf32 t = aie::zeros<accfloat, kV>();
      t = mac_vv(t, aie::load_v<kV>(xp), aie::load_v<kV>(w + j));
      accf32 o = aie::zeros<accfloat, kV>();
      o = mac_vs(o, t.template to_vector<float>(), ih, il);
      aie::store_v(xn + j, o.template to_vector<bfloat16>());
    }
  }
#else
  accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    const v32f y = aie::load_v<kV>(xp);
    v32b h, l;
    split32(y, h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kN) + LN_EPS);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, aie::load_v<kV>(xp), aie::load_v<kV>(w + j));
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), ih, il);
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
#endif
}
}
