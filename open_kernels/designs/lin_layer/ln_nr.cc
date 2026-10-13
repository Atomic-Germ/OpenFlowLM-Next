// Layer RMSNorm without a residual add (the layer-entry norm: add == 0):
//   xn = bf16( x * rsqrt(mean(x^2) + 1e-6) * w )
// Elements are 4 KB: x as two fp32[1024] halves; w, xn as bf16[2048].
// Same arithmetic as designs/ln/ln.cc (bit-identical xn for add == 0).
//
// LN_GROUPS=2 selects K2's GroupRMSNorm(2): one RMS reduction per contiguous half
// (mean over kHalf elements, eps inside rsqrt), channel-wise weight over the whole
// width. Default 1 keeps the single-reduction arithmetic bit-identical.
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
#if LN_GROUPS == 2
static_assert(LN_N % (2 * kV) == 0, "LN_GROUPS=2 needs each half a multiple of the vector width");
#endif

extern "C" {
void ln_nr(const float *__restrict x0, const float *__restrict x1, const bfloat16 *__restrict w,
           bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#if LN_GROUPS == 2
  accf32 ss0 = aie::zeros<accfloat, kV>();
  accf32 ss1 = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    const v32f y = aie::load_v<kV>(xp);
    v32b h, l;
    split32(y, h, l);
    if (j < kHalf) {
      ss0 = aie::mac(ss0, h, h);
      ss0 = aie::mac(ss0, h, l);
      ss0 = aie::mac(ss0, h, l);
    } else {
      ss1 = aie::mac(ss1, h, h);
      ss1 = aie::mac(ss1, h, l);
      ss1 = aie::mac(ss1, h, l);
    }
  }
  const float inv0 = srsqrt(aie::reduce_add(ss0.template to_vector<float>()) * (1.0f / kHalf) + LN_EPS);
  const float inv1 = srsqrt(aie::reduce_add(ss1.template to_vector<float>()) * (1.0f / kHalf) + LN_EPS);
  const bfloat16 ih0 = (bfloat16)inv0;
  const bfloat16 il0 = (bfloat16)(inv0 - (float)ih0);
  const bfloat16 ih1 = (bfloat16)inv1;
  const bfloat16 il1 = (bfloat16)(inv1 - (float)ih1);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, aie::load_v<kV>(xp), aie::load_v<kV>(w + j));
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), (j < kHalf) ? ih0 : ih1, (j < kHalf) ? il0 : il1);
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
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
