// Layer RMSNorm without a residual add (the layer-entry norm: add == 0):
//   xn = bf16( x * rsqrt(mean(x^2) + 1e-6) * w )
// Elements are 4 KB: x as two fp32[1024] halves; w, xn as bf16[2048].
// Same arithmetic as designs/ln/ln.cc (bit-identical xn for add == 0).
#include "vecmath.h"

static constexpr unsigned kN = 2048;
static constexpr unsigned kV = 32;

extern "C" {
void ln_nr(const float *__restrict x0, const float *__restrict x1, const bfloat16 *__restrict w,
           bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < 1024) ? (x0 + j) : (x1 + (j - 1024));
    const v32f y = aie::load_v<kV>(xp);
    v32b h, l;
    split32(y, h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kN) + 1e-6f);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < 1024) ? (x0 + j) : (x1 + (j - 1024));
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, aie::load_v<kV>(xp), aie::load_v<kV>(w + j));
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), ih, il);
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
}
}
