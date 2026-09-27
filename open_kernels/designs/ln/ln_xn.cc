// One entry point per TU (IRON compiles a source once per ExternalFunction); the math is ln.h.
#include "ln.h"

extern "C" {
// xn = bf16((x + a) * inv * w) (one output element; the statistics over both halves)
void ln_xn(const float *__restrict x0, const float *__restrict x1, const float *__restrict a0,
           const float *__restrict a1, const bfloat16 *__restrict w, bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#if LN_GROUPS == 2
  float inv0, inv1;
  ln_inv_g2(x0, x1, a0, a1, &inv0, &inv1);
  const bfloat16 ih0 = (bfloat16)inv0;
  const bfloat16 il0 = (bfloat16)(inv0 - (float)ih0);
  const bfloat16 ih1 = (bfloat16)inv1;
  const bfloat16 il1 = (bfloat16)(inv1 - (float)ih1);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    const float *ap = (j < kHalf) ? (a0 + j) : (a1 + (j - kHalf));
    const v32f y = fadd32(aie::load_v<kV>(xp), aie::load_v<kV>(ap));
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, y, aie::load_v<kV>(w + j));
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), (j < kHalf) ? ih0 : ih1, (j < kHalf) ? il0 : il1);
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
#else
  const float inv = ln_inv(x0, x1, a0, a1);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    const float *ap = (j < kHalf) ? (a0 + j) : (a1 + (j - kHalf));
    const v32f y = fadd32(aie::load_v<kV>(xp), aie::load_v<kV>(ap));
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, y, aie::load_v<kV>(w + j));
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), ih, il);
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
#endif
}
}
