#include "ln.h"

extern "C" void ln_stream_xn(const float *__restrict saved, const float *__restrict sums,
                             const bfloat16 *__restrict w, bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float inv = srsqrt(aie::reduce_add(aie::load_v<kV>(sums)) * (1.0f / kN) + LN_EPS);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, aie::load_v<kV>(saved + j), aie::load_v<kV>(w + j));
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), ih, il);
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
}
