#include "ln.h"
#include "vecmath_precise.h"

extern "C" void ln_stream_xn(const float *__restrict saved, const float *__restrict sums,
                             const bfloat16 *__restrict w, bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float inv = srsqrt(aie::reduce_add(aie::load_v<kV>(sums)) * (1.0f / kN) + LN_EPS);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    accf32 weight(aie::load_v<kV>(w + j));
    const v32f t = precise_mulN<kV>(aie::load_v<kV>(saved + j), weight.template to_vector<float>());
    accf32 o(precise_mulN<kV>(t, aie::broadcast<float, kV>(inv)));
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
}
