#include "ln.h"
#include "vecmath_precise.h"

extern "C" void ln_stream_acc(const float *__restrict add, float *__restrict saved,
                              float *__restrict sums, float *__restrict out, int half) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 ss = aie::zeros<accfloat, kV>();
  if (half != 0) ss = accf32(aie::load_v<kV>(sums));
#if LN_STREAM_COMPENSATED
  v32f correction = half ? aie::load_v<kV>(sums+kV) : aie::zeros<float,kV>();
#endif
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) {
    float *yp = saved + half * kHalf + j;
    const v32f y = fadd32(aie::load_v<kV>(yp), aie::load_v<kV>(add + j));
    aie::store_v(yp, y);
    aie::store_v(out + j, y);
#if LN_STREAM_COMPENSATED
    const v32f old = ss.template to_vector<float>();
    const v32f term = fsub32(precise_mulN<kV>(y,y),correction);
    const v32f total = fadd32(old,term);
    correction = fsub32(fsub32(total,old),term);
    ss.from_vector(total);
#else
    ss = aie::add(ss, precise_mulN<kV>(y, y));
#endif
  }
  aie::store_v(sums, ss.template to_vector<float>());
#if LN_STREAM_COMPENSATED
  aie::store_v(sums+kV,correction);
#endif
}
