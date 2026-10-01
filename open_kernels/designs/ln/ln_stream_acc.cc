#include "ln.h"
#include "vecmath_precise.h"
#if LN_RESIDUAL_RNE
using namespace aie::operators;
#include "fp32_add_rne.h"
struct ResidualRne {
  using V = aie::vector<uint32_t, 32>;
  static V v(uint32_t x) { return aie::broadcast<uint32_t, 32>(x); }
  static V choose(aie::mask<32> c, V a, V b) { return aie::select(b, a, c); }
};
__attribute__((noinline)) static v32f residual_add_rne(v32f a, v32f b) {
  return fp32_add_rne<ResidualRne>(a.cast_to<uint32_t>(), b.cast_to<uint32_t>()).cast_to<float>();
}
#endif

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
#if LN_RESIDUAL_RNE
    const v32f y = residual_add_rne(aie::load_v<kV>(yp), aie::load_v<kV>(add + j));
#else
    const v32f y = fadd32(aie::load_v<kV>(yp), aie::load_v<kV>(add + j));
#endif
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
