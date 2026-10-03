#include "ln.h"

extern "C" void ln_stream_acc(const float *__restrict add, float *__restrict saved,
                              float *__restrict sums, float *__restrict out, int half) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 ss = aie::zeros<accfloat, kV>();
  if (half != 0) ss = accf32(aie::load_v<kV>(sums));
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) {
    float *yp = saved + half * kHalf + j;
    const v32f y = fadd32(aie::load_v<kV>(yp), aie::load_v<kV>(add + j));
    aie::store_v(yp, y);
    aie::store_v(out + j, y);
    v32b h, l;
    split32(y, h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  aie::store_v(sums, ss.template to_vector<float>());
}
