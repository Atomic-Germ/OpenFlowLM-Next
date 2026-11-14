#include "dn_glue.h"

// Production [32 rows,64 lanes] BF16 weight tiles; FP32 input is split only
// at the MAC, keeping both high and low terms. Accumulation order is unchanged.
extern "C" void linear_ab_fp32(const bfloat16 *__restrict W, const float *__restrict xn,
                               float *__restrict acc, int tile, int block) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 lo, hi;
  if (block == 0 && tile == 0) {
    lo = aie::zeros<accfloat, kV>(); hi = aie::zeros<accfloat, kV>();
  } else {
    lo.from_vector(aie::load_v<kV>(acc)); hi.from_vector(aie::load_v<kV>(acc + kV));
  }
#pragma clang loop unroll(disable)
  for (unsigned r = 0; r < 32; ++r) {
    v32b xh, xl;
    split32(aie::broadcast<float,kV>(xn[(unsigned)tile * 32 + r]), xh, xl);
    const auto wl = aie::load_v<kV>(W + r * 64);
    const auto wh = aie::load_v<kV>(W + r * 64 + kV);
    lo = aie::mac(lo, wl, xh); lo = aie::mac(lo, wl, xl);
    hi = aie::mac(hi, wh, xh); hi = aie::mac(hi, wh, xl);
  }
  aie::store_v(acc,lo.template to_vector<float>());
  aie::store_v(acc+kV,hi.template to_vector<float>());
}
