// Opt-in AB projection with a persistent low component across all input chunks.
#include "dn_glue.h"
#include "gemv_tab.h"

extern "C" void wide_ab_carry(const bfloat16 *__restrict w,
                              const bfloat16 *__restrict xn,
                              float *__restrict acc, int tile, int first) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 sum;
  v32f low;
  if (first && tile == 0) {
    sum = aie::zeros<accfloat,32>();
    low = aie::zeros<float,32>();
  } else {
    sum.from_vector(aie::load_v<32>(acc));
    low = aie::load_v<32>(acc + 32);
  }
  const bfloat16 *x = xn + tile * 64;
#pragma clang loop unroll(disable)
  for (unsigned r = 0; r < 64; ++r)
    q4_product_add(sum, low, aie::load_v<32>(w + r * 32), aie::broadcast<bfloat16,32>(x[r]));
  aie::store_v(acc, sum.to_vector<float>());
  aie::store_v(acc + 32, low);
}
