// moe_experts: h[64] = bf16(silu(g) * u), one core's 64 rows of the expert hidden.
#include "vecmath.h"

extern "C" {
void moe_silu(const float *__restrict g, const float *__restrict u, bfloat16 *__restrict h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {
    const v32f r = fmul32(vsiluN<32>(aie::load_v<32>(g + j)), aie::load_v<32>(u + j));
    accf32 a;
    a.from_vector(r);
    aie::store_v(h + j, a.template to_vector<bfloat16>());
  }
}
}
