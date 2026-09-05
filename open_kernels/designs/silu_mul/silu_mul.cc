// h = bf16(silu(g) * u) for one 512-wide expert hidden (g, u fp32[512] elements; h bf16[512]).
#include "vecmath.h"
static constexpr unsigned kN = 512;
static constexpr unsigned kV = 32;
extern "C" {
void silu_mul(const float *__restrict g, const float *__restrict u, bfloat16 *__restrict h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const v32f r = fmul32(vsiluN<32>(aie::load_v<kV>(g + j)), aie::load_v<kV>(u + j));
    accf32 a;
    a.from_vector(r);
    aie::store_v(h + j, a.template to_vector<bfloat16>());
  }
}
}
