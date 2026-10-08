// h = silu(g) * u for band b of all DXL_L tokens into the y element; u is accumulator band b,
// g is band g0 + b (the core's [band][L][64] accumulators). dense_act's arithmetic per token.
#include "dxl_gemv.h"
#include "vecmath.h"

extern "C" {
void dxl_act(const float *__restrict acc, float *__restrict h, int32_t b, int32_t g0) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = acc + (unsigned)b * kL * kBandFloats;
  const float *__restrict g = acc + (unsigned)(g0 + b) * kL * kBandFloats;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kL * kBandFloats; j += 32) {
    const v32f a = vsiluN<32>(aie::load_v<32>(g + j));
    aie::store_v(h + j, fmul32(a, aie::load_v<32>(u + j)));
  }
}
}
