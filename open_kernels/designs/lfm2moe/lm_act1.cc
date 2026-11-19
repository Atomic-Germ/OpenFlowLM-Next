// h band = silu(g) * u for one 64-row band (ms: u @0, g @64) -> one f32 y
// element. silu(x) = x sigmoid(x) as the dense dx computes it (vsiluN).
#include "vecmath.h"

extern "C" {
void lm_act1(const float *__restrict ms, float *__restrict h, const float *__restrict wv, int32_t e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + 0;
  const float *__restrict g = ms + 64;
  (void)wv;
  (void)e;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {
    const v32f a = vsiluN<32>(aie::load_v<32>(g + j));
    aie::store_v(h + j, fmul32(a, aie::load_v<32>(u + j)));
  }
}
}
