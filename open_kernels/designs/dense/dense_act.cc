// h band = act(g) * u for one 64-row band (ms: u @0, g @64) -> one f32 y element.
// act = silu: silu(x) = x sigmoid(x); gelu_tanh(x) = 0.5 x (1 + tanh(z)) = x sigmoid(2z),
// z = sqrt(2/pi) (x + 0.044715 x^3). Vector ops only (no scalar float on this core).
#include "vecmath.h"

extern "C" {
void dense_act(const float *__restrict ms, float *__restrict h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + 0;
  const float *__restrict g = ms + 64;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {
    const v32f x = aie::load_v<32>(g + j);
#if 0
    const v32f x3 = fmul32(fmul32(x, x), x);
    const v32f z2 = fscaleN<32>(fadd32(x, fscaleN<32>(x3, 0.044715f)), 2.0f * 0.7978845608028654f);
    const v32f a = fmul32(x, vsigmoidN<32>(z2));
#else
    const v32f a = vsiluN<32>(x);
#endif
    aie::store_v(h + j, fmul32(a, aie::load_v<32>(u + j)));
  }
}
}
