// h band = silu(g) * u for one 64-row band (ms: u @0, g @64) -> one f32 y element.
// silu(x) = x sigmoid(x). Vector ops only (no scalar float on this core).
#include "vecmath.h"

extern "C" {
void dense_act(const float *__restrict ms, float *__restrict h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + 0;
  const float *__restrict g = ms + 64;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32)
    aie::store_v(h + j, fmul32(vsiluN<32>(aie::load_v<32>(g + j)), aie::load_v<32>(u + j)));
}
}
