// h part = silu(g) * u for this core's 64 rows (ms: u @544, g @608), emitted as f32 (one 256 B y
// element); the bf16 rounding happens in the consumer's table prep.
#include "vecmath.h"

extern "C" {
void moe_silu32(const float *__restrict ms, float *__restrict h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + 544;
  const float *__restrict g = ms + 608;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32)
    aie::store_v(h + j, fmul32(vsiluN<32>(aie::load_v<32>(g + j)), aie::load_v<32>(u + j)));
}
}
