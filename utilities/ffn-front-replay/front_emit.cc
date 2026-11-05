#include "gemv_q4.h"
#include "vecmath.h"
extern "C" {
void front_emit(const float *ms, float *y, int32_t which) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned j = 0; j < 64; j += 32) {
    auto u = aie::load_v<32>(ms + j), g = aie::load_v<32>(ms + 64 + j);
    auto h = which == 0 ? u : (which == 1 ? g : fmul32(vsiluN<32>(g), u));
    aie::store_v(y + j, h);
  }
}
}
