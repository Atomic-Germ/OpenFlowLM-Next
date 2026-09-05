// y element j (64 floats) = rows 64j..64j+63 of this core's 256-row block output (ms: acc @288).
#include "vecmath.h"

extern "C" {
void moe_out(const float *__restrict ms, float *__restrict y, int32_t j) {
  const float *__restrict acc = ms + 288;
  aie::store_v(y, aie::load_v<32>(acc + 64 * j));
  aie::store_v(y + 32, aie::load_v<32>(acc + 64 * j + 32));
}
}
