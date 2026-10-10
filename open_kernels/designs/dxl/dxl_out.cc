// Accumulator band b ([L][64] floats of the core's [band][L][64]) into the y element.
#include "dxl_gemv.h"

extern "C" {
void dxl_out(const float *__restrict acc, float *__restrict y, int32_t b) {
  const float *__restrict a = acc + (unsigned)b * kL * kBandFloats;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kL * kBandFloats; j += 32)
    aie::store_v(y + j, aie::load_v<32>(a + j));
}
}
