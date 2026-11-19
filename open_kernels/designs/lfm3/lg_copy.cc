// One accumulated down band (64 floats of acc, band `band`) into its y element.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void lg_copy(const float *__restrict acc, float *__restrict y, int32_t band) {
  const float *__restrict a = acc + band * 64;
#pragma clang loop unroll(full)
  for (unsigned j = 0; j < 64; j += 16) aie::store_v(y + j, aie::load_v<16>(a + j));
}
}
