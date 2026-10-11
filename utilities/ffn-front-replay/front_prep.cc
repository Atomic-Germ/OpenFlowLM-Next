#include "gemv_q4.h"
#include "vecmath.h"
extern "C" {
void front_prep(const float *x, uint8_t *tab, int32_t k, int32_t i) {
  unsigned b0 = 32 * i, nb = std::min(32u, (unsigned)k / 32 - b0);
#if FRONT_FP32
  gemv_q4_prep_f32_direct_blocks(x, tab, k, b0, nb);
#else
  gemv_q4_prep_f32_blocks(x, tab, k, b0, nb);
#endif
}

}
