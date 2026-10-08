// Strict > on order-preserving int bits, so the first maximal row wins as in the host argmax.
#include "dxl_gemv.h"

extern "C" {
void dxl_lm_arg(const float *__restrict y, int32_t *__restrict best, int32_t band0, int32_t b,
                int32_t real_vocab) {
  const int32_t row0 = (band0 + b) * 64;
  const int32_t *__restrict yi = (const int32_t *)y;
  const int32_t valid = (real_vocab - row0 < 64) ? real_vocab - row0 : 64;
  for (unsigned t = 0; t < kL; ++t) {
    int32_t bv = best[t], bi = best[kL + t];
    for (int32_t r = 0; r < valid; ++r) {
      const int32_t v = yi[t * kBandFloats + r];
      const int32_t s = v < 0 ? v ^ 0x7fffffff : v;
      if (s > bv) {
        bv = s;
        bi = row0 + r;
      }
    }
    best[t] = bv;
    best[kL + t] = bi;
  }
}
}
