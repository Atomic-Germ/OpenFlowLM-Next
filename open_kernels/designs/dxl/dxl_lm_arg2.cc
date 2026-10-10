// Strict > on greedy_key, so the first maximal row wins as in the app's greedy.
#include "dxl_gemv.h"

extern "C" {
void dxl_lm_arg2(const float *__restrict y, int32_t *__restrict best, int32_t *__restrict cnt, int32_t band0,
                 int32_t real_vocab) {
  const int32_t *__restrict yi = (const int32_t *)y;
  const int32_t row0 = (band0 + cnt[0]) * 64;
  cnt[0] += 1;
  const int32_t valid = (real_vocab - row0 < 64) ? real_vocab - row0 : 64;
  for (unsigned t = 0; t < kL; ++t) {
    int32_t bv = best[t], bi = best[kL + t];
    for (int32_t r = 0; r < valid; ++r) {
      const int32_t v = yi[t * kBandFloats + r];
      const int32_t sv = greedy_key(v);
      if (sv > bv) {
        bv = sv;
        bi = row0 + r;
      }
    }
    best[t] = bv;
    best[kL + t] = bi;
  }
}
}
