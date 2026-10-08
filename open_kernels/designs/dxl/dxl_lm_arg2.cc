// The running argmax over the next head band (the core's cnt[0]-th, from band0): float bits as an
// order-preserving int, strict > so the first maximal row wins; rows at or past real_vocab never win.
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
      const int32_t sv = v < 0 ? v ^ 0x7fffffff : v;
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
