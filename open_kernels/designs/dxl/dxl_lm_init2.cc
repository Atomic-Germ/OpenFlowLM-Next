// The running argmax's start (no value, no row) and the band counter's.
#include "dxl_gemv.h"

extern "C" {
void dxl_lm_init2(int32_t *__restrict best, int32_t *__restrict cnt) {
  for (unsigned t = 0; t < kL; ++t) {
    best[t] = (int32_t)0x80000000;
    best[kL + t] = -1;
  }
  cnt[0] = 0;
}
}
