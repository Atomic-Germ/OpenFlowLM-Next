// The running argmax's start: no value yet, no row.
#include "dxl_gemv.h"

extern "C" {
void dxl_lm_init(int32_t *__restrict best) {
  for (unsigned t = 0; t < kL; ++t) {
    best[t] = (int32_t)0x80000000;
    best[kL + t] = -1;
  }
}
}
