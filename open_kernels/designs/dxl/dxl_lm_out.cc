// The core's [value L | row L] into one y element (the rest zero), the last its stream carries.
#include "dxl_gemv.h"

extern "C" {
void dxl_lm_out(const int32_t *__restrict best, float *__restrict yf) {
  int32_t *__restrict y = (int32_t *)yf;
  for (unsigned i = 0; i < kL * kBandFloats; ++i) y[i] = i < 2 * kL ? best[i] : 0;
}
}
