// One weight chunk of a band's slice into the band's [L][64] accumulator (dxl_gemv.h).
#include "dxl_gemv.h"

extern "C" {
void dxl_gemv(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
              int32_t c, int32_t s, int32_t ks, int32_t kt_total) {
  dxl_gemv_chunk(t, tab, y, (unsigned)c, (unsigned)s, (unsigned)ks, (unsigned)kt_total);
}
}
