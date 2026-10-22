// One weight chunk into band b of a [band][L][64] accumulator (b = 0: a y fifo element).
#include "dxl_gemv.h"

extern "C" {
void dxl_gemv_at(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                 int32_t b, int32_t c, int32_t s, int32_t ks, int32_t kt_total) {
  dxl_gemv_chunk(t, tab, y + (unsigned)b * kL * kBandFloats, (unsigned)c, (unsigned)s, (unsigned)ks,
                 (unsigned)kt_total);
}
}
