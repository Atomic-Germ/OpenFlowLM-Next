// One 10 KB weight element = the two 32-row halves of k-tile e of a band's slice s, into accumulator
// band b ([band][L][64]); dxl_gemv_chunk per half.
#include "dxl_gemv.h"

extern "C" {
void dxl_gemv2(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict acc, int32_t b,
               int32_t e, int32_t s, int32_t ks, int32_t kt_total) {
  float *y = acc + (unsigned)b * kL * kBandFloats;
  dxl_gemv_chunk(t, tab, y, 2u * (unsigned)e, (unsigned)s, (unsigned)ks, (unsigned)kt_total);
  dxl_gemv_chunk(t + kTileBytes, tab, y, 2u * (unsigned)e + 1u, (unsigned)s, (unsigned)ks, (unsigned)kt_total);
}
}
