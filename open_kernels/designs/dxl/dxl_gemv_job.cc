#include "dxl_job.h"

extern "C" {
void dxl_gemv_job(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict acc,
                  const int32_t *__restrict jp, int32_t b, int32_t c, int32_t s) {
  dxl_gemv_chunk(t, tab, acc + (unsigned)(jp[F_B0] + b) * kL * kBandFloats, (unsigned)c, (unsigned)(s + jp[F_S0]),
                 (unsigned)jp[F_KS], (unsigned)jp[F_KT]);
}
}
