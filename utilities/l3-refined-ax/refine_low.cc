#include <stdint.h>
// Call the SAME production entry, preserving its accumulation and rounding.
extern "C" void gemv_q4_gy(const uint8_t *, const uint8_t *, float *, int32_t, int32_t, int32_t);
extern "C" void refine_low(const uint8_t *__restrict t, const uint8_t *__restrict tab,
                            float *__restrict y, int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_gy(t, tab + 11520, y, group, per_band, rs);
}
