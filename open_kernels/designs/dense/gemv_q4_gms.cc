#define GEMV_PER_CALL 2
#include "gemv_q4.h"
// A 64-row band into the silu scratch at ms + dst (the up band at 0, the gate band at 64).
extern "C" {
void gemv_q4_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                 int32_t group, int32_t per_band, int32_t dst) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}
}
