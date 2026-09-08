#define GEMV_PER_CALL 2
#include "gemv_q4.h"
// A 64-row band into the silu scratch at ms + dst (the up band at 0, the gate band at 64).
#define GEMV_Q4_WRAP__(PFX, NAME) PFX##_g##NAME
#define GEMV_Q4_WRAP(PFX, NAME) GEMV_Q4_WRAP__(PFX, NAME)
extern "C" {
void GEMV_Q4_WRAP(GEMV_Q4_PREFIX, ms)(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                                      int32_t group, int32_t per_band, int32_t dst) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}
}
