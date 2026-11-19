#define GEMV_PER_CALL 2
#define GEMV_Q4_0_NATIVE 1      // LFM3_Q40: 4608-B chunks, m = -8 d
#include "gemv_q4.h"
// A q4_1 64-row band of a slot's gate or up into the scratch at ms + dst (up at 0, gate at 64).
extern "C" {
void lg_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
            int32_t group, int32_t per_band, int32_t dst) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}
}
