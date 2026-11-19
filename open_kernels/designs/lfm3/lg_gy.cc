#define GEMV_PER_CALL 2
#define GEMV_Q4_0_NATIVE 1      // LFM3_Q40: 4608-B chunks, m = -8 d
#include "gemv_q4.h"
// A q4_1 band into its y element (runtime band law: per_band chunks, row split rs).
extern "C" {
void lg_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
           int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
