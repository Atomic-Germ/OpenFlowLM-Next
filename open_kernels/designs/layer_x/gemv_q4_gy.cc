#define GEMV_PER_CALL 2
#include "gemv_q4.h"
// A projection band into its y element: (per_band, rs) = (16, 2) K=2048, (32, 2) K=4096.
extern "C" {
void gemv_q4_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
