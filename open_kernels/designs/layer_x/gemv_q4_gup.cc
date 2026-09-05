#define GEMV_PER_CALL 2
#include "gemv_q4.h"
// MoE up (band 0) / gate (band 1): 64-row K=2048 bands (the routed stripe halves through the
// strided tap, the shared expert's band as it lies) into ms[544 + 64 band ..].
extern "C" {
void gemv_q4_gup(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                 int32_t group, int32_t band) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + 544 + 64 * band, 16, 2);
}
}
