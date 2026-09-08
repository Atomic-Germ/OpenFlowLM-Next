#define GEMV_PER_CALL 2
#include "gemv_q4.h"
// MoE down, element j of 8, into ms[672 ..] (the core's 256 rows) against h's table at tab + 4608:
// routed slots: 2 128-row RS=4 bands of 4 elements; the shared expert: 4 64-row RS=2 bands of 2.
extern "C" {
void gemv_q4_gdown(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                   int32_t j, int32_t slot) {
  if (slot < 8)
    gemv_q4_pool_group_rt(t, tab + 4608, (unsigned)(j % 4), ms + 672 + 128 * (j / 4), 8, 4);
  else
    gemv_q4_pool_group_rt(t, tab + 4608, (unsigned)(j % 2), ms + 672 + 64 * (j / 2), 4, 2);
}
}
