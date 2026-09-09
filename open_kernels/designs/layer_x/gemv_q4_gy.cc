#define GEMV_PER_CALL 1
#include "gemv_q4.h"
// A band into its y element: runtime band law (per_band chunks, row split rs).
// The entry name carries GEMV_Q4_PREFIX: the f32-scale build (-DGEMV_Q4_PREFIX=gemv_q4s32)
// references the suffixed symbol (gemv_q4s32_gy) from its MLIR.
#define GEMV_Q4_WRAP__(PFX, NAME) PFX##_g##NAME
#define GEMV_Q4_WRAP(PFX, NAME) GEMV_Q4_WRAP__(PFX, NAME)
extern "C" {
void GEMV_Q4_WRAP(GEMV_Q4_PREFIX, y)(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                 int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
