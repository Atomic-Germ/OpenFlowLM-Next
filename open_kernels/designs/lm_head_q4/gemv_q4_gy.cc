// The lm_head's GEMV entry: a band into its y element with the runtime band law
// (per_band chunks, row split rs). This design's own copy -- designs/layer_x and
// designs/dense generate theirs per spec (gen_kernels.py) and remove them again for a
// spec that does not need them, so a file under layer_x is not there to be included
// from here (2026-09-08: Gemma's lm_head failed to build right after a Qwen3.5 export).
// GEMV_PER_CALL matches lm_head_q4.py's PER_CALL.
#define GEMV_PER_CALL 2
#include "gemv_q4.h"
extern "C" {
void gemv_q4_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
