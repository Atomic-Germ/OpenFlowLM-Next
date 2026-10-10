// F_MODE 0 bf16, 1 fp32, 2 the LoRA's fp32 z row at F_OFF floats.
#include "dxl_job.h"

extern "C" {
void dxl_prep_job(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t tok, const int32_t *__restrict jp) {
  const unsigned K = (unsigned)jp[F_KS];
  uint8_t *t = tab + (unsigned)tok * gemv_q4_tab_bytes(K);
  if (jp[F_MODE] == 0)
    gemv_q4_prep_blocks(e, t, K, 0, K / 32);
  else
    gemv_q4_prep_f32_blocks((const float *)e + (jp[F_MODE] == 2 ? jp[F_OFF] : 0), t, K, 0, K / 32);
}
}
