// Token j's KS-wide fp32 slice (the fifo types it as bf16) into slice table j, rounded to bf16 first.
#include "dxl_gemv.h"

extern "C" {
void dxl_prep_f32(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t j, int32_t ks) {
  const unsigned K = (unsigned)ks;
  gemv_q4_prep_f32_blocks((const float *)e, tab + (unsigned)j * gemv_q4_tab_bytes(K), K, 0, K / 32);
}
}
