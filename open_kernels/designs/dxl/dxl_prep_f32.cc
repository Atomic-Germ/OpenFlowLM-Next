// Token j's KS-wide fp32 slice (the fifo types it as bf16) into the slice table group, rounded to bf16 first.
#include "dxl_gemv.h"
#include "dxl_prep4.h"

extern "C" {
void dxl_prep_f32(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t j, int32_t ks) {
  const unsigned K = (unsigned)ks;
  dxl_prep_tok<true>(e, tab, K, (unsigned)j, 0, K / 32);
}
}
