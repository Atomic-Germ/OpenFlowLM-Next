// Token j's KS-wide bf16 slice into slice table j (tables gemv_q4_tab_bytes(KS) apart).
#include "dxl_gemv.h"

extern "C" {
void dxl_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t j, int32_t ks) {
  const unsigned K = (unsigned)ks;
  gemv_q4_prep_blocks(e, tab + (unsigned)j * gemv_q4_tab_bytes(K), K, 0, K / 32);
}
}
