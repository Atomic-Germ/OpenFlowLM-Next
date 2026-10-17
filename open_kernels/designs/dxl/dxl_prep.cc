// Token j's KS-wide bf16 slice into the slice table group (dxl_prep4.h).
#include "dxl_gemv.h"
#include "dxl_prep4.h"

extern "C" {
void dxl_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t j, int32_t ks) {
  const unsigned K = (unsigned)ks;
  dxl_prep_tok<false>(e, tab, K, (unsigned)j, 0, K / 32);
}
}
