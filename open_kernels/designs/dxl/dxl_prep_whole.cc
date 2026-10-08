// The head's K = hidden fits whole, so its per-token tables are not K-sliced.
#include "dxl_gemv.h"

extern "C" {
void dxl_prep_whole(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t j, int32_t i, int32_t k) {
  const unsigned K = (unsigned)k;
  gemv_q4_prep_blocks(e, tab + (unsigned)j * gemv_q4_tab_bytes(K), K, 32u * (unsigned)i, 32u);
}
}
