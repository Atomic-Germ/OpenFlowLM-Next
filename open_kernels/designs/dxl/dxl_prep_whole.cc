// The head's K = hidden fits whole, so its per-token tables are not K-sliced.
#include "dxl_gemv.h"
#include "dxl_prep4.h"

extern "C" {
void dxl_prep_whole(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t j, int32_t i, int32_t k) {
  const unsigned K = (unsigned)k, b0 = 32u * (unsigned)i, nb = K / 32 - b0 < 32u ? K / 32 - b0 : 32u;
  dxl_prep_tok<false>(e, tab, K, (unsigned)j, b0, nb);   // a last partial element: its real blocks
}
}
