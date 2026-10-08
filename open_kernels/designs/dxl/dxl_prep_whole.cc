// Element i (1024 bf16) of token j's K-wide activation into the whole-K table j: blocks
// [32 i, 32 i + 32) of a table gemv_q4_tab_bytes(K) long (the head's K = hidden fits whole).
#include "dxl_gemv.h"

extern "C" {
void dxl_prep_whole(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t j, int32_t i, int32_t k) {
  const unsigned K = (unsigned)k;
  gemv_q4_prep_blocks(e, tab + (unsigned)j * gemv_q4_tab_bytes(K), K, 32u * (unsigned)i, 32u);
}
}
