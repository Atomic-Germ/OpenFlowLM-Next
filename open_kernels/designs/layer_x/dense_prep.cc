// Element i of a bf16 activation of K values (2048 per 4 KB element) into the table: blocks
// [64 i, min(64 i + 64, K/32)).
#include "gemv_q4.h"

extern "C" {
void dense_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 64u * (unsigned)i;
  const unsigned nb = (b0 + 64u <= total) ? 64u : total - b0;
  gemv_q4_prep_blocks(e, tab, (unsigned)K, b0, nb);
}
}
