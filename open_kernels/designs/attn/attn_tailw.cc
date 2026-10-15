// The pb[5] rows past the last whole block, from the padded window element, one at a time as attn_step takes them.
#include "attn.h"
#if ATTN_KVS
extern "C" {
void attn_tailw(const bfloat16 *__restrict e, const ATTN_QT *__restrict qs, float *__restrict oacc,
                float *__restrict ml, int32_t *__restrict pb ATTN_H0_PARM) {
  constexpr unsigned kS = ATTN_KVS * kHD;
  const unsigned n = (unsigned)pb[5];
  for (unsigned r = 0; r < n; ++r) attn_step_impl(e + 2 * r * kS, e + (2 * r + 1) * kS, qs, oacc, ml, pb ATTN_H0_ARG);
}
}
#endif
