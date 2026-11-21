#include "attn.h"
extern "C" {
#if ATTN_KVH == 1
// one kv head: a cached row is ONE fifo element, [K_t | V_t]
void attn_step(const bfloat16 *__restrict KV, const ATTN_QT *__restrict qs,
               float *__restrict oacc, float *__restrict ml, int32_t *__restrict pb ATTN_H0_PARM) {
  attn_step_impl(KV, KV + kHD, qs, oacc, ml, pb ATTN_H0_ARG);
}
#else
void attn_step(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt, const ATTN_QT *__restrict qs,
               float *__restrict oacc, float *__restrict ml, int32_t *__restrict pb ATTN_H0_PARM) {
  attn_step_impl(Kt, Vt, qs, oacc, ml, pb ATTN_H0_ARG);
}
#endif
}
