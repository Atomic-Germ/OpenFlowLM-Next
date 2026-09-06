// One call, ATTN_RB cached rows. IRON hands each acquired fifo element to the kernel
// as its own argument, so the pointers are gathered back into an array here.
#include "attn.h"
extern "C" {
#if ATTN_RB == 4
void attn_stepb(const bfloat16 *__restrict K0, const bfloat16 *__restrict V0,
                const bfloat16 *__restrict K1, const bfloat16 *__restrict V1,
                const bfloat16 *__restrict K2, const bfloat16 *__restrict V2,
                const bfloat16 *__restrict K3, const bfloat16 *__restrict V3,
                const ATTN_QT *__restrict qs, float *__restrict oacc, float *__restrict ml,
                int32_t *__restrict pb ATTN_H0_PARM) {
  const bfloat16 *K[4] = {K0, K1, K2, K3};
  const bfloat16 *V[4] = {V0, V1, V2, V3};
#elif ATTN_RB == 2
void attn_stepb(const bfloat16 *__restrict K0, const bfloat16 *__restrict V0,
                const bfloat16 *__restrict K1, const bfloat16 *__restrict V1,
                const ATTN_QT *__restrict qs, float *__restrict oacc, float *__restrict ml,
                int32_t *__restrict pb ATTN_H0_PARM) {
  const bfloat16 *K[2] = {K0, K1};
  const bfloat16 *V[2] = {V0, V1};
#else
#error "attn_stepb.cc: ATTN_RB must be 2 or 4"
#endif
  pb[2] += (int32_t)kRB;
  attn_rowb_impl(K, V, qs, oacc, ml ATTN_H0_ARG);
}
}
