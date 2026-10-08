// The LAST block of the block-only walk (attn.h ATTN_BLOCK_ONLY): ATTN_RB - 1 cached rows
// off the `ain` fifo and, in the final slot, the NEW position's row from the core's own
// k'/v' scratch. A second symbol rather than an argument to attn_stepb because IRON
// type-checks memref arguments per ExternalFunction and the scratch rows are a different
// memref type from a fifo element -- the same reason attn_step_new exists.
//
// The rows between `pos` and the end of this block are padding the host streams to make
// cached + new a whole number of blocks; `nv` is how many of the fifo slots are real, and
// attn_rowb_impl masks the rest with -1e30. pb[0] is the real cached-row count and pb[2]
// the rows already consumed, so nv = pb[0] - pb[2], clamped because an unsigned wrap here
// would index off the end of the score block.
#include "attn.h"
#if ATTN_BLOCK_ONLY
extern "C" {
#if ATTN_RB == 4
void attn_stepb_new(const bfloat16 *__restrict K0, const bfloat16 *__restrict V0,
                    const bfloat16 *__restrict K1, const bfloat16 *__restrict V1,
                    const bfloat16 *__restrict K2, const bfloat16 *__restrict V2,
                    const bfloat16 *__restrict Kn, const bfloat16 *__restrict Vn,
                    const ATTN_QT *__restrict qs, float *__restrict oacc, float *__restrict ml,
                    int32_t *__restrict pb ATTN_H0_PARM) {
  const bfloat16 *K[4] = {K0, K1, K2, Kn};
  const bfloat16 *V[4] = {V0, V1, V2, Vn};
#elif ATTN_RB == 2
void attn_stepb_new(const bfloat16 *__restrict K0, const bfloat16 *__restrict V0,
                    const bfloat16 *__restrict Kn, const bfloat16 *__restrict Vn,
                    const ATTN_QT *__restrict qs, float *__restrict oacc, float *__restrict ml,
                    int32_t *__restrict pb ATTN_H0_PARM) {
  const bfloat16 *K[2] = {K0, Kn};
  const bfloat16 *V[2] = {V0, Vn};
#else
#error "attn_stepb_new.cc: ATTN_RB must be 2 or 4"
#endif
#if !ATTN_NULL
  const int32_t d = pb[0] - pb[2];
  const unsigned nv = d > 0 ? (unsigned)d : 0u;
  attn_rowb_impl(K, V, qs, oacc, ml, nv ATTN_H0_ARG);
#else
  (void)K; (void)V; (void)qs; (void)oacc; (void)ml; (void)pb;   // the probe covers this path too
#endif
}
}
#endif
