// The new position's row as a block of its own (attn.h ATTN_BLOCK_WIN): the window has
// already gone through attn_stepb in whole blocks, so the new row's k'/v' (the core's own
// scratch, as attn_step_new takes them) fill every slot and all but the final one are masked.
// The same block kernel, so the single-row path need not be built at all. A symbol of its own
// because IRON type-checks memref arguments per ExternalFunction and the scratch rows are a
// different memref type from a fifo element -- the reason attn_step_new exists.
#include "attn.h"
#if ATTN_BLOCK_WIN
extern "C" {
void attn_stepb_nr(const bfloat16 *__restrict Kn, const bfloat16 *__restrict Vn, const ATTN_QT *__restrict qs,
                   float *__restrict oacc, float *__restrict ml ATTN_H0_PARM) {
  const bfloat16 *K[kRB];
  const bfloat16 *V[kRB];
  AIE_LOOP_UNROLL_FULL
  for (unsigned r = 0; r < kRB; ++r) {
    K[r] = Kn;
    V[r] = Vn;
  }
#if !ATTN_NULL
  attn_rowb_impl(K, V, qs, oacc, ml, 0u, kRB - 1u ATTN_H0_ARG);
#else
  (void)K; (void)V; (void)qs; (void)oacc; (void)ml;   // the probe covers this path too
#endif
}
}
#endif
