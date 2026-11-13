// The step for the NEW position, whose k'/v' live in bf16 scratch: never masked
// (a second symbol only because IRON type-checks memref args per ExternalFunction).
#include "attn.h"
extern "C" {
#if ATTN_KVH == 1 && ATTN_KV_SHARED
// a KV-shared layer at one kv head: this position's row comes off the fifo as one element
void attn_step_new(const bfloat16 *__restrict KV, const ATTN_QT *__restrict qs,
                   float *__restrict oacc, float *__restrict ml ATTN_H0_PARM) {
  attn_row_impl(KV, KV + kHD, qs, oacc, ml ATTN_H0_ARG);
}
#else
void attn_step_new(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt, const ATTN_QT *__restrict qs,
                   float *__restrict oacc, float *__restrict ml ATTN_H0_PARM) {
#if ATTN_KVS
  // attn_row_impl indexes this core's window slice; the scratch row is whole, so start it at the slice
  attn_row_impl(Kt + attn_kvb(h0) * kHD, Vt + attn_kvb(h0) * kHD, qs, oacc, ml ATTN_H0_ARG);
#else
  attn_row_impl(Kt, Vt, qs, oacc, ml ATTN_H0_ARG);
#endif
}
#endif
}
