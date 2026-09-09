// The step for the NEW position, whose k'/v' live in bf16 scratch: never masked
// (a second symbol only because IRON type-checks memref args per ExternalFunction).
#include "attn.h"
extern "C" {
void attn_step_new(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt, const ATTN_QT *__restrict qs,
                   float *__restrict oacc, float *__restrict ml ATTN_H0_PARM) {
  attn_row_impl(Kt, Vt, qs, oacc, ml ATTN_H0_ARG);
}
}
