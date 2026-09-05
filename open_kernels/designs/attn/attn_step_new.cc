// The step for the NEW position, whose k'/v' live in bf16 scratch: never masked
// (a second symbol only because IRON type-checks memref args per ExternalFunction).
#include "attn.h"
extern "C" {
void attn_step_new(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt, const float *__restrict qs,
                   float *__restrict oacc, float *__restrict ml) {
  attn_row_impl(Kt, Vt, qs, oacc, ml);
}
}
