#include "attn.h"
extern "C" {
void attn_step(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt, const float *__restrict qs,
               float *__restrict oacc, float *__restrict ml, int32_t *__restrict pb) {
  attn_step_impl(Kt, Vt, qs, oacc, ml, pb);
}
}
