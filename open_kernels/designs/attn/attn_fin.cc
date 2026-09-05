#include "attn.h"
extern "C" {
void attn_fin(const float *__restrict oacc, const float *__restrict ml, const float *__restrict g0,
              const float *__restrict g1, bfloat16 *__restrict og, int hp) {
  attn_fin_impl(oacc, ml, g0, g1, og, hp);
}
}
