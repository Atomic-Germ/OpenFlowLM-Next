#include "attn.h"
extern "C" {
void attn_q(const float *__restrict qh, const bfloat16 *__restrict qn, const float *__restrict cs,
            float *__restrict qs, int h) {
  attn_q_impl(qh, qn, cs, qs, h);
}
}
