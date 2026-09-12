#include "attn.h"
extern "C" {
void attn_q(const float *__restrict qh ATTN_BIAS_PARM, const bfloat16 *__restrict qn,
            const float *__restrict cs, ATTN_QT *__restrict qs, int h) {
  attn_q_impl(qh ATTN_BIAS_ARG, qn, cs, qs, h);
}
}
