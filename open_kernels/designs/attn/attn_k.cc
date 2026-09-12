#include "attn.h"
extern "C" {
void attn_k(const float *__restrict kh ATTN_BIAS_PARM, const bfloat16 *__restrict kn,
            const float *__restrict cs, float *__restrict tmp, bfloat16 *__restrict kout, int h) {
  attn_k_impl(kh ATTN_BIAS_ARG, kn, cs, tmp, kout, h);
}
}
