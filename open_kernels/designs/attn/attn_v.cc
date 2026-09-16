#include "attn.h"
extern "C" {
void attn_v(const float *__restrict vh ATTN_BIAS_PARM, bfloat16 *__restrict vout, int h) {
  attn_v_impl(vh ATTN_BIAS_ARG, vout, h);
}
}
