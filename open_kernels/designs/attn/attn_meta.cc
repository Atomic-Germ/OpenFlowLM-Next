#include "attn.h"
extern "C" {
void attn_meta(const uint8_t *__restrict m0, const uint8_t *__restrict m1 ATTN_PTAB2_PARM,
               bfloat16 *__restrict qn, bfloat16 *__restrict kn, float *__restrict cs,
               int32_t *__restrict pb ATTN_SINK_OUT_PARM ATTN_SINK_H0_PARM) {
  attn_meta_impl(m0, m1 ATTN_PTAB2_ARG, qn, kn, cs, pb ATTN_SINK_ARG ATTN_SINK_H0_ARG);
}
}
