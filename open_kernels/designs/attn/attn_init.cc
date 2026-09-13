#include "attn.h"
extern "C" {
void attn_init(float *__restrict oacc, float *__restrict ml ATTN_SINK_IN_PARM) {
  attn_init_impl(oacc, ml ATTN_SINK_ARG);
}
}
