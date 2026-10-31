#include "attn_pair.h"
extern "C" void attn_pair_step_new(const bfloat16 *K, const bfloat16 *V,
                             const ATTN_QT *q, float *o, float *ml ATTN_H0_PARM) {
  attn_pair_row(K,V,K+kKVH*kHD,V+kKVH*kHD,q,o,ml ATTN_H0_ARG);
}
