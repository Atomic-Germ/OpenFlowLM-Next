#include "kv_attention.h"
extern "C" void kv_step_new(const bfloat16 *K, const bfloat16 *V,
                             const ATTN_QT *q, float *o, float *ml ATTN_H0_PARM) {
  kv_row(K,V,K+kKVH*kHD,V+kKVH*kHD,q,o,ml ATTN_H0_ARG);
}
