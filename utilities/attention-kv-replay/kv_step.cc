#include "kv_attention.h"
extern "C" void kv_step(const bfloat16 *K, const bfloat16 *Kl,
                         const bfloat16 *V, const bfloat16 *Vl,
                         const ATTN_QT *q, float *o, float *ml, int32_t *pb ATTN_H0_PARM) {
  const int t=pb[2]++;
  if(t<pb[0]) kv_row(K,V,Kl,Vl,q,o,ml ATTN_H0_ARG);
}
