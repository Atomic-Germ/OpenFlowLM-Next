#include "attn.h"
extern "C" {
void attn_meta(const uint8_t *__restrict m0, const uint8_t *__restrict m1, bfloat16 *__restrict qn,
               bfloat16 *__restrict kn, float *__restrict cs, int32_t *__restrict pb) {
  attn_meta_impl(m0, m1, qn, kn, cs, pb);
}
}
