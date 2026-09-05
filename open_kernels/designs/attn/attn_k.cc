#include "attn.h"
extern "C" {
void attn_k(const float *__restrict kh, const bfloat16 *__restrict kn, const float *__restrict cs,
            float *__restrict tmp, bfloat16 *__restrict kout, int h) {
  attn_k_impl(kh, kn, cs, tmp, kout, h);
}
}
