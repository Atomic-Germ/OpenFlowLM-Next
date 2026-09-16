// The short-conv core's single entry point. One fifo element's channels per call.
#include "sc.h"

extern "C" {
void short_conv_step(const float *__restrict Bv, const float *__restrict Cv,
                     const float *__restrict uv, const bfloat16 *__restrict w0,
                     const bfloat16 *__restrict w1, const bfloat16 *__restrict w2,
                     const float *__restrict s0, const float *__restrict s1,
                     bfloat16 *__restrict y, float *__restrict n0, float *__restrict n1) {
  short_conv_elem(Bv, Cv, uv, w0, w1, w2, s0, s1, y, n0, n1);
}
}
