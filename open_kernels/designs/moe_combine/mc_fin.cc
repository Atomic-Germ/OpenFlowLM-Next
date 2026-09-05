#include "moe_combine.h"
extern "C" {
void mc_fin(const float *__restrict c0, const float *__restrict c1, const float *__restrict x0,
            const float *__restrict x1, const float *__restrict s0, const float *__restrict s1,
            const bfloat16 *__restrict xm, const bfloat16 *__restrict sgw, float *__restrict o0,
            float *__restrict o1) {
  mc_fin_impl(c0, c1, x0, x1, s0, s1, xm, sgw, o0, o1);
}
}
