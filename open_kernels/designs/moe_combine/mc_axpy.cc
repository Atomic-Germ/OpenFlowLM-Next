#include "moe_combine.h"
extern "C" {
void mc_axpy(const float *__restrict rout, const float *__restrict y0, const float *__restrict y1,
             const float *__restrict a0, const float *__restrict a1, const int32_t *__restrict eb,
             float *__restrict o0, float *__restrict o1) {
  mc_axpy_impl(rout, y0, y1, a0, a1, eb, o0, o1);
}
}
