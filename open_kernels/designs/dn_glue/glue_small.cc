// cache-bust: 1788363878 (IRON hashes only this file, not included headers)
#include "dn_glue.h"
extern "C" {
void glue_small_fn(const float *__restrict small, const float *__restrict acc_a,
                   const float *__restrict acc_b, float *__restrict decay, float *__restrict beta) {
  glue_small(small, acc_a, acc_b, decay, beta);
}
}
