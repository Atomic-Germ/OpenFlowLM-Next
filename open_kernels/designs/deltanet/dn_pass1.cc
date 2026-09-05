// DeltaNet pass 1 entry point (one TU per entry point: IRON compiles the
// source once per ExternalFunction).
#include "dn_step.h"

extern "C" {
void dn_pass1(const float *__restrict S, const float *__restrict vec, float *__restrict t,
              bfloat16 *__restrict k_hl, bfloat16 *__restrict q_hl, int blk) {
  dn_pass1_slice(S, vec, t, k_hl, q_hl, (unsigned)blk);
}
}
