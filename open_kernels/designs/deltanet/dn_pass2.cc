// DeltaNet pass 2 entry point (one TU per entry point).
#include "dn_step.h"

extern "C" {
void dn_pass2(const float *__restrict S, float *__restrict Sout, const float *__restrict vec,
              float *__restrict t, float *__restrict o, bfloat16 *__restrict k_hl,
              bfloat16 *__restrict q_hl, bfloat16 *__restrict delta_hl, int blk) {
  dn_pass2_slice(S, Sout, vec, t, o, k_hl, q_hl, delta_hl, (unsigned)blk);
}
}
