#include "dn_glue.h"

// The separate AB dispatch emits [alpha | beta logits | decay | beta].
// This is a copy only; do not recompute or quantize the nonlinearities.
extern "C" void wide_glue_load_ab(const float *__restrict ab,
                                  float *__restrict decay, float *__restrict beta) {
  for (unsigned h = 0; h < kNHead; ++h) {
    decay[h] = ab[2 * kNHead + h];
    beta[h] = ab[3 * kNHead + h];
  }
}
