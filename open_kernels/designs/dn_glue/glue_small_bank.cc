#include "dn_glue.h"
#include "glue_small_bank.h"

extern "C" void glue_small_bank_fn(const float *__restrict small,
                                    const float *__restrict acc_a, const float *__restrict acc_b,
                                    float *__restrict decay, float *__restrict beta,
                                    int base, int active) {
  glue_small_bank<kNHead>(small, acc_a, acc_b, decay, beta,
                         static_cast<unsigned>(base), static_cast<unsigned>(active));
}
