#include "dn_glue.h"
#include "glue_small_bank.h"

extern "C" void wide_ab_store(const float *__restrict small,
                               const float *__restrict a, const float *__restrict b,
                               float *__restrict out, int base, int active) {
  glue_small_bank<kNHead>(small, a, b, out + 2 * kNHead, out + 3 * kNHead,
                          static_cast<unsigned>(base), static_cast<unsigned>(active));
  for (int h = 0; h < active; ++h) {
    out[base + h] = a[h];
    out[kNHead + base + h] = b[h];
  }
}
