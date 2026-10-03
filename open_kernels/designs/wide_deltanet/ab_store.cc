#include "dn_glue.h"
#include "glue_small_bank.h"

extern "C" void wide_ab_store(const float *__restrict small,
                               const float *__restrict a, const float *__restrict b,
                               float *__restrict out, int base, int active) {
#if WIDE_AB_CARRY
  float aa[32], bb[32];
  for (int h = 0; h < active; ++h) {
    aa[h] = a[h] + a[32 + h];
    bb[h] = b[h] + b[32 + h];
  }
  a = aa;
  b = bb;
#endif
  glue_small_bank<kNHead>(small, a, b, out + 2 * kNHead, out + 3 * kNHead,
                          static_cast<unsigned>(base), static_cast<unsigned>(active));
  for (int h = 0; h < active; ++h) {
    out[base + h] = a[h];
    out[kNHead + base + h] = b[h];
  }
}
