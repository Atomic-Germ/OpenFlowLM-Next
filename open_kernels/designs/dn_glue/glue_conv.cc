// cache-bust: 1788363877 (IRON hashes only this file, not included headers)
#include "dn_glue.h"
extern "C" {
// t = loop index within a rolled loop, base = 0 (q/k tiles) or 4 (v tiles).
void glue_conv(const float *__restrict q0, const float *__restrict q1, const bfloat16 *__restrict s0,
               const bfloat16 *__restrict s1, const bfloat16 *__restrict s2,
               const bfloat16 *__restrict w01, const bfloat16 *__restrict w23,
               bfloat16 *__restrict ns0, bfloat16 *__restrict ns1, bfloat16 *__restrict ns2,
               float *__restrict qk, float *__restrict vt, int t, int base) {
  glue_conv_tile(q0, q1, s0, s1, s2, w01, w23, ns0, ns1, ns2, qk, vt, (unsigned)(t + base));
}
}
