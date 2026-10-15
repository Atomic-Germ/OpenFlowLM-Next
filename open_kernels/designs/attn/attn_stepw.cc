// One block of ATTN_RB cached rows from this core's own window element (attn.h ATTN_KVS):
// row r is [K slice | V slice], ATTN_KVS kv heads each, so the pointers are fixed offsets.
#include "attn.h"
#if ATTN_KVS
extern "C" {
void attn_stepw(const bfloat16 *__restrict e, const ATTN_QT *__restrict qs, float *__restrict oacc,
                float *__restrict ml, int32_t *__restrict pb ATTN_H0_PARM) {
  constexpr unsigned kS = ATTN_KVS * kHD;
  const bfloat16 *K[kRB], *V[kRB];
  for (unsigned r = 0; r < kRB; ++r) {
    K[r] = e + 2 * r * kS;
    V[r] = e + (2 * r + 1) * kS;
  }
  pb[2] += (int32_t)kRB;
#if !ATTN_NULL
#ifdef ATTN_REPEAT
  for (int rep = 0; rep < ATTN_REPEAT; ++rep)    // PROBE ONLY: compute the block again, prices pure compute
#endif
  attn_rowb_impl(K, V, qs, oacc, ml ATTN_H0_ARG);
#else
  (void)K; (void)V; (void)qs; (void)oacc; (void)ml;
#endif
}
}
#endif
