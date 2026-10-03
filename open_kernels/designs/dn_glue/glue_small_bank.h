#pragma once

// The scalar functions are the existing vecmath.h implementations on the AIE.
// Keep this indexing helper separate so host tests can exercise the same loop.
template <unsigned NHead>
static inline void glue_small_bank(const float *__restrict small,
                                   const float *__restrict acc_a, const float *__restrict acc_b,
                                   float *__restrict decay, float *__restrict beta,
                                   unsigned base, unsigned active) {
  // Each bank reuses acc_a/b[32]; small and the output arrays span all real heads.
  for (unsigned h = 0; h < active; ++h) {
    const unsigned head = base + h;
    decay[head] = sexp(small[head] * ssoftplus(acc_a[h] + small[NHead + head]));
    beta[head] = ssigmoid(acc_b[h]);
  }
}
