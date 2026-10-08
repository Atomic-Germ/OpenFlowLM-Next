// Layer RMSNorm with fused residual add (decode, one call):
//   y  = x + add                       (fp32 [N], the new residual)
//   xn = bf16( y * rsqrt(mean(y^2) + eps) * w )
// Elements are LN_N*2 bytes: x, add, y as two fp32 halves; w, xn as bf16[LN_N].
//
// LN_N (the width) and LN_EPS come from the design (designs/ln/ln.py, the whole-layer designs);
// the defaults are the Qwen3.6-27B's. Two entry points:
//   ln_fn  -- everything in one call (three output elements held at once; the 27B designs)
//   ln_y / ln_xn -- the same outputs one element per call, for a width whose five input and
//            three output elements would not fit the norm core's memory together (Llama 3 8B:
//            8 KB elements); y half i is x_i + a_i, xn needs the whole y for its statistics.
//
// LN_GROUPS=G > 1 selects K2's GroupRMSNorm(G): G independent RMS reductions over
// contiguous spans (group g = [g*kGrp, (g+1)*kGrp), mean over kGrp elements each; K2 3.7B
// G=2, 7B G=4), the affine weight stays channel-wise over the whole width. Default 1 keeps
// the single-reduction arithmetic bit-identical to the unpatched kernels.
#pragma once
#include "vecmath.h"

#ifndef LN_N
#define LN_N 2048
#endif
#ifndef LN_EPS
#define LN_EPS 1e-6f
#endif
#ifndef LN_GROUPS
#define LN_GROUPS 1
#endif
static constexpr unsigned kN = LN_N;
static constexpr unsigned kHalf = LN_N / 2;
static constexpr unsigned kV = 32;
static_assert(kHalf % kV == 0, "LN_N must be a multiple of 64");
#if LN_GROUPS > 1
static constexpr unsigned kGrp = LN_N / LN_GROUPS;
static_assert(LN_N % LN_GROUPS == 0 && kGrp % kV == 0, "LN_GROUPS needs each group a multiple of the vector width");
static_assert(kHalf % kGrp == 0, "a group may not straddle the two fp32 halves");
#endif

// rsqrt(mean((x + a)^2) + eps) over both halves
static inline float ln_inv(const float *__restrict x0, const float *__restrict x1, const float *__restrict a0,
                           const float *__restrict a1) {
  accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    const float *ap = (j < kHalf) ? (a0 + j) : (a1 + (j - kHalf));
    const v32f y = fadd32(aie::load_v<kV>(xp), aie::load_v<kV>(ap));
    v32b h, l;
    split32(y, h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  return srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kN) + LN_EPS);
}

#if LN_GROUPS > 1
// K2 GroupRMSNorm(G): one rsqrt(mean(y^2)+eps) per contiguous group, mean over kGrp.
static inline void ln_inv_g(const float *__restrict x0, const float *__restrict x1, const float *__restrict a0,
                            const float *__restrict a1, float *__restrict inv) {
  for (unsigned g = 0; g < LN_GROUPS; ++g) {
    accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
    for (unsigned j = g * kGrp; j < (g + 1) * kGrp; j += kV) {
      const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
      const float *ap = (j < kHalf) ? (a0 + j) : (a1 + (j - kHalf));
      const v32f y = fadd32(aie::load_v<kV>(xp), aie::load_v<kV>(ap));
      v32b h, l;
      split32(y, h, l);
      ss = aie::mac(ss, h, h);
      ss = aie::mac(ss, h, l);
      ss = aie::mac(ss, h, l);
    }
    inv[g] = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kGrp) + LN_EPS);
  }
}
#endif
