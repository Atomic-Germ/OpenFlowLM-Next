// One entry point per TU (IRON compiles a source once per ExternalFunction); the math is ln.h.
#include "ln.h"

extern "C" {
// xn = bf16((x + a) * inv * w) (one output element; the statistics over both halves)
void ln_xn(const float *__restrict x0, const float *__restrict x1, const float *__restrict a0,
           const float *__restrict a1, const bfloat16 *__restrict w, bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#if LN_GROUPS > 1
  float inv[LN_GROUPS];
  ln_inv_g(x0, x1, a0, a1, inv);
  for (unsigned g = 0; g < LN_GROUPS; ++g) {
#pragma clang loop unroll(disable)
    for (unsigned j = g * kGrp; j < (g + 1) * kGrp; j += kV) {
      const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
      const float *ap = (j < kHalf) ? (a0 + j) : (a1 + (j - kHalf));
      const v32f y = fadd32(aie::load_v<kV>(xp), aie::load_v<kV>(ap));
      // (y * w) as fp32, then * inv as fp32. The two-term split (`mac_vv`/`mac_vs`)
      // reaches ~16 mantissa bits, a relative 2e-5 on a 5120-wide row applied to EVERY
      // element the layer goes on to use. `mul3_32` is exact to fp32 here.
      const v32f t = mul3bv_32(y, aie::load_v<kV>(w + j));
      const v32f o = mul3_32(t, aie::broadcast<float, kV>(inv));
      aie::store_v(xn + j, o.template to_vector<bfloat16>());
    }
  }
#else
  const float inv = ln_inv(x0, x1, a0, a1);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    const float *ap = (j < kHalf) ? (a0 + j) : (a1 + (j - kHalf));
    const v32f y = fadd32(aie::load_v<kV>(xp), aie::load_v<kV>(ap));
    // full-precision (y * w) * inv: the two-term split is ~2e-5 relative, see mul3_32
    const v32f t = mul3bv_32(y, aie::load_v<kV>(w + j));
    const v32f o = mul3_32(t, aie::broadcast<float, kV>(inv));
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
#endif
}
}
