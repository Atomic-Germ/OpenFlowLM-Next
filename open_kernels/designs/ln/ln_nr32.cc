// Layer RMSNorm without a residual add, emitting fp32 half i (Gemma's post-attention / post-FFN
// norms: t = norm(out) * w, added to the residual by the next ln call):
//   y_i = (x * rsqrt(mean(x^2) + eps) * w)[half i]
// Elements are LN_N*2 bytes: x as two fp32 halves, w bf16[LN_N], y_i one fp32 half. One entry per TU.
#include "ln.h"

extern "C" {
void ln_nr32(const float *__restrict x0, const float *__restrict x1, const bfloat16 *__restrict w,
             float *__restrict y, int32_t i) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    v32b h, l;
    split32(aie::load_v<kV>(xp), h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kN) + LN_EPS);
  const float *x = i ? x1 : x0;
  const bfloat16 *wh = w + (i ? kHalf : 0);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) {
    // (x * w) * inv at full fp32: `mac_vv`/`mac_vs` reach only ~16 mantissa bits,
    // a relative 2e-5 that this path applies to every element of the layer.
    const v32f t = mul3bv_32(aie::load_v<kV>(x + j), aie::load_v<kV>(wh + j));
    const v32f o = mul3_32(t, aie::broadcast<float, kV>(inv));
    aie::store_v(y + j, o);
  }
}
}
