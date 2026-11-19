// The conv core: LFM2's short depthwise causal conv for one 512-channel element (designs/short_conv/sc.h's
// arithmetic), the taps as f32 tap-major rows (the GGUF's F32 weight; bf16 values, so the conversion below is
// exact and the MACs are sc.h's mac_vv bit for bit):
//     bx = B * u;  conv = s0 * w0 + s1 * w1 + bx * w2  (window [s0, s1, bx], newest LAST);  y = bf16(C * conv)
//     n0 = s1;  n1 = bx                                 (the window slides; the state stays f32)
#include "vecmath.h"

extern "C" {
void lm_sc(const float *__restrict Bv, const float *__restrict Cv, const float *__restrict uv,
           const float *__restrict w0, const float *__restrict w1, const float *__restrict w2,
           const float *__restrict s0, const float *__restrict s1, bfloat16 *__restrict y,
           float *__restrict n0, float *__restrict n1) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 512; j += V) {
    accN<V> t0, t1, t2;
    t0.from_vector(aie::load_v<V>(w0 + j));
    t1.from_vector(aie::load_v<V>(w1 + j));
    t2.from_vector(aie::load_v<V>(w2 + j));
    const vfN<V> bx = fmulN<V>(aie::load_v<V>(Bv + j), aie::load_v<V>(uv + j));
    accN<V> acc = aie::zeros<accfloat, V>();
    acc = mac_vv(acc, aie::load_v<V>(s0 + j), t0.template to_vector<bfloat16>());
    acc = mac_vv(acc, aie::load_v<V>(s1 + j), t1.template to_vector<bfloat16>());
    acc = mac_vv(acc, bx, t2.template to_vector<bfloat16>());
    const vfN<V> conv = acc.template to_vector<float>();
    accN<V> out;
    out.from_vector(fmulN<V>(aie::load_v<V>(Cv + j), conv));
    aie::store_v(y + j, out.template to_vector<bfloat16>());
    aie::store_v(n0 + j, aie::load_v<V>(s1 + j));
    aie::store_v(n1 + j, bx);
  }
}
}
