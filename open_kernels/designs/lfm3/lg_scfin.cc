// The conv core, the new token: bx = B * u; conv = acc + bx * w_6; y = bf16(C * conv); the newest state row is bx.
// (designs/short_conv/sc.h's arithmetic, the window [s_0 .. s_5, bx] with the LAST tap on this token.)
#include "vecmath.h"

extern "C" {
void lg_scfin(const float *__restrict Bv, const float *__restrict Cv, const float *__restrict uv,
              const float *__restrict w, const float *__restrict acc, bfloat16 *__restrict y,
              float *__restrict n) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
  const float *__restrict wk = w + 6 * 128;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 128; j += V) {
    const vfN<V> bx = fmulN<V>(aie::load_v<V>(Bv + j), aie::load_v<V>(uv + j));
    accN<V> t;
    t.from_vector(aie::load_v<V>(wk + j));
    accN<V> a;
    a.from_vector(aie::load_v<V>(acc + j));
    a = mac_vv(a, bx, t.template to_vector<bfloat16>());
    const vfN<V> conv = a.template to_vector<float>();
    accN<V> out;
    out.from_vector(fmulN<V>(aie::load_v<V>(Cv + j), conv));
    aie::store_v(y + j, out.template to_vector<bfloat16>());
    aie::store_v(n + j, bx);
  }
}
}
