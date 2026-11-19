// The conv core, state row 0: acc = s_0 * w_0 (w the core-local [7][128] taps; bf16 values).
#include "vecmath.h"

extern "C" {
void lg_scmac0(const float *__restrict s, const float *__restrict w, float *__restrict acc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 128; j += V) {
    accN<V> t;
    t.from_vector(aie::load_v<V>(w + j));
    accN<V> a = aie::zeros<accfloat, V>();
    a = mac_vv(a, aie::load_v<V>(s + j), t.template to_vector<bfloat16>());
    aie::store_v(acc + j, a.template to_vector<float>());
  }
}
}
