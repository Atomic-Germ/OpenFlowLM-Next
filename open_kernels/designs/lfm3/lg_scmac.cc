// The conv core, state row k >= 1: acc += s_k * w_k, and the new state row k - 1 is this one (n = s_k).
#include "vecmath.h"

extern "C" {
void lg_scmac(const float *__restrict s, const float *__restrict w, float *__restrict acc,
              float *__restrict n, int32_t k) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned V = 32;
  const float *__restrict wk = w + k * 128;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 128; j += V) {
    accN<V> t;
    t.from_vector(aie::load_v<V>(wk + j));
    accN<V> a;
    a.from_vector(aie::load_v<V>(acc + j));
    a = mac_vv(a, aie::load_v<V>(s + j), t.template to_vector<bfloat16>());
    aie::store_v(acc + j, a.template to_vector<float>());
    aie::store_v(n + j, aie::load_v<V>(s + j));
  }
}
}
