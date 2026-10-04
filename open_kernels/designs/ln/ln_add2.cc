// One entry point per TU (IRON compiles a source once per ExternalFunction); the math is ln.h.
#include "ln.h"

extern "C" {
// y = x + a over ONE half (kHalf f32, an LN_N*2-byte element): the split norm helper's
// residual add (recipes/qwen36moe.py norm_split). ln_y computes the same half but needs all
// four x / a halves held at once; this needs one of each, so a width whose five inputs and
// one output do not fit the norm core (the 27B's 10 KB elements) streams them. The sum is
// fadd32(x, a) exactly as ln_y / ln_xn form it, so ln_nr over it reproduces ln_xn bit for bit.
void ln_add2(const float *__restrict x, const float *__restrict a, float *__restrict y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) aie::store_v(y + j, fadd32(aie::load_v<kV>(x + j), aie::load_v<kV>(a + j)));
}
}
