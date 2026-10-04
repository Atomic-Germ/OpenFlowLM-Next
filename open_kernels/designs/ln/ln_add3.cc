// One entry point per TU (IRON compiles a source once per ExternalFunction); the math is ln.h.
#include "ln.h"

extern "C" {
// y = x + (a + b) over ONE half: the layer's closing residual when the dense FFN's down GEMV
// ran in two K pieces (recipes/qwen36moe.py down_split) -- a and b are the pieces' partial
// outputs, whose sum is the down projection, and x the residual after attention.
void ln_add3(const float *__restrict x, const float *__restrict a, const float *__restrict b,
             float *__restrict y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV)
    aie::store_v(y + j, fadd32(aie::load_v<kV>(x + j), fadd32(aie::load_v<kV>(a + j), aie::load_v<kV>(b + j))));
}
}
