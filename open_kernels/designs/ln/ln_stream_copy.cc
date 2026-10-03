#include "ln.h"

extern "C" void ln_stream_copy(const float *__restrict x, float *__restrict saved, int half) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV)
    aie::store_v(saved + half * kHalf + j, aie::load_v<kV>(x + j));
}
