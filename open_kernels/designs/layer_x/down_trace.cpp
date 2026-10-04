// Read-only snapshots: segment high/low, then accumulated high/low.
#include "vecmath.h"
extern "C" {
void dense_down_trace(const float *__restrict ms, const float *__restrict ds,
                      const uint8_t *__restrict tab, float *__restrict y,
                      int32_t band, int32_t K, int32_t plane) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {
    if (plane == 1) {
      const auto p = aie::load_v<32>((const float *)(tab + 4*K + K/4 + K/16) + j);
      auto [lo, hi] = aie::interleave_zip(p.template extract<16>(0), p.template extract<16>(1), 1);
      aie::store_v(y+j, aie::concat(lo, hi));
    } else {
      const float *src = plane == 0 ? ms : ds + 64*band + (plane == 3 ? DENSE_DOWN_ROWS : 0);
      aie::store_v(y+j, aie::load_v<32>(src+j));
    }
  }
}
}
