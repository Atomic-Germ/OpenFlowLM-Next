// lm_head q8 GEMV entry point: one call = kPerCall consecutive pool-order
// chunks of one 128-row band, chunk group `group` (a RUNTIME value passed from
// the core's loop, so one entry point covers every group).
#include "lm_head_q8.h"

extern "C" {
void lm_head_q8_group(const uint8_t *__restrict t, const uint8_t *__restrict tab,
                      float *__restrict y, int group) {
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kPerCall; ++i) {
    const unsigned c = (unsigned)group * kPerCall + i;   // index within the band
    const unsigned quarter = c % kRowSplit;
    const unsigned kt = c / kRowSplit;
    gemv_q8_tile(t + i * kTileBytes, tab, kt, kt == 0, y + quarter * kRows);
  }
}
}
