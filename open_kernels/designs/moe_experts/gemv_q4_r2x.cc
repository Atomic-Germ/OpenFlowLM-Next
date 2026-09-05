// moe_experts: standard-layout (RS=2, 64-row band) GEMV group against xm, for
// the shared expert's up/gate stripes (8 bands of 16 chunks). `group` is the
// 4-chunk group within the band, `off` the float offset of the band's 64-float
// accumulator inside the caller's 128-float buffer.
#define GEMV_PER_CALL 4
#define GEMV_PER_BAND 16
#define GEMV_ROWSPLIT 2
#include "gemv_q4.h"

extern "C" {
void gemv_q4_r2x(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                 int32_t group, int32_t off) {
  gemv_q4_pool_group(t, tab, group, y + off);
}
}
