// moe_experts: standard-layout (RS=2, 64-row band, K = 512) GEMV band against
// the shared expert's hidden h, for its down projection (32 bands of 4 chunks
// = one element each). `off` is the float offset of the band's 64-float
// accumulator inside the caller's 128-float buffer.
#define GEMV_PER_CALL 4
#define GEMV_PER_BAND 4
#define GEMV_ROWSPLIT 2
#include "gemv_q4.h"

extern "C" {
void gemv_q4_r2h(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                 int32_t group, int32_t off) {
  gemv_q4_pool_group(t, tab, group, y + off);
}
}
