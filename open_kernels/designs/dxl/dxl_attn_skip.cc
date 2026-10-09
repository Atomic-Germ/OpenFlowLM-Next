// The window streams pos0 + L rows; a group walks pb[1] of them (decode's count for its own position, one dummy at 0).
#include <stdint.h>

extern "C" {
void dxl_attn_skip(int32_t *__restrict pb, int32_t rows_after) {
  pb[6] = pb[0] + rows_after - pb[1];
}
}
