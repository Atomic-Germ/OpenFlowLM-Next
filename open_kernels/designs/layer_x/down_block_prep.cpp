#include "gemv_tab.h"
extern "C" void down_block_prep(const float *x, uint8_t *tab, int32_t element) {
  gemv_q4_prep_f32_blocks(x, tab, 4096, element*32, 32);
}
