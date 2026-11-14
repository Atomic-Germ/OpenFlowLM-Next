#include "gemv_tab.h"
extern "C" void probe(const float* x, uint8_t* table) {
  gemv_q4_prep_f32_direct_blocks(x, table, 1024, 0, 32);
}
