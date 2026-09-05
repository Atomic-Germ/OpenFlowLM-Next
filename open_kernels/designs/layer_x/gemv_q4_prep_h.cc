// the expert hidden h (f32[512], assembled in DDR from the cores' parts) -> bf16 -> its table
// at tab + 4608 (past xm's K=2048 table).
#include "gemv_q4.h"

extern "C" {
void gemv_q4_prep_h(const float *__restrict hf, uint8_t *__restrict tab) {
  gemv_q4_prep_f32(hf, tab + 4608, 512);
}
}
