// KVSLICE: the pos0 + L window streams as whole DXL_RB-row elements; pb[3] is how many lie past this group's walk.
#include <stdint.h>

#ifndef DXL_RB
#define DXL_RB 4
#endif

extern "C" {
void dxl_attn_skipw(int32_t *__restrict pb, int32_t rows_after) {
  const int32_t elems = (pb[0] + rows_after + DXL_RB - 1) / DXL_RB;
  pb[3] = elems - pb[4] - pb[6];
}
}
