#include "dnx.h"
extern "C" {
void dnx_row(const float *__restrict S, float *__restrict ds, float *__restrict ye, int32_t blk, int32_t j) {
  dnx_row_half(S, ds, ye, (unsigned)blk, (unsigned)(j >> 1), (unsigned)(j & 1));
}
}
