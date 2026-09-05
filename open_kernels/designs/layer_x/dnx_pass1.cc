#include "dnx.h"
extern "C" {
void dnx_pass1(const float *__restrict S, float *__restrict ds, int32_t blk) {
  dnx_pass1_slice(S, ds, (unsigned)blk);
}
}
