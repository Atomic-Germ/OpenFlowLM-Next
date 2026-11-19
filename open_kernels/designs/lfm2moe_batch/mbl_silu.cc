// h band `band` = silu(g) * u, into one 1024 B output element
#include "mbl.h"
extern "C" {
void mbl_silu(const float *__restrict u, const float *__restrict g, float *__restrict h, int band) {
  mbl_silu_band(u + band * 64 * MBL_NT, g + band * 64 * MBL_NT, reinterpret_cast<bfloat16 *>(h));
}
}
