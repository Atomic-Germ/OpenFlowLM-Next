// half `half` (32 rows: row groups 2 half, 2 half + 1) of the y accumulator into a 1024 B output element
#include "mbl.h"
extern "C" {
void mbl_copy_y(const float *__restrict y, float *__restrict out, int half) {
  const float *__restrict s = y + half * 32 * MBL_NT;
  for (unsigned i = 0; i < 32 * MBL_NT; i += 32) aie::store_v(out + i, aie::load_v<32>(s + i));
}
}
