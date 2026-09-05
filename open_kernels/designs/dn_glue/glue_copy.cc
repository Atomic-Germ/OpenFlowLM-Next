// cache-bust: 1788363878 (IRON hashes only this file, not included headers)
#include "dn_glue.h"
extern "C" {
// Copy the xn element (bf16[2048]) into L1 scratch so the fifo element can be
// released: release(n) frees the n OLDEST acquired elements, so an element
// cannot be held across later acquire/release pairs on the same fifo.
void glue_copy_xn(const bfloat16 *__restrict src, bfloat16 *__restrict dst) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHid; j += kV)
    aie::store_v(dst + j, aie::load_v<kV>(src + j));
}
}
