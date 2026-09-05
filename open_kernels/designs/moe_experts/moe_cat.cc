// moe_experts: hp[64..127] = nb[0..63] -- the odd neighbour's 64 rows of the
// expert hidden, appended to the even core's own 64 (moe_silu wrote hp[0..63])
// to form the pair's 128-row part for the memtile join.
#include "vecmath.h"

extern "C" {
void moe_cat(const bfloat16 *__restrict nb, bfloat16 *__restrict hp) {
  aie::store_v(hp + 64, aie::load_v<32>(nb));
  aie::store_v(hp + 96, aie::load_v<32>(nb + 32));
}
}
