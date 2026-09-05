// Copy ssm_norm weight (bf16[128], first 256 B of a 4 KB element) to scratch.
#include "vecmath.h"
extern "C" {
void post_copy_nw(const bfloat16 *__restrict src, bfloat16 *__restrict dst) {
  for (unsigned j = 0; j < 128; j += 32)
    aie::store_v(dst + j, aie::load_v<32>(src + j));
}
}
