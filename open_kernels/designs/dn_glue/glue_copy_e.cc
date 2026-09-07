// cache-bust: 1788363879 (IRON hashes only this file, not included headers)
#include "dn_glue.h"
extern "C" {
// Element i of the layer-entry xn (bf16[HID]) into the glue core's scratch, so the fifo
// element can be released (release(n) frees the n OLDEST acquired elements, so an element
// cannot be held across later acquire/release pairs on the same fifo).
//
// The x / side stream's element is 4 KB = 2048 bf16 whatever HID is, so a HID wider than
// 2048 arrives as several of them. glue_copy.cc's single-element form is what the Qwen3.6
// MoE designs use (their HID is exactly one element) and is left untouched; the Qwen3.5
// dense composition at HID 4096 uses this one.
void glue_copy_xn_e(const bfloat16 *__restrict src, bfloat16 *__restrict dst, int i) {
  bfloat16 *__restrict d = dst + (unsigned)i * 2048u;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 2048u; j += kV)
    aie::store_v(d + j, aie::load_v<kV>(src + j));
}
}
