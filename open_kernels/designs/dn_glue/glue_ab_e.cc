// cache-bust: 1788363880 (IRON hashes only this file, not included headers)
#include "dn_glue.h"
extern "C" {
// One alpha/beta weight tile against ONE 4 KB half of the layer-entry norm output.
//
// glue_ab.cc holds the whole xn in L1 and infers the accumulator reset from `tile == 0`.
// That costs the glue core bf16[HID] of its 64 KB, which at HID 4096 is 2 560 B more than
// the core has (.claude/plans/q-hw-results.md section 1). Here `xn` is a single 4 KB
// element re-streamed per half, `tile` restarts at 0 in every half, and the reset comes in
// as `first` -- set by the host loop for the FIRST half only, so half 1 accumulates onto
// half 0's partial sum.
//
// glue_ab.cc is in the shipped 27B xclbin and is not touched; this TU is used by the dense
// (Qwen3.5) composition only.
void glue_ab_e(const bfloat16 *__restrict W, const bfloat16 *__restrict xn, float *__restrict acc,
               int tile, int first) {
  glue_ab_tile(W, xn, acc, (unsigned)tile, first != 0 && tile == 0);
}
}
