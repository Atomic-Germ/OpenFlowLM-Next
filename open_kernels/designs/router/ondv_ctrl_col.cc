#include "ondv_ctrl.h"

// One entry point per .cc: IRON compiles a source once per ExternalFunction, and the design
// notes say every kernel TU holds exactly one entry. ondv_ctrl_col therefore lives here on its
// own -- sharing ondv_ctrl.cc with ondv_ctrl is what produced XAIE_INVALID_ELF with no
// ondv_ctrl_col.o at all (see benchmarks/RESULTS-ondv-fused-layer-descriptor-milestone-*).
//
// rout: an x-stream element holding the router's top-8 at +1024 B (router.h's layout)
// cfg : an x-stream element holding [base_lo, base_hi] at +0
// col : this core's column
// out : 8 slots x 15 words for THAT COLUMN (column-major slice of the full stream)
extern "C" {
void ondv_ctrl_col(const uint8_t *__restrict rout, const uint32_t *__restrict cfg,
                   int32_t col, int32_t *__restrict out) {
  ondv_ctrl_col_impl((const int32_t *)(rout + 1024), cfg[0], cfg[1], (unsigned)col, out);
}
}
