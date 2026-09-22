#include "ondv_ctrl.h"

extern "C" {
// One entry point per .cc (IRON compiles the source once per ExternalFunction).
void ondv_ctrl(const int32_t *__restrict idx, uint32_t base_lo, uint32_t base_hi,
               int32_t *__restrict out) {
  ondv_ctrl_impl(idx, base_lo, base_hi, out);
}
}
