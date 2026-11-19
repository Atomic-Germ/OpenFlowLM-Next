#include "mbl.h"
extern "C" {
// Every gate / up band starts here, before its first dequantised weight is rounded to bf16 (mbl_step_parity's
// to_vector), so the rounding mode is set here and not only in mbl_silu: a core comes up in its default mode
// after a context change, and the first band of that dispatch was rounded in it (xing_batch/xmb_zero_ug.cc;
// open_kernels/UPSTREAM.md, 2026-11-08).
void mbl_zero_ug(float *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  mbl_zero_n(c, MBL_NRB_UP * 64 * MBL_NT);
}
}
