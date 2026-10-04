// The rotated-basis ternary lm head (OPEN-QUANT-T2, OPEN-HADAMARD): Ternary Bonsai 2's
// output.weight as PrismML stores it -- ternary, after the blockwise Walsh-Hadamard rotation --
// so logits = W' @ (H(hn) / 32) per 1024 block. The s_hidden signs are in the final norm's gain
// (q4nx-build folds them there), so hn arrives already sign-flipped.
//
// lm_head_t2_group: one w element = LMT2_PER_CALL consecutive pool-order t2 chunks of one
// 64-row band (gemv_q4's band law, rs 2: half = c % 2, k-tile = c / 2). The prep is
// lm_head_t2_prep.cc. LMT2_K, LMT2_PER_CALL and GEMV_T2_CHUNK come from lm_head_t2.py's flags.
#ifndef LMT2_PER_CALL
#error "LMT2_PER_CALL is set by lm_head_t2.py"
#endif
#define GEMV_PER_CALL LMT2_PER_CALL
#include "gemv_t2.h"

static constexpr unsigned kLmK = LMT2_K;
static constexpr unsigned kLmPerBand = 2 * kLmK / kT2TileK;   // chunks per 64-row band
static_assert(kLmPerBand % kT2PerCall == 0, "a w element never straddles two bands");

extern "C" {
void lm_head_t2_group(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                      int32_t group) {
#pragma clang loop unroll(disable)
  for (unsigned sub = 0; sub < kT2PerCall; ++sub)
    gemv_t2_chunk_rt(t, tab, (unsigned)group, kLmPerBand | (sub << 16), y, 2u);
}
}
