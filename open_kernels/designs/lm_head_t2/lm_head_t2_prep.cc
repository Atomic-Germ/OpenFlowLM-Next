// The rotated-basis ternary lm head's activation prep (OPEN-HADAMARD, OPEN-QUANT-T2): hn
// through H/32 per 1024 block in place (wht.h -- the input signs are already in the final
// norm's gain), then gemv_t2's per-128 table. One call per token per core.
#include "gemv_t2.h"
#include "wht.h"

static constexpr unsigned kLmK = LMT2_K;
static_assert(kLmK % 1024 == 0, "the head's input is whole 1024 blocks");

extern "C" {
void lm_head_t2_prep(bfloat16 *__restrict x, uint8_t *__restrict tab) {
#pragma clang loop unroll(disable)
  for (unsigned b = 0; b < kLmK / 1024; ++b)
    xh_wht_bf16(x + 1024 * b);
  gemv_t2_prep_groups(x, tab, kLmK, 0u, kLmK / 128);
}
}
