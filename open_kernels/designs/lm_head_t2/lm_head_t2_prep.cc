// OPEN-HADAMARD: the input signs are already in the final norm's gain, so only H/32 runs here.
#include "gemv_t2.h"
#include "wht_int.h"

static constexpr unsigned kLmK = LMT2_K;
static_assert(kLmK % 1024 == 0, "the head's input is whole 1024 blocks");

extern "C" {
void lm_head_t2_prep(bfloat16 *__restrict x, uint8_t *__restrict tab) {
#pragma clang loop unroll(disable)
  for (unsigned b = 0; b < kLmK / 1024; ++b)
    xh_int_wht<bfloat16>(x + 1024 * b, tab, kLmK, 8u * b);
}
}
