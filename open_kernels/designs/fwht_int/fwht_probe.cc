// Repeating over the overwritten element is valid timing because the work is data-independent.
#include "gemv_t2.h"
#if FWHT_KIND >= 2
#include "wht.h"
#else
#include "wht_int.h"
#endif

extern "C" {
void fwht_probe(uint8_t *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i, int32_t reps) {
#pragma clang loop unroll(disable)
  for (int32_t r = 0; r < reps; ++r) {
#if FWHT_KIND == 0
    bfloat16 *x = (bfloat16 *)e;
    xh_int_wht<bfloat16>(x, tab, (unsigned)K, 16u * (unsigned)i);
    xh_int_wht<bfloat16>(x + 1024, tab, (unsigned)K, 16u * (unsigned)i + 8u);
#elif FWHT_KIND == 1
    xh_int_wht<float>((float *)e, tab, (unsigned)K, 8u * (unsigned)i);
#elif FWHT_KIND == 2
    bfloat16 *x = (bfloat16 *)e;
    xh_wht_bf16(x);
    xh_wht_bf16(x + 1024);
    gemv_t2_prep_groups(x, tab, (unsigned)K, 16u * (unsigned)i, 16u);
#else
    bfloat16 *x = (bfloat16 *)e;
    xh_wht_f32((const float *)e, x);
    gemv_t2_prep_groups(x, tab, (unsigned)K, 8u * (unsigned)i, 8u);
#endif
  }
}
}
