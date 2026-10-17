// F_MODE 0 bf16, 1 fp32, 2 the LoRA's fp32 z row at F_OFF floats.
#include "dxl_job.h"
#include "dxl_prep4.h"

extern "C" {
void dxl_prep_job(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t tok, const int32_t *__restrict jp) {
  const unsigned K = (unsigned)jp[F_KS];
  if (jp[F_MODE] == 0)
    dxl_prep_tok<false>(e, tab, K, (unsigned)tok, 0, K / 32);
  else
    dxl_prep_tok<true>((const float *)e + (jp[F_MODE] == 2 ? jp[F_OFF] : 0), tab, K, (unsigned)tok, 0, K / 32);
}
}
