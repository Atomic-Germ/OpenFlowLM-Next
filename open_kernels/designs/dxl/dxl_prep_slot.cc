// dxl_prep_job into slice s's table (dxl_tab_off).
#include "dxl_job.h"
#include "dxl_prep4.h"

extern "C" {
void dxl_prep_slot(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t tok, int32_t s, const int32_t *__restrict jp) {
  const unsigned K = (unsigned)jp[F_KS];
  uint8_t *t = tab + dxl_tab_off(jp, (unsigned)s);
  if (jp[F_MODE] == 0)
    dxl_prep_tok<false>(e, t, K, (unsigned)tok, 0, K / 32);
  else
    dxl_prep_tok<true>((const float *)e + (jp[F_MODE] == 2 ? jp[F_OFF] : 0), t, K, (unsigned)tok, 0, K / 32);
}
}
