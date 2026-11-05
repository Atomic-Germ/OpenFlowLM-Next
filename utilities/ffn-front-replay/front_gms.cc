#include "gemv_q4.h"
#include "vecmath.h"
extern "C" {
void front_gms(const uint8_t *w, const uint8_t *tab, float *ms, int32_t group, int32_t per_band, int32_t dst) {
  gemv_q4_pool_group_rt(w, tab, group, ms + dst, per_band, 2);
}

}
