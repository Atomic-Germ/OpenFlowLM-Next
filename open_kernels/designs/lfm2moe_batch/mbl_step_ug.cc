// up / gate: k-tile ky of a band into band `half` of the core's u / g buffer
#include "mbl.h"
extern "C" {
void mbl_step_ug(const uint8_t *__restrict band, const bfloat16 *__restrict xa, float *__restrict c, int ky, int half) {
  mbl_step_tile(band, ky, xa, c + half * 64 * MBL_NT);
}
}
