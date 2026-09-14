// down: k-tile ky of a band into a C element
#include "moe_batch.h"
extern "C" {
void mb_step_dn(const uint8_t *__restrict band, const bfloat16 *__restrict ha, float *__restrict c, int ky) {
  mb_step_tile(band, ky, ha, c);
}
}
