// The routing-weight record (f32[16], the host's hy_route) from xm's last x element, byte 0.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void lm_wgrab(const bfloat16 *__restrict xe, float *__restrict wv) {
  aie::store_v(wv, aie::load_v<16>((const float *)((const uint8_t *)xe + 0)));
}
}
