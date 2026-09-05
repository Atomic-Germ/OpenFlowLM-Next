// moe_experts: acc[256] = (e == 0 ? 0 : acc) + w[e] * [y0 | y1] -- the routed
// weight applied to a core's two 128-row down bands, accumulated over the 8
// experts in the output element. rw = router floats 256..287 (w[e] at 8 + e);
// e from the core's loop.
#include "vecmath.h"

extern "C" {
void moe_acc(const float *__restrict y0, const float *__restrict y1, const float *__restrict rw,
             float *__restrict acc, int32_t e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float we = rw[8 + e];
  const bfloat16 wh = (bfloat16)we;
  const bfloat16 wl = (bfloat16)(we - (float)wh);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 256; j += 32) {
    const float *yp = (j < 128) ? (y0 + j) : (y1 + (j - 128));
    accf32 a;
    if (e == 0)
      a = aie::zeros<accfloat, 32>();
    else
      a.from_vector(aie::load_v<32>(acc + j));
    a = mac_vs(a, aie::load_v<32>(yp), wh, wl);
    aie::store_v(acc + j, a.template to_vector<float>());
  }
}
}
