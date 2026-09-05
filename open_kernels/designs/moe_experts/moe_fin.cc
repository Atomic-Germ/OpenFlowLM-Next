// moe_experts: the block output for a core's 256 rows,
//   acc = xres + acc + sigmoid(xm . sgw) * [y0 | y1]
// where acc holds the routed experts' weighted sum, [y0 | y1] the shared
// expert's down projection and rw[0] the gate (moe_hdr computed it).
#include "vecmath.h"

extern "C" {
void moe_fin(const float *__restrict y0, const float *__restrict y1, const float *__restrict rw,
             const float *__restrict xr, float *__restrict acc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float sg = rw[0];
  const bfloat16 gh = (bfloat16)sg;
  const bfloat16 gl = (bfloat16)(sg - (float)gh);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 256; j += 32) {
    const float *yp = (j < 128) ? (y0 + j) : (y1 + (j - 128));
    accf32 a;
    a.from_vector(fadd32(aie::load_v<32>(xr + j), aie::load_v<32>(acc + j)));
    a = mac_vs(a, aie::load_v<32>(yp), gh, gl);
    aie::store_v(acc + j, a.template to_vector<float>());
  }
}
}
