// slot < 8:  acc = (slot == 0 ? 0 : acc) + w[slot] * yd     (the routed weight, rw[8 + slot])
// slot == 8: acc = xres + acc + sigmoid(xm . sgw) * yd        (the shared expert, rw[0]) = the block output
// ms: rw @0, xr @32, acc @288, yd @672. No scalar float ops (soft-float library).
#include "vecmath.h"

extern "C" {
void moe_accfin(float *__restrict ms, int32_t slot) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict rw = ms;
  const float *__restrict xr = ms + 32;
  float *__restrict acc = ms + 288;
  const float *__restrict y = ms + 672;
  const bool shared = slot >= 8;
  v32b wh, wl;
  split32(aie::broadcast<float, 32>(shared ? rw[0] : rw[8 + slot]), wh, wl);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 256; j += 32) {
    accf32 a;
    if (shared)
      a.from_vector(fadd32(aie::load_v<32>(xr + j), aie::load_v<32>(acc + j)));
    else if (slot == 0)
      a = aie::zeros<accfloat, 32>();
    else
      a.from_vector(aie::load_v<32>(acc + j));
    v32b yh, yl;
    split32(aie::load_v<32>(y + j), yh, yl);
    a = aie::mac(a, yh, wh);
    a = aie::mac(a, yh, wl);
    a = aie::mac(a, yl, wh);
    aie::store_v(acc + j, a.template to_vector<float>());
  }
}
}
