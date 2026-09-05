// The MoE header, three 10 KB w-stream elements per core (mode 0, 1, 2):
//   0: [router output f32[1024] | junk]  -> rw = floats 256..287 (w[e] at 8 + e)
//   1: [sgw bf16[2048] | junk]           -> rw[0] = sigmoid(xm . sgw), xm = the act element
//   2: [xres slice f32[256] | junk]      -> xr (this core's 256 residual rows)
#include "vecmath.h"

extern "C" {
void moe_hdr2(const uint8_t *__restrict e, const bfloat16 *__restrict xm, float *__restrict ms, int32_t mode) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *__restrict rw = ms;
  float *__restrict xr = ms + 32;
  if (mode == 0) {
    aie::store_v(rw, aie::load_v<32>((const float *)(e + 1024)));
  } else if (mode == 1) {
    const bfloat16 *__restrict sgw = (const bfloat16 *)e;
    accf32 d = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 2048; j += 32)
      d = aie::mac(d, aie::load_v<32>(xm + j), aie::load_v<32>(sgw + j));
    // sigmoid on a vector lane: no scalar float ops (they pull in the soft-float library)
    const v32f u = aie::broadcast<float, 32>(aie::reduce_add(d.template to_vector<float>()));
    rw[0] = vsigmoidN<32>(u)[0];
  } else {
    const float *__restrict xrs = (const float *)e;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 256; j += 32)
      aie::store_v(xr + j, aie::load_v<32>(xrs + j));
  }
}
}
