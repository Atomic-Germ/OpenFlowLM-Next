// moe_experts: the first element of every core's weight stream is the header
//   [xm bf16[2048] | router output f32[1024] | sgw bf16[2048] | xres f32[2048]]
// (20480 B: the ln, router, pack and residual buffers back to back). A core has
// only 2 input DMA channels (w and h), so everything else rides the w stream;
// copied out because release() frees the oldest element of the fifo. (xm itself
// goes straight into the GEMV table: gemv_q4_prep_k2048 on the same element.)
//   rw <- router floats 256..287 (w[e] = rw[8 + e]); rw[0] <- sigmoid(xm . sgw),
//         the shared-expert gate (idx[0] was there, unused on the core)
//   xr <- this core's 256 rows of xres (core c)
// One entry point per TU.
#include "vecmath.h"

extern "C" {
void moe_hdr(const uint8_t *__restrict e, float *__restrict rw, float *__restrict xr, int32_t c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *__restrict xs = (const bfloat16 *)e;
  const float *__restrict rs = (const float *)(e + 4096 + 1024);
  const bfloat16 *__restrict sgw = (const bfloat16 *)(e + 8192);
  const float *__restrict xrs = (const float *)(e + 12288) + c * 256;
  accf32 d = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 2048; j += 32) {
    d = aie::mac(d, aie::load_v<32>(xs + j), aie::load_v<32>(sgw + j));
  }
  aie::store_v(rw, aie::load_v<32>(rs));
  rw[0] = ssigmoid(aie::reduce_add(d.template to_vector<float>()));
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 256; j += 32)
    aie::store_v(xr + j, aie::load_v<32>(xrs + j));
}
}
