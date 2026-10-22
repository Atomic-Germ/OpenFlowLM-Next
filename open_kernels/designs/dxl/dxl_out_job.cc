// The silu branch is dense_act's arithmetic per token, so each row's h stays bit-identical to decode.
#include "dxl_job.h"
#include "vecmath.h"

extern "C" {
void dxl_out_job(const float *__restrict acc, float *__restrict y, const int32_t *__restrict jp, int32_t b) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int32_t *r = jp;
  if (r[F_DMODE] == 0) {
    const float *__restrict a = acc + (unsigned)(r[F_B0] + b) * kL * kBandFloats;
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < kL * kBandFloats; i += 32) aie::store_v(y + i, aie::load_v<32>(a + i));
  } else {
    const float *__restrict u = acc + (unsigned)b * kL * kBandFloats;
    const float *__restrict g = acc + (unsigned)(r[F_G0] + b) * kL * kBandFloats;
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < kL * kBandFloats; i += 32) {
      const v32f a = vsiluN<32>(aie::load_v<32>(g + i));
      aie::store_v(y + i, fmul32(a, aie::load_v<32>(u + i)));
    }
  }
}
}
