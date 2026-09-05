// DeltaNet post step for 8 heads (one 4 KB element of o and of z):
//   og[h] = bf16( o[h] * rsqrt(mean(o[h]^2) + 1e-6) * ssm_norm_w * silu(z[h]) )
// (tools/kernel-interp/decode_step.py: og = (rms(o) * nw).reshape(4096) * z_silu)
#include "vecmath.h"

static constexpr unsigned kHD = 128;
static constexpr unsigned kV = 32;
static constexpr unsigned kHeads = 8;

extern "C" {
void post_fn(const float *__restrict o, const float *__restrict z, const bfloat16 *__restrict nw,
             bfloat16 *__restrict og) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned h = 0; h < kHeads; ++h) {
    const float *oh = o + h * kHD;
    const float *zh = z + h * kHD;
    bfloat16 *gh = og + h * kHD;
    accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kHD; j += kV) {
      v32b hi, lo;
      split32(aie::load_v<kV>(oh + j), hi, lo);
      ss = aie::mac(ss, hi, hi);
      ss = aie::mac(ss, hi, lo);
      ss = aie::mac(ss, hi, lo);
    }
    const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kHD) + 1e-6f);
    const bfloat16 ih = (bfloat16)inv;
    const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kHD; j += kV) {
      accf32 on = aie::zeros<accfloat, kV>();
      on = mac_vs(on, aie::load_v<kV>(oh + j), ih, il);                 // o * inv
      accf32 t = aie::zeros<accfloat, kV>();
      t = mac_vv(t, on.template to_vector<float>(), aie::load_v<kV>(nw + j));   // * nw
      const v32f sz = vsiluN<32>(aie::load_v<kV>(zh + j));
      const v32f r = fmul32(t.template to_vector<float>(), sz);
      accf32 rr;
      rr.from_vector(r);
      aie::store_v(gh + j, rr.template to_vector<bfloat16>());
    }
  }
}
}
