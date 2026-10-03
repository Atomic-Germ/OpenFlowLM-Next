// DeltaNet post step for 8 heads (one 4 KB element of o and of z):
//   og[h] = bf16( o[h] * rsqrt(mean(o[h]^2) + 1e-6) * ssm_norm_w * silu(z[h]) )
// (tools/kernel-interp/decode_step.py: og = (rms(o) * nw).reshape(4096) * z_silu)
// Keep legacy fused arithmetic unchanged. Standalone acceptance at both
// 32 and 48 heads opts into the precise BF16-boundary path.
#ifndef POST_PRECISE
#define POST_PRECISE 0
#endif
#if POST_PRECISE
#include "vecmath_precise.h"
#else
#include "vecmath.h"
#endif

#if POST_CARRY
#include "post_carry.h"
#endif

static constexpr unsigned kHD = 128;
static constexpr unsigned kV = 32;
static constexpr unsigned kHeads = 8;

extern "C" {
void post_fn(const float *__restrict o, const float *__restrict z, const bfloat16 *__restrict nw,
             bfloat16 *__restrict og
#if POST_TRACE
             , float *__restrict trace
#endif
             ) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned h = 0; h < kHeads; ++h) {
    const float *oh = o + h * kHD;
    const float *zh = z + h * kHD;
    bfloat16 *gh = og + h * kHD;
#if POST_PRECISE
    accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kHD; j += kV) {
      const v32f v = aie::load_v<kV>(oh + j);
      ss = aie::add(ss, precise_mulN<kV>(v, v));
    }
    const float sum_sq = aie::reduce_add(ss.template to_vector<float>());
    const float inv = srsqrt(sum_sq * (1.0f / kHD) + 1e-6f);
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kHD; j += kV) {
#if POST_CARRY
      const auto ov = aie::load_v<kV>(oh+j), zv = aie::load_v<kV>(zh+j);
      const auto iv = aie::broadcast<float,kV>(inv);
      accf32 weight(aie::load_v<kV>(nw+j));
      const auto wv = weight.to_vector<float>();
      const auto sigmoid = post_sigmoid(zv);
      const auto r = post_product(ov,iv,wv,zv,sigmoid);
#if POST_TRACE
      const auto on = ln_mul_rne(ov,iv), t = ln_mul_rne(on,wv), sz = ln_mul_rne(zv,sigmoid);
#endif
#else
      const v32f on = precise_mulN<kV>(aie::load_v<kV>(oh + j), aie::broadcast<float, kV>(inv));
      accf32 weight(aie::load_v<kV>(nw + j));
      const v32f t = precise_mulN<kV>(on, weight.template to_vector<float>());
      const v32f sz = precise_siluN<kV>(aie::load_v<kV>(zh + j));
      const v32f r = precise_mulN<kV>(t, sz);
#endif
#if POST_TRACE
      const unsigned off = h * kHD + j;
      aie::store_v(trace + off, aie::broadcast<float,kV>(sum_sq));
      aie::store_v(trace + 1024 + off, aie::broadcast<float,kV>(inv));
      aie::store_v(trace + 2048 + off, on);
      aie::store_v(trace + 3072 + off, t);
      aie::store_v(trace + 4096 + off, sz);
      aie::store_v(trace + 5120 + off, r);
#endif
      accf32 rr;
      rr.from_vector(r);
      aie::store_v(gh + j, rr.template to_vector<bfloat16>());
    }
#else
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
#endif
  }
}
}
