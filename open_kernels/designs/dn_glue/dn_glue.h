#pragma once
//===- dn_glue.h -------------------------------------------*- C++ -*-===//
//
// Linear-attention layer glue around the DeltaNet step (decode, T = 1):
//
//   alpha  = xn @ Wa            (2048 x 32 bf16)  -> decay[h] = exp(A[h] * softplus(alpha[h] + dt_bias[h]))
//   betal  = xn @ Wb            (2048 x 32 bf16)  -> beta[h]  = sigmoid(betal[h])
//   c      = silu(w0*s0 + w1*s1 + w2*s2 + w3*qkv)  depthwise conv k=4 over
//            [state rows 0..2, this token's qkv], 8192 channels
//   q, k   = L2-normalise per 128-head (channels 0..2047 / 2048..4095)
//   v      = c[4096..8191]
//   state' = [s1, s2, bf16(qkv)]
//   record[h] (fp32[512]) = [k[h/2] | q[h/2] | v[h] | decay[h] @384 | beta[h] @385]
//
// Reference: tools/kernel-interp/decode_step.py linear_decode. Idioms:
// sigmoid(x) = (tanh(x/2)+1)/2 with aie::tanh<bfloat16> (LLMNpuTest); every
// fp32 x anything product is a bf16 hi/lo split (no fp32 vector multiply).
//
// Channel tiles of 1024 (8 tiles): tiles 0-1 = q heads, 2-3 = k heads, 4-7 = v.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kHid = 2048;
static constexpr unsigned kNHead = 32;
static constexpr unsigned kHD = 128;
static constexpr unsigned kTile = 1024;      // conv channels per tile
static constexpr unsigned kV = 32;

#include "vecmath.h"   // fp32 vector math on bf16 MACs (split32, fmul32, vexp32, vsigmoid32, srsqrt)

// ---- alpha/beta projection tile: W element = kAbRows rows x 32 bf16 (4 KB); acc[32] += x[rows] @ W
static constexpr unsigned kAbRows = 64;
static inline void glue_ab_tile(const bfloat16 *__restrict W, const bfloat16 *__restrict xn,
                                float *__restrict acc, unsigned tile, bool first) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 a;
  if (first)
    a = aie::zeros<accfloat, kV>();
  else
    a.from_vector(aie::load_v<kV>(acc));
  const bfloat16 *x = xn + tile * kAbRows;
#pragma clang loop unroll(disable)
  for (unsigned r = 0; r < kAbRows; ++r)
    a = aie::mac(a, aie::load_v<kV>(W + r * kV), x[r]);
  aie::store_v(acc, a.template to_vector<float>());
}

// (scalar sexp/slog/ssoftplus/ssigmoid live in vecmath.h)

// ---- decay/beta from the two projections
static inline void glue_small(const float *__restrict small /* A[32] @0, dt_bias[32] @32 */,
                              const float *__restrict acc_a, const float *__restrict acc_b,
                              float *__restrict decay, float *__restrict beta) {
  for (unsigned h = 0; h < kNHead; ++h) {
    decay[h] = sexp(small[h] * ssoftplus(acc_a[h] + small[32 + h]));
    beta[h] = ssigmoid(acc_b[h]);
  }
}

// ---- conv tile t: c = silu(conv), new state rows, q/k L2 norm
static inline void glue_conv_tile(const float *__restrict q0, const float *__restrict q1,
                                  const bfloat16 *__restrict s0, const bfloat16 *__restrict s1,
                                  const bfloat16 *__restrict s2, const bfloat16 *__restrict w01,
                                  const bfloat16 *__restrict w23,
                                  bfloat16 *__restrict ns0, bfloat16 *__restrict ns1,
                                  bfloat16 *__restrict ns2, float *__restrict qk,
                                  float *__restrict vt, unsigned t) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *__restrict w = w01;                // rows 0,1 (4 KB element)
  const bfloat16 *__restrict w2 = w23;               // rows 2,3
  (void)w2;
  const v32b half = aie::broadcast<bfloat16, kV>((bfloat16)0.5f);
  const v32b one = aie::broadcast<bfloat16, kV>((bfloat16)1.0f);
  float *__restrict dst = (t < 4) ? (qk + t * kTile) : vt;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kTile; j += kV) {
    const float *qp = (j < 512) ? (q0 + j) : (q1 + (j - 512));
    const v32f x = aie::load_v<kV>(qp);
    const v32b s0v = aie::load_v<kV>(s0 + j);
    const v32b s1v = aie::load_v<kV>(s1 + j);
    const v32b s2v = aie::load_v<kV>(s2 + j);
    accf32 a = aie::mul(aie::load_v<kV>(w + j), s0v);
    a = aie::mac(a, aie::load_v<kV>(w + kTile + j), s1v);
    a = aie::mac(a, aie::load_v<kV>(w2 + j), s2v);
    a = mac_vv(a, x, aie::load_v<kV>(w2 + kTile + j));
    // silu(a) = a * sigmoid(a), fp32 throughout (see vsigmoid32)
    const v32f af = a.template to_vector<float>();
    aie::store_v(dst + j, fmul32(af, vsigmoid32(af)));
    // state shift
    aie::store_v(ns0 + j, s1v);
    aie::store_v(ns1 + j, s2v);
    v32b xh, xl;
    split32(x, xh, xl);
    aie::store_v(ns2 + j, xh);
  }
  if (t < 4) {
    // L2-normalise the 8 heads of this tile in place
#pragma clang loop unroll(disable)
    for (unsigned hh = 0; hh < kTile / kHD; ++hh) {
      float *__restrict hp = dst + hh * kHD;
      accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
      for (unsigned j = 0; j < kHD; j += kV) {
        v32b ch, cl;
        split32(aie::load_v<kV>(hp + j), ch, cl);
        ss = aie::mac(ss, ch, ch);
        ss = aie::mac(ss, ch, cl);
        ss = aie::mac(ss, ch, cl);
      }
      // aie::invsqrt is a coarse hardware approximation (~2% observed);
      // two Newton steps bring it to fp32.
      const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) + 1e-6f);
      const bfloat16 ih = (bfloat16)inv;
      const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
      for (unsigned j = 0; j < kHD; j += kV) {
        accf32 o = aie::zeros<accfloat, kV>();
        o = mac_vs(o, aie::load_v<kV>(hp + j), ih, il);
        aie::store_v(hp + j, o.template to_vector<float>());
      }
    }
  }
}

// ---- emit record for head h (v tile already in vt): rec fp32[512]
static inline void glue_emit(const float *__restrict qk, const float *__restrict vt,
                             const float *__restrict decay, const float *__restrict beta,
                             float *__restrict rec, unsigned h) {
  const unsigned kh = h / 2;
  const float *kp = qk + 2048 + kh * kHD;
  const float *qp = qk + kh * kHD;
  const float *vp = vt + (h % 8) * kHD;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHD; j += kV) {
    aie::store_v(rec + j, aie::load_v<kV>(kp + j));
    aie::store_v(rec + kHD + j, aie::load_v<kV>(qp + j));
    aie::store_v(rec + 2 * kHD + j, aie::load_v<kV>(vp + j));
    aie::store_v(rec + 3 * kHD + j, aie::zeros<float, kV>());
  }
  rec[384] = decay[h];
  rec[385] = beta[h];
}
