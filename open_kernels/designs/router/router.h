#pragma once
// MoE router on one core: logits = xm @ W (W bf16 [2048][256], streamed as 4 KB
// elements of 8 rows), p = softmax(logits), top-8 with renormalised weights.
//   out (4 KB): [p fp32[256] @0][idx int32[8] @1024 B][w fp32[8] @1056 B][pad]
// Reference: decode_step.py moe_decode (p = softmax(xm @ router); top = argsort(-p)[:8]; w8 = p[top]/sum).
// One entry point per .cc (IRON compiles the source once per ExternalFunction).
#include "vecmath.h"

static constexpr unsigned kE = 256;
static constexpr unsigned kRowsPerElem = 8;
static constexpr unsigned kV = 32;

static inline void router_copy_x_impl(const bfloat16 *__restrict src, bfloat16 *__restrict dst) {
  for (unsigned j = 0; j < 2048; j += kV)
    aie::store_v(dst + j, aie::load_v<kV>(src + j));
}

// acc[256] += sum_{r in element} x[8*rb + r] * W[r][:]
static inline void router_acc_impl(const bfloat16 *__restrict W, const bfloat16 *__restrict x,
                                   float *__restrict acc, int rb) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *xr = x + rb * kRowsPerElem;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kE; j += kV) {
    accf32 a;
    if (rb == 0)
      a = aie::zeros<accfloat, kV>();
    else
      a.from_vector(aie::load_v<kV>(acc + j));
#pragma clang loop unroll(disable)
    for (unsigned r = 0; r < kRowsPerElem; ++r)
      a = aie::mac(a, aie::load_v<kV>(W + r * kE + j), xr[r]);
    aie::store_v(acc + j, a.template to_vector<float>());
  }
}

// softmax + top-8
static inline void router_fin_impl(const float *__restrict acc, float *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  v32f mx = aie::load_v<kV>(acc);
  for (unsigned j = kV; j < kE; j += kV)
    mx = aie::max(mx, aie::load_v<kV>(acc + j));
  const float m = aie::reduce_max(mx);
  const v32f mb = aie::broadcast<float, kV>(m);
  accf32 sacc = aie::zeros<accfloat, kV>();
  for (unsigned j = 0; j < kE; j += kV) {
    const v32f e = vexpN<32>(fsub32(aie::load_v<kV>(acc + j), mb));
    aie::store_v(out + j, e);
    accf32 ea;
    ea.from_vector(e);
    sacc = aie::add(sacc, ea);
  }
  const float inv = 1.0f / aie::reduce_add(sacc.template to_vector<float>());
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
  for (unsigned j = 0; j < kE; j += kV) {
    accf32 p = aie::zeros<accfloat, kV>();
    p = mac_vs(p, aie::load_v<kV>(out + j), ih, il);
    aie::store_v(out + j, p.template to_vector<float>());
  }
  // top-8: probabilities are positive, so their bit patterns order as uint32.
  const uint32_t *pu = (const uint32_t *)out;
  int32_t *idx = (int32_t *)(out + kE);
  float *w = out + kE + 8;
  uint32_t taken[kE / 32];
  for (unsigned i = 0; i < kE / 32; ++i) taken[i] = 0;
  float wsum = 0.f;
  for (unsigned k = 0; k < 8; ++k) {
    uint32_t best = 0;
    int bi = -1;
    for (unsigned e = 0; e < kE; ++e) {
      if (taken[e >> 5] & (1u << (e & 31))) continue;
      if (bi < 0 || pu[e] > best) { best = pu[e]; bi = (int)e; }
    }
    taken[bi >> 5] |= 1u << (bi & 31);
    idx[k] = bi;
    w[k] = out[bi];
    wsum += out[bi];
  }
  const float rn = 1.0f / wsum;
  for (unsigned k = 0; k < 8; ++k) w[k] *= rn;
}
