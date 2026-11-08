#pragma once
// MoE router on one core: logits = xm @ W (W bf16 [2048][E], streamed as 4 KB
// elements of rows), p = softmax(logits) or sigmoid(logits) with expert-bias
// selection, top-k with renormalised weights.
//   out (4 KB): [p fp32[E] @0][idx int32[topk] @E*4 B][w fp32[topk] @E*4+topk*4 B][pad]
// The healthy default (ROUTER_SIGMOID=0, ROUTER_BIAS=0, E=256, topk 8) is the shipped
// softmax router. LFM2-MoE compiles with -DROUTER_SIGMOID=1 -DROUTER_BIAS=1
// -DROUTER_E=64/32 -DROUTER_TOPK=4 (sigmoid scores, biased selection, unbiased
// renormalisation). The macro defaults keep every existing build byte-identical.
#include "vecmath.h"

#ifndef ROUTER_E
#define ROUTER_E 256
#endif
#ifndef ROUTER_TOPK
#define ROUTER_TOPK 8
#endif
#ifndef ROUTER_SIGMOID
#define ROUTER_SIGMOID 0
#endif
#ifndef ROUTER_BIAS
#define ROUTER_BIAS 0
#endif
#ifndef ROUTER_NORM
#define ROUTER_NORM 1
#endif

static constexpr unsigned kE = ROUTER_E;
static constexpr unsigned kTopk = ROUTER_TOPK;
static constexpr unsigned kRowsPerElem = 4096 / (2 * kE);   // 8 for E=256, 32 for E=64, 64 for E=32
static constexpr unsigned kV = 32;

static inline void router_copy_x_impl(const bfloat16 *__restrict src, bfloat16 *__restrict dst) {
  for (unsigned j = 0; j < 2048; j += kV)
    aie::store_v(dst + j, aie::load_v<kV>(src + j));
}

// acc[E] += sum_{r in element} x[rows*rb + r] * W[r][:]
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

static inline float sel_score(float p, const float *__restrict bias, unsigned e) {
#if ROUTER_BIAS
  return p + bias[e];
#else
  (void)bias; (void)e;
  return p;
#endif
}

// softmax/sigmoid + top-k over the selection scores
static inline void router_fin_impl(const float *__restrict acc, const float *__restrict bias, float *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#if ROUTER_SIGMOID
  // p = sigmoid(logits)
  for (unsigned j = 0; j < kE; j += kV) {
    const v32f s = vsigmoidN<32>(aie::load_v<kV>(acc + j));
    aie::store_v(out + j, s);
  }
#else
  // softmax (the shipped law)
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
#endif
  // top-k by the selection scores (p, or p + bias); the weights renormalise the
  // UNBIASED p over the winners (norm_topk_prob), then scale by 1.0.
  int32_t *idx = (int32_t *)(out + kE);
  float *w = out + kE + kTopk;
  uint32_t taken[kE / 32];
  for (unsigned i = 0; i < kE / 32; ++i) taken[i] = 0;
  float wsum = 0.f;
  for (unsigned k = 0; k < kTopk; ++k) {
    float best = 0.f;
    int bi = -1;
    for (unsigned e = 0; e < kE; ++e) {
      if (taken[e >> 5] & (1u << (e & 31))) continue;
      const float s = sel_score(out[e], bias ? bias : nullptr, e);
      if (bi < 0 || s > best) { best = s; bi = (int)e; }
    }
    taken[bi >> 5] |= 1u << (bi & 31);
    idx[k] = bi;
    w[k] = out[bi];
    wsum += out[bi];
  }
  const float rn = 1.0f / wsum;
  for (unsigned k = 0; k < kTopk; ++k) w[k] *= rn;
}

static inline void router_fin_impl(const float *__restrict acc, float *__restrict out) {
  router_fin_impl(acc, nullptr, out);
}
