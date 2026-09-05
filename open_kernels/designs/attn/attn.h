#pragma once
//===- attn.h ----------------------------------------------*- C++ -*-===//
//
// Full-attention decode step (one token, one core), after the q/gate/k/v GEMVs:
//   q' = rope_p( rms256(q_h) * qn )     16 heads x 256   (qn = effective norm weight)
//   k' = rope_p( rms256(k_h) * kn )      2 heads x 256, v as is; k', v -> bf16 cache rows
//   for head h (kv head h/8): s_t = q'_h . K_t / 16 over t in [0, pos] (cache rows + new)
//   o_h = softmax(s) V  (online softmax, fp32 accumulators), og_h = o_h * sigmoid(gate_h)
// Partial RoPE: rotary dim 64 of 256, half-split pairs (i, i+32), theta 1e7; cos/sin
// for position p come from the host in the meta record.
// Reference: tools/kernel-interp/decode_step.py attn_decode.
//
// meta = two 1 KB elements: [qn bf16[256] @0 | kn bf16[256] @512] (per layer) and the position
// record [int32 pos @0 | int32 nf @4 | cos f32[32] @512 | sin f32[32] @640] (layer_x/layout.py
// ptab row pos; nf = the number of cache rows streamed). pb (int32[4]) = [pos, nf, rows seen]:
// attn_meta fills it, the core loops nf times, attn_step masks rows t >= pos (the whole-layer
// design streams one dummy row at position 0: a zero-length DMA is not an option).
// Elements are 1 KB: q/gate = one head (fp32[256]) each, k/v = one head each,
// K_t/V_t cache rows = bf16[512] (two kv heads).

#include "vecmath.h"

static constexpr unsigned kNH = 16;
static constexpr unsigned kKVH = 2;
static constexpr unsigned kHD = 256;
static constexpr unsigned kV = 32;

static inline void attn_meta_impl(const uint8_t *__restrict m0, const uint8_t *__restrict m1,
                                  bfloat16 *__restrict qn, bfloat16 *__restrict kn,
                                  float *__restrict cs, int32_t *__restrict pb) {
  const bfloat16 *q = (const bfloat16 *)m0;
  for (unsigned j = 0; j < kHD; j += kV) aie::store_v(qn + j, aie::load_v<kV>(q + j));
  const bfloat16 *k = (const bfloat16 *)(m0 + 512);
  for (unsigned j = 0; j < kHD; j += kV) aie::store_v(kn + j, aie::load_v<kV>(k + j));
  const float *c = (const float *)(m1 + 512);
  for (unsigned j = 0; j < 64; j += kV) aie::store_v(cs + j, aie::load_v<kV>(c + j));
  const int32_t *p = (const int32_t *)m1;
  pb[0] = p[0];
  pb[1] = p[1];
  pb[2] = 0;
  pb[3] = 0;
}

// x (fp32[256]) -> rms256 * w -> partial rope -> dst (fp32[256])
__attribute__((noinline)) inline void norm_rope(const float *__restrict x, const bfloat16 *__restrict w,
                             const float *__restrict cs, float *__restrict dst) {
  accf32 ss = aie::zeros<accfloat, kV>();
  for (unsigned j = 0; j < kHD; j += kV) {
    v32b h, l;
    split32(aie::load_v<kV>(x + j), h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kHD) + 1e-6f);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
  for (unsigned j = 0; j < kHD; j += kV) {
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vs(t, aie::load_v<kV>(x + j), ih, il);
    accf32 u = aie::zeros<accfloat, kV>();
    u = mac_vv(u, t.template to_vector<float>(), aie::load_v<kV>(w + j));
    aie::store_v(dst + j, u.template to_vector<float>());
  }
  // rope on dims [0,64): a = dst[0..32), b = dst[32..64)
  const v32f c = aie::load_v<kV>(cs);
  const v32f s = aie::load_v<kV>(cs + 32);
  const v32f a = aie::load_v<kV>(dst);
  const v32f b = aie::load_v<kV>(dst + 32);
  aie::store_v(dst, fsub32(fmul32(a, c), fmul32(b, s)));
  aie::store_v(dst + 32, fadd32(fmul32(b, c), fmul32(a, s)));
}

static inline void to_bf16_256(const float *__restrict src, bfloat16 *__restrict dst) {
  for (unsigned j = 0; j < kHD; j += kV) {
    accf32 a;
    a.from_vector(aie::load_v<kV>(src + j));
    aie::store_v(dst + j, a.template to_vector<bfloat16>());
  }
}

// q head h -> qs[h]
static inline void attn_q_impl(const float *__restrict qh, const bfloat16 *__restrict qn,
                               const float *__restrict cs, float *__restrict qs, int h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  norm_rope(qh, qn, cs, qs + h * kHD);
}
// k head h -> bf16 kout[h]  (the cache row half); v head h -> bf16 vout[h]
static inline void attn_k_impl(const float *__restrict kh, const bfloat16 *__restrict kn,
                               const float *__restrict cs, float *__restrict tmp,
                               bfloat16 *__restrict kout, int h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  norm_rope(kh, kn, cs, tmp);
  to_bf16_256(tmp, kout + h * kHD);
}
static inline void attn_v_impl(const float *__restrict vh, bfloat16 *__restrict vout, int h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  to_bf16_256(vh, vout + h * kHD);
}

static inline void attn_init_impl(float *__restrict oacc, float *__restrict ml) {
  for (unsigned j = 0; j < kNH * kHD; j += kV) aie::store_v(oacc + j, aie::zeros<float, kV>());
  for (unsigned h = 0; h < kNH; ++h) { ml[h] = -1e30f; ml[kNH + h] = 0.f; }
}

// one position: K_t, V_t bf16[512] (kv head 0 then 1)
__attribute__((noinline)) inline void attn_row_impl(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt,
                                  const float *__restrict qs, float *__restrict oacc,
                                  float *__restrict ml) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned h = 0; h < kNH; ++h) {
    const unsigned kvh = h / (kNH / kKVH);
    const float *q = qs + h * kHD;
    const bfloat16 *k = Kt + kvh * kHD;
    const bfloat16 *v = Vt + kvh * kHD;
    accf32 d = aie::zeros<accfloat, kV>();
    for (unsigned j = 0; j < kHD; j += kV)
      d = mac_vv(d, aie::load_v<kV>(q + j), aie::load_v<kV>(k + j));
    const float s = aie::reduce_add(d.template to_vector<float>()) * 0.0625f;   // / sqrt(256)
    const float m_old = ml[h];
    const float m_new = (s > m_old) ? s : m_old;
    const float a = sexp(m_old - m_new);
    const float b = sexp(s - m_new);
    ml[h] = m_new;
    ml[kNH + h] = ml[kNH + h] * a + b;
    const bfloat16 bh = (bfloat16)b;
    const bfloat16 bl = (bfloat16)(b - (float)bh);
    float *o = oacc + h * kHD;
    for (unsigned j = 0; j < kHD; j += kV) {
      accf32 acc;
      acc.from_vector(fscaleN<32>(aie::load_v<kV>(o + j), a));
      acc = aie::mac(acc, aie::load_v<kV>(v + j), bh);
      acc = aie::mac(acc, aie::load_v<kV>(v + j), bl);
      aie::store_v(o + j, acc.template to_vector<float>());
    }
  }
}

// cached row number pb[2] of the nf streamed: rows t >= pos are the position-0 dummy
static inline void attn_step_impl(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt,
                                  const float *__restrict qs, float *__restrict oacc,
                                  float *__restrict ml, int32_t *__restrict pb) {
  const int32_t t = pb[2];
  pb[2] = t + 1;
  if (t >= pb[0]) return;
  attn_row_impl(Kt, Vt, qs, oacc, ml);
}

// two heads -> one output element: og = o/l * sigmoid(gate)
__attribute__((noinline)) inline void attn_fin_impl(const float *__restrict oacc, const float *__restrict ml,
                                 const float *__restrict g0, const float *__restrict g1,
                                 bfloat16 *__restrict og, int hp) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned i = 0; i < 2; ++i) {
    const unsigned h = 2 * hp + i;
    const float *g = (i == 0) ? g0 : g1;
    const float inv = 1.0f / ml[kNH + h];
    const float *o = oacc + h * kHD;
    for (unsigned j = 0; j < kHD; j += kV) {
      const v32f on = fscaleN<32>(aie::load_v<kV>(o + j), inv);
      const v32f r = fmul32(on, vsigmoidN<32>(aie::load_v<kV>(g + j)));
      accf32 a;
      a.from_vector(r);
      aie::store_v(og + i * kHD + j, a.template to_vector<bfloat16>());
    }
  }
}
