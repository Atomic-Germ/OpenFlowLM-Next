#pragma once
//===- dn_step.h -------------------------------------------*- C++ -*-===//
//
// Gated DeltaNet decode step for one v-head on the AIE core (phase 1, correct
// first). Math = tools/kernel-interp/decode_step.py linear_decode, per head:
//
//   S  *= decay
//   delta = beta * (v - S^T k)          (S^T k uses the decayed S)
//   S  += k (x) delta
//   o   = (S^T q) / sqrt(128)
//
// folded into two passes over S (fp32 [128 rows i][128 cols j], row-major):
//   pass 1:  t[j]   = sum_i k[i] * S[i][j]
//   pass 2:  delta  = beta * (v - decay * t)                 (once, blk 0)
//            S'[i][j] = decay * S[i][j] + k[i] * delta[j]
//            o[j]   += S'[i][j] * q[i];   o /= sqrt(128)      (end, blk 7)
//
// S does not fit in L1 (64 KB), so it streams through in 16-row slices (8 KB),
// twice per head. All products are bf16 x bf16 into fp32 accumulators: AIE2P
// has no fp32 vector multiplier (aie::mul<float> returns zero, silently -- see
// LLMNpuTest CLAUDE.md), so every fp32 operand is split into hi + lo bf16
// halves and the three significant cross terms are accumulated (~2^-16 rel).
//
// Per-head vector record (fp32[512]): [k 0..127][q 128..255][v 256..383]
//                                     [decay @384][beta @385][pad]
// Scratch (L1, fp32/bf16 [128]): t, o, k_hl, q_hl, delta_hl (hi/lo bf16 pairs
// packed as [hi 0..127][lo 128..255]).

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kD = 128;          // head dim
static constexpr unsigned kSliceRows = 16;   // rows of S per streamed element
static constexpr unsigned kV = 16;           // vector width
static constexpr unsigned kNBlk = kD / kSliceRows;   // 8 slices per S

using vf = aie::vector<float, kV>;
using vb = aie::vector<bfloat16, kV>;
using accf16 = aie::accum<accfloat, kV>;

// fp32 vector -> (hi, lo) bf16 halves, hi + lo == v to ~16 mantissa bits.
static inline void split16(const vf &v, vb &h, vb &l) {
  accf16 a;
  a.from_vector(v);
  h = a.template to_vector<bfloat16>();
  l = aie::sub(a, h).template to_vector<bfloat16>();
}

// Split fp32[128] into packed bf16 [hi 128 | lo 128].
static inline void split_vec(const float *__restrict src, bfloat16 *__restrict hl) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kV) {
    vb h, l;
    split16(aie::load_v<kV>(src + j), h, l);
    aie::store_v(hl + j, h);
    aie::store_v(hl + kD + j, l);
  }
}

// acc += a * s for fp32 vector a (split inline) and pre-split scalar s.
static inline accf16 mac_split(accf16 acc, const vf &a, bfloat16 sh, bfloat16 sl) {
  vb ah, al;
  split16(a, ah, al);
  acc = aie::mac(acc, ah, sh);
  acc = aie::mac(acc, ah, sl);
  acc = aie::mac(acc, al, sh);
  return acc;
}

// ---- pass 1, slice blk: t[j] += sum_{i in slice} k[i] * S[i][j]
static inline void dn_pass1_slice(const float *__restrict S, const float *__restrict vec,
                                  float *__restrict t, bfloat16 *__restrict k_hl,
                                  bfloat16 *__restrict q_hl, unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (blk == 0) {
    split_vec(vec, k_hl);          // k
    split_vec(vec + kD, q_hl);     // q
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kD; j += kV)
      aie::store_v(t + j, aie::zeros<float, kV>());
  }
  const bfloat16 *kh = k_hl + blk * kSliceRows;
  const bfloat16 *kl = k_hl + kD + blk * kSliceRows;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kV) {
    accf16 acc;
    acc.from_vector(aie::load_v<kV>(t + j));
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < kSliceRows; ++i)
      acc = mac_split(acc, aie::load_v<kV>(S + i * kD + j), kh[i], kl[i]);
    aie::store_v(t + j, acc.template to_vector<float>());
  }
}

// ---- pass 2, slice blk: S' = decay*S + k (x) delta ; o += S'^T q
static inline void dn_pass2_slice(const float *__restrict S, float *__restrict Sout,
                                  const float *__restrict vec, const float *__restrict t,
                                  float *__restrict o, const bfloat16 *__restrict k_hl,
                                  const bfloat16 *__restrict q_hl,
                                  bfloat16 *__restrict delta_hl, unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float decay = vec[384];
  const float beta = vec[385];
  const bfloat16 dh = (bfloat16)decay;
  const bfloat16 dl = (bfloat16)(decay - (float)dh);
  if (blk == 0) {
    const bfloat16 bh = (bfloat16)beta;
    const bfloat16 bl = (bfloat16)(beta - (float)bh);
    const bfloat16 ndh = (bfloat16)(-decay);
    const bfloat16 ndl = (bfloat16)(-decay - (float)ndh);
    const float *v = vec + 2 * kD;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kD; j += kV) {
      // u = v - decay * t
      accf16 a;
      a.from_vector(aie::load_v<kV>(v + j));
      a = mac_split(a, aie::load_v<kV>(t + j), ndh, ndl);
      // delta = beta * u
      accf16 d = aie::zeros<accfloat, kV>();
      d = mac_split(d, a.template to_vector<float>(), bh, bl);
      vb h, l;
      split16(d.template to_vector<float>(), h, l);
      aie::store_v(delta_hl + j, h);
      aie::store_v(delta_hl + kD + j, l);
      aie::store_v(o + j, aie::zeros<float, kV>());
    }
  }
  const bfloat16 *kh = k_hl + blk * kSliceRows;
  const bfloat16 *kl = k_hl + kD + blk * kSliceRows;
  const bfloat16 *qh = q_hl + blk * kSliceRows;
  const bfloat16 *ql = q_hl + kD + blk * kSliceRows;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kV) {
    const vb deh = aie::load_v<kV>(delta_hl + j);
    const vb del = aie::load_v<kV>(delta_hl + kD + j);
    accf16 oacc;
    oacc.from_vector(aie::load_v<kV>(o + j));
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < kSliceRows; ++i) {
      // S' = decay * S + k[i] * delta
      accf16 s = aie::zeros<accfloat, kV>();
      s = mac_split(s, aie::load_v<kV>(S + i * kD + j), dh, dl);
      s = aie::mac(s, deh, kh[i]);
      s = aie::mac(s, deh, kl[i]);
      s = aie::mac(s, del, kh[i]);
      const vf sn = s.template to_vector<float>();
      aie::store_v(Sout + i * kD + j, sn);
      // o += S' * q[i]
      oacc = mac_split(oacc, sn, qh[i], ql[i]);
    }
    aie::store_v(o + j, oacc.template to_vector<float>());
  }
  if (blk == kNBlk - 1) {
    // o /= sqrt(128)
    const float inv = 0.088388347648318f;
    const bfloat16 ih = (bfloat16)inv;
    const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kD; j += kV) {
      accf16 a = aie::zeros<accfloat, kV>();
      a = mac_split(a, aie::load_v<kV>(o + j), ih, il);
      aie::store_v(o + j, a.template to_vector<float>());
    }
  }
}
