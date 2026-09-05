#pragma once
//===- dnx.h -----------------------------------------------*- C++ -*-===//
//
// Gated DeltaNet decode step on the whole-layer design's main cores (phase 2
// "whole-layer context"): the math of designs/deltanet/dn_step.h, re-sliced to
// the main cores' streams. S (fp32 [128 rows][128 cols], one head per 64 KB)
// arrives through the weight fifo in 10240 B elements = 20 rows, so a head is
// streamed as 7 slices = 140 rows: the 12 pad rows are ZERO in DDR and stay
// zero (S' = decay*0 + k[i]*delta with k[i] = 0 for i >= 128, the hi/lo
// records are zero-padded to 160 entries), and they contribute nothing to
// t / o (k[i] = q[i] = 0). The updated rows leave through the result fifo in
// 256 B elements = half rows: pass 2 runs per (row, half).
//
//   pass 1 (per slice):  t[j] += sum_i k[i] * S[i][j]
//   delta (per head):    delta = beta * (v - decay * t);  o = 0
//   pass 2 (per row i, half h):  S'[i][h] = decay * S[i][h] + k[i] * delta[h]
//                                o[h]    += S'[i][h] * q[i]
//   ofin (per head, half h):     ye = o[h] / sqrt(128)
//
// Per-head vector record (fp32[512], the first 2 KB of a 10 KB element):
//   [k 0..127][q 128..255][v 256..383][decay @384][beta @385][pad]
// L1 scratch: one f32[1280] buffer `ds` (layout below).

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kD = 128;          // head dim
static constexpr unsigned kRowsX = 20;       // rows of S per streamed element
static constexpr unsigned kPad = 160;        // hi/lo record stride (7 slices x 20)
static constexpr unsigned kV = 16;
static constexpr unsigned kHalf = 64;        // columns per result element

using vf = aie::vector<float, kV>;
using vb = aie::vector<bfloat16, kV>;
using accf16 = aie::accum<accfloat, kV>;

static inline void split16(const vf &v, vb &h, vb &l) {
  accf16 a;
  a.from_vector(v);
  h = a.template to_vector<bfloat16>();
  l = aie::sub(a, h).template to_vector<bfloat16>();
}

// fp32[128] -> packed bf16 [hi 0..159 | lo 0..159], entries 128..159 zero.
static inline void split_vec_pad(const float *__restrict src, bfloat16 *__restrict hl) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kV) {
    vb h, l;
    split16(aie::load_v<kV>(src + j), h, l);
    aie::store_v(hl + j, h);
    aie::store_v(hl + kPad + j, l);
  }
#pragma clang loop unroll(disable)
  for (unsigned j = kD; j < kPad; j += kV) {
    aie::store_v(hl + j, aie::zeros<bfloat16, kV>());
    aie::store_v(hl + kPad + j, aie::zeros<bfloat16, kV>());
  }
}

static inline accf16 mac_split(accf16 acc, const vf &a, bfloat16 sh, bfloat16 sl) {
  vb ah, al;
  split16(a, ah, al);
  acc = aie::mac(acc, ah, sh);
  acc = aie::mac(acc, ah, sl);
  acc = aie::mac(acc, al, sh);
  return acc;
}

// A scalar fp32 -> bf16 hi/lo WITHOUT scalar float ops (no soft-float library).
static inline void split_scalar(float s, bfloat16 *__restrict out2) {
  vb h, l;
  split16(aie::broadcast<float, kV>(s), h, l);
  out2[0] = h[0];
  out2[1] = l[0];
}

// ds (f32[1280]) layout, floats: vec @0 (512) | t @512 | o @640 | k_hl bf16[320] @768 | q_hl @928
//                                | delta_hl bf16[256] @1088 | dd bf16[16] @1216
static constexpr unsigned DS_VEC = 0, DS_T = 512, DS_O = 640, DS_KHL = 768, DS_QHL = 928, DS_DHL = 1088, DS_DD = 1216;

// ---- pass 1, slice blk (20 rows): t[j] += sum_i k[i] * S[i][j]; blk 0 splits k, q and zeroes t.
static inline void dnx_pass1_slice(const float *__restrict S, float *__restrict ds, unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict vec = ds + DS_VEC;
  float *__restrict t = ds + DS_T;
  bfloat16 *__restrict k_hl = (bfloat16 *)(ds + DS_KHL);
  bfloat16 *__restrict q_hl = (bfloat16 *)(ds + DS_QHL);
  if (blk == 0) {
    split_vec_pad(vec, k_hl);
    split_vec_pad(vec + kD, q_hl);
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < kD; j += kV)
      aie::store_v(t + j, aie::zeros<float, kV>());
  }
  const bfloat16 *kh = k_hl + blk * kRowsX;
  const bfloat16 *kl = k_hl + kPad + blk * kRowsX;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kV) {
    accf16 acc;
    acc.from_vector(aie::load_v<kV>(t + j));
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < kRowsX; ++i)
      acc = mac_split(acc, aie::load_v<kV>(S + i * kD + j), kh[i], kl[i]);
    aie::store_v(t + j, acc.template to_vector<float>());
  }
}

// ---- once per head after pass 1: delta = beta * (v - decay * t) as hi/lo, o = 0, dd = decay hi/lo
static inline void dnx_delta_head(float *__restrict ds) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict vec = ds + DS_VEC;
  const float *__restrict t = ds + DS_T;
  float *__restrict o = ds + DS_O;
  bfloat16 *__restrict delta_hl = (bfloat16 *)(ds + DS_DHL);
  bfloat16 *__restrict dd = (bfloat16 *)(ds + DS_DD);
  const float decay = vec[384];
  const float beta = vec[385];
  bfloat16 nd[2], bb[2];
  split_scalar(-decay, nd);
  split_scalar(beta, bb);
  split_scalar(decay, dd);
  const float *v = vec + 2 * kD;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kV) {
    accf16 a;
    a.from_vector(aie::load_v<kV>(v + j));
    a = mac_split(a, aie::load_v<kV>(t + j), nd[0], nd[1]);       // u = v - decay * t
    accf16 d = aie::zeros<accfloat, kV>();
    d = mac_split(d, a.template to_vector<float>(), bb[0], bb[1]);  // delta = beta * u
    vb h, l;
    split16(d.template to_vector<float>(), h, l);
    aie::store_v(delta_hl + j, h);
    aie::store_v(delta_hl + kD + j, l);
    aie::store_v(o + j, aie::zeros<float, kV>());
  }
}

// ---- pass 2, row i of the slice at S, half hf: S'[i][hf] -> ye (64 floats); o[hf] += S' * q[i]
static inline void dnx_row_half(const float *__restrict S, float *__restrict ds, float *__restrict ye,
                                unsigned blk, unsigned i, unsigned hf) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *__restrict o = ds + DS_O;
  const bfloat16 *__restrict k_hl = (const bfloat16 *)(ds + DS_KHL);
  const bfloat16 *__restrict q_hl = (const bfloat16 *)(ds + DS_QHL);
  const bfloat16 *__restrict delta_hl = (const bfloat16 *)(ds + DS_DHL);
  const bfloat16 *__restrict dd = (const bfloat16 *)(ds + DS_DD);
  const unsigned r = blk * kRowsX + i;            // row index within the (padded) head
  const bfloat16 kh = k_hl[r], kl = k_hl[kPad + r];
  const bfloat16 qh = q_hl[r], ql = q_hl[kPad + r];
  const bfloat16 dh = dd[0], dl = dd[1];
  const float *__restrict Si = S + i * kD + hf * kHalf;
  float *__restrict oh = o + hf * kHalf;
  const bfloat16 *__restrict deh0 = delta_hl + hf * kHalf;
  const bfloat16 *__restrict del0 = delta_hl + kD + hf * kHalf;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) {
    accf16 s = aie::zeros<accfloat, kV>();
    s = mac_split(s, aie::load_v<kV>(Si + j), dh, dl);
    const vb deh = aie::load_v<kV>(deh0 + j);
    const vb del = aie::load_v<kV>(del0 + j);
    s = aie::mac(s, deh, kh);
    s = aie::mac(s, deh, kl);
    s = aie::mac(s, del, kh);
    const vf sn = s.template to_vector<float>();
    aie::store_v(ye + j, sn);
    accf16 oacc;
    oacc.from_vector(aie::load_v<kV>(oh + j));
    oacc = mac_split(oacc, sn, qh, ql);
    aie::store_v(oh + j, oacc.template to_vector<float>());
  }
}

// ---- per head, half hf: ye = o[hf] / sqrt(128)
static inline void dnx_ofin_half(const float *__restrict ds, float *__restrict ye, unsigned hf) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict o = ds + DS_O;
  bfloat16 ii[2];
  split_scalar(0.088388347648318f, ii);                    // 1/sqrt(128) as bf16 hi/lo
  const bfloat16 ih = ii[0], il = ii[1];
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) {
    accf16 a = aie::zeros<accfloat, kV>();
    a = mac_split(a, aie::load_v<kV>(o + hf * kHalf + j), ih, il);
    aie::store_v(ye + j, a.template to_vector<float>());
  }
}
