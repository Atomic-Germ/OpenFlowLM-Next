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
// 256 B elements = half rows: pass 2 is called per (row, half), and the slice's
// first call updates the whole slice in place before the copy-outs.
//
//   pass 1 (per slice):  t[j] += sum_i k[i] * S[i][j]
//   delta (per head):    delta = beta * (v - decay * t);  o = 0
//   pass 2 (per slice, in place):  S'[i] = decay * S[i] + k[i] * delta
//                                  o    += S'[i] * q[i]
//   ofin (per head, half h):     ye = o[h] / sqrt(128)
//
// Per-head vector record (fp32[512], the first 2 KB of a 10 KB element):
//   [k 0..127][q 128..255][v 256..383][decay @384][beta @385][pad]
// L1 scratch: one f32[1280] buffer `ds` (layout below).

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// Rows of S per streamed weight element: a property of the element size the recipe chose --
// 10 KB elements (the 27B) give 10240 / (128*4) = 20 rows, 5 KB ones (Qwen3.5 dense, whose
// 12288-wide down table leaves no room for 10 KB) give 10. The default is the 27B's, so its
// kernels preprocess identically to the shipped ones; xcommon.py passes -DDNX_ROWS from
// `Common.DN_ROWS` only when it differs.
#ifndef DNX_ROWS
#define DNX_ROWS 20
#endif
static constexpr unsigned kD = 128;          // head dim
static constexpr unsigned kRowsX = DNX_ROWS; // rows of S per streamed element
// kPad is NOT the slice count times kRowsX (that is `Common.DN_PAD`, the padded S row count
// of the state buffer: 140 at 20 rows, 130 at 10). It is the hi/lo RECORD STRIDE inside ds,
// fixed by the ds slot geometry below -- k_hl at DS_KHL and q_hl at DS_QHL are 160 floats
// apart, so hi[0..kPad) and lo[kPad..2*kPad) as bf16 fit exactly at kPad = 160 -- and the
// same 160 serves every row count. Wiring it to DN_PAD instead moved every dnx_* object in
// the shipped 27B kernels (see .claude/plans/q-qwen35-handoff.md, "the lx xclbin size").
static constexpr unsigned kPad = 160;        // hi/lo record stride inside ds (DS_KHL / DS_QHL)
static_assert(kPad >= kD && kPad % 16 == 0, "dnx.h: the record stride must cover a head and be a vector multiple");
static_assert(((kD + kRowsX - 1) / kRowsX) * kRowsX <= kPad,
              "dnx.h: DNX_ROWS gives a padded row count past the ds hi/lo slot");
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

// fp32[128] -> packed bf16 [hi 0..159 | lo 0..159], entries 128..159 zero. Out of line: it is
// called twice per head and program memory is what the main core is short of, not cycles here.
static __attribute__((noinline)) void split_vec_pad(const float *__restrict src, bfloat16 *__restrict hl) {
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
#ifdef LX_NULL_DN
  return;                                    // timing ablation: the streams stay, the maths goes
#endif
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
  const bfloat16 *__restrict kh = k_hl + blk * kRowsX;
  const bfloat16 *__restrict kl = k_hl + kPad + blk * kRowsX;
  // two columns (one per half) per pass over the rows: each t[j] is a chain over i, and one
  // at a time the chain's latency was the whole cost. Same per-column order, so bit-identical.
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) {
    accf16 acc_a, acc_b;
    acc_a.from_vector(aie::load_v<kV>(t + j));
    acc_b.from_vector(aie::load_v<kV>(t + kHalf + j));
    const float *__restrict Sa = S + j;
    const float *__restrict Sb = S + kHalf + j;
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < kRowsX; ++i) {
      acc_a = mac_split(acc_a, aie::load_v<kV>(Sa), kh[i], kl[i]);
      acc_b = mac_split(acc_b, aie::load_v<kV>(Sb), kh[i], kl[i]);
      Sa += kD;
      Sb += kD;
    }
    aie::store_v(t + j, acc_a.template to_vector<float>());
    aie::store_v(t + kHalf + j, acc_b.template to_vector<float>());
  }
}

// ---- once per head after pass 1: delta = beta * (v - decay * t) as hi/lo, o = 0, dd = decay hi/lo
static inline void dnx_delta_head(float *__restrict ds) {
#ifdef LX_NULL_DN
  return;                                    // timing ablation: the streams stay, the maths goes
#endif
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

// ---- pass 2 for a whole slice, in place: S'[i] = decay * S[i] + k[i] * delta over the element's
// rows, o += S'[i] * q[i]. The element is the core's own copy of the slice (the fifo refills it
// from DDR next time), so S' overwrites S and the half-row calls below just copy out.
//
// Per row and column the arithmetic is exactly the per-(row, half) form this replaced, in the
// same order, so the result is bit-identical. What changed is the loop: the o accumulation is a
// true recurrence over rows (a split and three macs deep), and one half row per call it ran
// latency-bound, 8192 calls a layer. Here the same column vector of each half runs the
// recurrence side by side across the rows, so one chain's latency hides the other's.
static inline void dnx_slice_update(float *__restrict S, float *__restrict ds, unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *__restrict o = ds + DS_O;
  const bfloat16 *__restrict k_hl = (const bfloat16 *)(ds + DS_KHL);
  const bfloat16 *__restrict q_hl = (const bfloat16 *)(ds + DS_QHL);
  const bfloat16 *__restrict delta_hl = (const bfloat16 *)(ds + DS_DHL);
  const bfloat16 *__restrict dd = (const bfloat16 *)(ds + DS_DD);
  const bfloat16 dh = dd[0], dl = dd[1];
  const bfloat16 *__restrict kh_r = k_hl + blk * kRowsX;
  const bfloat16 *__restrict kl_r = k_hl + kPad + blk * kRowsX;
  const bfloat16 *__restrict qh_r = q_hl + blk * kRowsX;
  const bfloat16 *__restrict ql_r = q_hl + kPad + blk * kRowsX;
  // Two chains, not four: four column vectors in flight would hide more of the recurrence but
  // cost ~580 B more code, and the main core has ~370 B of program memory left.
  // S' and o go in two passes over the same rows rather than one. Together they keep more
  // accumulators live than the core has, and the single loop spilled four of them to stack and
  // back every row - half its bundles. S' is in L1 by the second pass, so re-reading it is
  // cheaper than the spills were.
#pragma clang loop unroll(disable)
  for (unsigned jv = 0; jv < kHalf; jv += kV) {
    const unsigned ja = jv, jb = kHalf + jv;               // the same column vector in each half
    const vb deha = aie::load_v<kV>(delta_hl + ja), dela = aie::load_v<kV>(delta_hl + kD + ja);
    const vb dehb = aie::load_v<kV>(delta_hl + jb), delb = aie::load_v<kV>(delta_hl + kD + jb);
    float *__restrict Sa = S + ja;
    float *__restrict Sb = S + jb;
#pragma clang loop min_iteration_count(DNX_ROWS)
    for (unsigned i = 0; i < kRowsX; ++i) {
      const bfloat16 kh = kh_r[i], kl = kl_r[i];
      accf16 sa = aie::zeros<accfloat, kV>();
      sa = mac_split(sa, aie::load_v<kV>(Sa), dh, dl);
      sa = aie::mac(sa, deha, kh);
      sa = aie::mac(sa, deha, kl);
      sa = aie::mac(sa, dela, kh);
      accf16 sb = aie::zeros<accfloat, kV>();
      sb = mac_split(sb, aie::load_v<kV>(Sb), dh, dl);
      sb = aie::mac(sb, dehb, kh);
      sb = aie::mac(sb, dehb, kl);
      sb = aie::mac(sb, delb, kh);
      aie::store_v(Sa, sa.template to_vector<float>());
      aie::store_v(Sb, sb.template to_vector<float>());
      Sa += kD;
      Sb += kD;
    }
    accf16 oa, ob;
    oa.from_vector(aie::load_v<kV>(o + ja));
    ob.from_vector(aie::load_v<kV>(o + jb));
    const float *__restrict Ra = S + ja;
    const float *__restrict Rb = S + jb;
#pragma clang loop min_iteration_count(DNX_ROWS)
    for (unsigned i = 0; i < kRowsX; ++i) {
      oa = mac_split(oa, aie::load_v<kV>(Ra), qh_r[i], ql_r[i]);
      ob = mac_split(ob, aie::load_v<kV>(Rb), qh_r[i], ql_r[i]);
      Ra += kD;
      Rb += kD;
    }
    aie::store_v(o + ja, oa.template to_vector<float>());
    aie::store_v(o + jb, ob.template to_vector<float>());
  }
}

// ---- pass 2, the half-row call (row i of the slice at S, half hf): the slice's first call does
// the whole update in place, then every call copies its updated half row out to ye.
static inline void dnx_row_half(float *__restrict S, float *__restrict ds, float *__restrict ye,
                                unsigned blk, unsigned i, unsigned hf) {
#ifdef LX_NULL_DN
  return;                                    // timing ablation: the streams stay, the maths goes
#endif
  if (i == 0 && hf == 0) dnx_slice_update(S, ds, blk);
  const float *__restrict src = S + i * kD + hf * kHalf;
#pragma clang loop unroll(full)
  for (unsigned j = 0; j < kHalf; j += kV) aie::store_v(ye + j, aie::load_v<kV>(src + j));
}

// ---- per head, half hf: ye = o[hf] / sqrt(128)
static inline void dnx_ofin_half(const float *__restrict ds, float *__restrict ye, unsigned hf) {
#ifdef LX_NULL_DN
  return;                                    // timing ablation: the streams stay, the maths goes
#endif
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
