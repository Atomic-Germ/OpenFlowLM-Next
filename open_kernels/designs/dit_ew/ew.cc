//===- ew.cc -----------------------------------------------------*- C++ -*-===//
//
// dit_ew's kernels (dit_ew.py): the row-wise and elementwise ops of a diffusion
// transformer step, one row element (EW_EL bf16) per call, round-to-nearest-even.
//
//   ln_mod_row     y = LN(x) * (1 + scale) + shift                   (FLUX.2 norm1/norm2/norm_out)
//                  or, R_NORM = 1, y = RMSNorm(x) * w                  (Qwen3 input/post-attn norm)
//   res_ln_mod_row x' = x + gate * y (R_UNIT_GATE: x + y);  z = the norm of x', as above
//   qk_row         per-head RMSNorm (weights wq / wk), then RoPE: FLUX.2's 4-axis interleaved
//                  pairs, or (R_ROPE = 1) Qwen3's 1-D rotate-half, theta 1e6
//   bin_row        swiglu: silu(a) * b     euler: a + dt * b
//   un_row         silu
//
// Arithmetic: bf16 x bf16 products into fp32 accumulators, 32 lanes. aie2p has no
// native fp32 vector multiply (it is emulated, and a first fp32 version of these
// kernels ran 4-6x slower than their DMA), so a float scalar that multiplies a vector
// (rstd, dt, cos/sin) is split into bf16 hi + lo and applied as two MACs -- exact to
// fp32 rounding. Values are rounded to bf16 where diffusers' bf16 pipeline rounds
// them: the LayerNorm / RMSNorm output before the modulation / weight, the residual
// stream, 1 + scale.
//
// LN and RMSNorm run over the first W columns (a runtime parameter); columns W..EL of
// a norm's output are written as zeros. Per-dispatch parameter vectors (the modulation
// shift/scale/gate, the qk weights and RoPE tables, Euler's dt) arrive ahead of the
// rows and are copied into `par` by load_param.
//
// Accuracy notes (utilities/aie-probes/vecmath_probe.py, exp2_probe.py): aie2p's
// vector tanh is an approximation (1.4-3.5% rms below |x| = 2). SiLU is written
// 0.5x(1 + tanh(x/2)), where the leading 1 keeps tanh's error under ~0.5% of the
// result.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef EW_EL
#define EW_EL 3072 // bf16 per row element (FLUX.2 [klein] hidden size)
#endif
#ifndef EW_NPAR
#define EW_NPAR 3 // parameter elements a core keeps
#endif
constexpr int EL = EW_EL;
constexpr int HD = 128; // head dim
constexpr float EPS = 1e-6f;

// Runtime parameter slots (dit_ew.py RTP_*).
enum {
  R_NPAR, R_CNT_LN, R_CNT_RES, R_CNT_QK, R_CNT_BIN, R_CNT_UN, R_W, R_IDX_GATE,
  R_IDX_SHIFT, R_IDX_SCALE, R_BINOP, R_ROW0, R_ROW_STEP, R_N_TXT, R_GRID_W, R_HEADS,
  R_NORM, R_UNIT_GATE, R_ROPE, R_B_QH, R_B_KH
};
enum { BIN_SWIGLU = 1, BIN_EULER = 2 };

// qk parameter block (make_test.py qk_params / qwen_qk_params): element 0 = wq[128],
// wk[128] (bf16); elements 1-2 = fp32 RoPE tables, as raw bits.
//   FLUX.2: FINE[p] = cos[16], sin[16] of p*f_j (p < 64), then COARSE[a] of 64a*f_j (a < 8).
//   Qwen3:  T64[a], T8[b], T1[c] = cos[64], sin[64] of (64a | 8b | c)*f_j (a, b, c < 8):
//           position 64a + 8b + c by two angle additions.
constexpr int ROPE_FINE = 0, ROPE_COARSE = 64 * 32;
constexpr int QROPE_T64 = 0, QROPE_T8 = 8 * 128, QROPE_T1 = 16 * 128;

using VB = aie::vector<bfloat16, 32>;
using AC = aie::accum<accfloat, 32>;
using VF16 = aie::vector<float, 16>;

struct Split { bfloat16 hi, lo; };
static inline Split split(float f) {
  bfloat16 hi = (bfloat16)f;
  return {hi, (bfloat16)(f - float(hi))};
}
static inline AC acc_of(VB v) {
  AC a;
  a.from_vector(v);
  return a;
}
static inline AC acc_bcast(float f) {
  AC a;
  a.from_vector(aie::broadcast<float, 32>(f));
  return a;
}
// a + v * s, s a float split into bf16 hi + lo
static inline AC mac_s(AC a, VB v, Split s) {
  return aie::mac(aie::mac(a, v, s.hi), v, s.lo);
}

// Sum and sum of squares of x[0:W] (bf16) in fp32; LayerNorm mean and 1/std, or for
// RMSNorm (rms) mean 0 and 1/rms.
static inline void ln_stats(const bfloat16 *__restrict x, int W, float &mean, float &rstd,
                            bool rms) {
  AC s = aie::zeros<accfloat, 32>(), q = aie::zeros<accfloat, 32>();
  for (int i = 0; i < W; i += 32) {
    VB v = aie::load_v<32>(x + i);
    s = aie::add(s, v);
    q = aie::mac(q, v, v);
  }
  const float inv_w = 1.0f / float(W);
  mean = rms ? 0.0f : aie::reduce_add(s.to_vector<float>()) * inv_w;
  float var = aie::reduce_add(q.to_vector<float>()) * inv_w - mean * mean;
  rstd = aie::invsqrt((var > 0.0f ? var : 0.0f) + EPS);
}

// y[0:W] = bf16((x - mean) * rstd) * onep + shift; y[W:EL] = 0. onep = bf16(1 + scale),
// or the RMSNorm weight with shift = null (no shift).
static inline void ln_apply(const bfloat16 *__restrict x, bfloat16 *__restrict y,
                            const bfloat16 *__restrict onep,
                            const bfloat16 *__restrict shift, int W, float mean,
                            float rstd) {
  const Split r = split(rstd);
  const AC off = acc_bcast(-mean * rstd);
  for (int i = 0; i < W; i += 32) {
    VB t = mac_s(off, aie::load_v<32>(x + i), r).to_vector<bfloat16>();
    AC o = shift ? aie::mac(acc_of(aie::load_v<32>(shift + i)), t, aie::load_v<32>(onep + i))
                 : aie::mul(t, aie::load_v<32>(onep + i));
    aie::store_v(y + i, o.to_vector<bfloat16>());
  }
  const VB z = aie::zeros<bfloat16, 32>();
  for (int i = W; i < EL; i += 32)
    aie::store_v(y + i, z);
}

// silu(x) = h (1 + tanh h), h = x / 2 (exact in bf16), in fp32, rounded once.
static inline VB silu32(VB x) {
  VB h = aie::mul(x, (bfloat16)0.5f).to_vector<bfloat16>();
  AC ha = acc_of(h);
  aie::vector<float, 32> hf = ha.to_vector<float>();
  VB th;
  th.insert(0, aie::tanh<bfloat16>(hf.extract<16>(0)));
  th.insert(1, aie::tanh<bfloat16>(hf.extract<16>(1)));
  return aie::mac(ha, h, th).to_vector<bfloat16>();
}

// Interleaved-pair swap: [x0, x1, x2, x3, ...] -> [x1, x0, x3, x2, ...].
static inline VB pair_swap(VB v) {
  auto u = aie::interleave_unzip(v, v, 1);
  return aie::interleave_zip(u.second, u.first, 1).first;
}

// Per-dim RoPE factors of the current token as bf16 hi + lo; sign baked into sin so
// that out = y*cos + pair_swap(y)*sin.
alignas(64) static bfloat16 s_chi[HD], s_clo[HD], s_shi[HD], s_slo[HD];
alignas(64) static float s_c[HD], s_s[HD];

// One 32-dim RoPE axis block: dims 32*axis + 2j, +1 rotate by (c_j, s_j) (fp32 16).
static inline void rope_axis(int axis, VF16 c, VF16 s) {
  const VF16 ns = aie::sub(aie::zeros<float, 16>(), s);
  auto cc = aie::interleave_zip(c, c, 1);
  auto ss = aie::interleave_zip(ns, s, 1);
  aie::store_v(s_c + 32 * axis, cc.first);
  aie::store_v(s_c + 32 * axis + 16, cc.second);
  aie::store_v(s_s + 32 * axis, ss.first);
  aie::store_v(s_s + 32 * axis + 16, ss.second);
}

static inline VF16 mulf(VF16 a, VF16 b) { return aie::mul(a, b).template to_vector<float>(); }

// RoPE factors for global token g of the joint [text; image] sequence: text token l:
// axis 3 (dims 96-127) by l; image token k: axis 1 by k / grid_w, axis 2 by k % grid_w.
// Other axes are the identity. An edit appends one reference image of the same grid
// (klein's [text | generated | reference]): image token k >= grid_w^2 is reference token
// k - grid_w^2, whose axis 0 turns by REF_T (diffusers' t = 10 for the first reference).
// Once per token; the fp32 work here is 128 values.
constexpr int REF_T = 10;
static __attribute__((noinline)) void rope_token(const float *__restrict tab, int g, int n_txt, int grid_w) {
  const VF16 one = aie::broadcast<float, 16>(1.0f), zero = aie::zeros<float, 16>();
  for (int a = 0; a < 4; a++)
    rope_axis(a, one, zero);
  if (g < n_txt) {
    const int hi = g >> 6, lo = g & 63;
    const float *ph = tab + ROPE_COARSE + hi * 32, *pl = tab + ROPE_FINE + lo * 32;
    VF16 ca = aie::load_v<16>(ph), sa = aie::load_v<16>(ph + 16);
    VF16 cb = aie::load_v<16>(pl), sb = aie::load_v<16>(pl + 16);
    // cos(a+b) = ca cb - sa sb;  sin(a+b) = sa cb + ca sb
    rope_axis(3, aie::sub(mulf(ca, cb), mulf(sa, sb)), aie::add(mulf(sa, cb), mulf(ca, sb)));
  } else {
    int k = g - n_txt;
    if (k >= grid_w * grid_w) {
      k -= grid_w * grid_w;
      const float *pt = tab + ROPE_FINE + REF_T * 32;
      rope_axis(0, aie::load_v<16>(pt), aie::load_v<16>(pt + 16));
    }
    const int h = k / grid_w, w = k - h * grid_w;
    const float *ph = tab + ROPE_FINE + h * 32, *pw = tab + ROPE_FINE + w * 32;
    rope_axis(1, aie::load_v<16>(ph), aie::load_v<16>(ph + 16));
    rope_axis(2, aie::load_v<16>(pw), aie::load_v<16>(pw + 16));
  }
  // bf16 hi (vector), then lo = fp32 - hi rounded to bf16 (scalar, 2 x 128 per token)
  for (int i = 0; i < HD; i += 16) {
    aie::accum<accfloat, 16> c, sn;
    c.from_vector(aie::load_v<16>(s_c + i));
    sn.from_vector(aie::load_v<16>(s_s + i));
    aie::store_v(s_chi + i, c.template to_vector<bfloat16>());
    aie::store_v(s_shi + i, sn.template to_vector<bfloat16>());
  }
  for (int i = 0; i < HD; i++) {
    s_clo[i] = (bfloat16)(s_c[i] - float(s_chi[i]));
    s_slo[i] = (bfloat16)(s_s[i] - float(s_shi[i]));
  }
}

// (The qk helpers are noinline: qk_row calls each up to three times, and inlined copies
// overflow the core's 16 KB program memory.)
// Qwen3 rotate-half RoPE factors for position g: cos/sin of g*f_j, j < 64, into
// s_chi/s_clo (cos) and s_shi/s_slo (sin), dims 0..63 (the pair (j, j+64) shares them).
static __attribute__((noinline)) void qwen_rope_token(const float *__restrict tab, int g) {
  const float *pa = tab + QROPE_T64 + (g >> 6) * 128;
  const float *pb = tab + QROPE_T8 + ((g >> 3) & 7) * 128;
  const float *pc = tab + QROPE_T1 + (g & 7) * 128;
  for (int i = 0; i < 64; i += 16) {
    VF16 ca = aie::load_v<16>(pa + i), sa = aie::load_v<16>(pa + 64 + i);
    VF16 cb = aie::load_v<16>(pb + i), sb = aie::load_v<16>(pb + 64 + i);
    VF16 c1 = aie::sub(mulf(ca, cb), mulf(sa, sb)), s1 = aie::add(mulf(sa, cb), mulf(ca, sb));
    VF16 cc = aie::load_v<16>(pc + i), sc = aie::load_v<16>(pc + 64 + i);
    aie::store_v(s_c + i, aie::sub(mulf(c1, cc), mulf(s1, sc)));
    aie::store_v(s_s + i, aie::add(mulf(s1, cc), mulf(c1, sc)));
  }
  for (int i = 0; i < 64; i += 16) {
    aie::accum<accfloat, 16> c, sn;
    c.from_vector(aie::load_v<16>(s_c + i));
    sn.from_vector(aie::load_v<16>(s_s + i));
    aie::store_v(s_chi + i, c.template to_vector<bfloat16>());
    aie::store_v(s_shi + i, sn.template to_vector<bfloat16>());
  }
  for (int i = 0; i < 64; i++) {
    s_clo[i] = (bfloat16)(s_c[i] - float(s_chi[i]));
    s_slo[i] = (bfloat16)(s_s[i] - float(s_shi[i]));
  }
}

// RMSNorm (weight w) then Qwen3 rotate-half RoPE of one row of `heads` heads:
// out[j] = t[j] c_j - t[j+64] s_j,  out[j+64] = t[j+64] c_j + t[j] s_j.
static __attribute__((noinline)) void qk_heads_half(const bfloat16 *__restrict x, bfloat16 *__restrict y,
                                 const bfloat16 *__restrict w, int heads) {
  for (int h = 0; h < heads; h++) {
    const bfloat16 *__restrict xh = x + h * HD;
    bfloat16 *__restrict yh = y + h * HD;
    AC q = aie::zeros<accfloat, 32>();
    for (int i = 0; i < HD; i += 32) {
      VB v = aie::load_v<32>(xh + i);
      q = aie::mac(q, v, v);
    }
    const Split rs = split(aie::invsqrt(aie::reduce_add(q.to_vector<float>()) *
                                            (1.0f / HD) + EPS));
    for (int i = 0; i < 64; i += 32) {
      VB lo = aie::mul(mac_s(aie::zeros<accfloat, 32>(), aie::load_v<32>(xh + i), rs)
                           .to_vector<bfloat16>(), aie::load_v<32>(w + i)).to_vector<bfloat16>();
      VB hi = aie::mul(mac_s(aie::zeros<accfloat, 32>(), aie::load_v<32>(xh + 64 + i), rs)
                           .to_vector<bfloat16>(), aie::load_v<32>(w + 64 + i)).to_vector<bfloat16>();
      VB ch = aie::load_v<32>(s_chi + i), cl = aie::load_v<32>(s_clo + i);
      VB sh = aie::load_v<32>(s_shi + i), sl = aie::load_v<32>(s_slo + i);
      AC o_lo = aie::mac(aie::mul(lo, ch), lo, cl);
      o_lo = aie::msc(aie::msc(o_lo, hi, sh), hi, sl);
      AC o_hi = aie::mac(aie::mul(hi, ch), hi, cl);
      o_hi = aie::mac(aie::mac(o_hi, lo, sh), lo, sl);
      aie::store_v(yh + i, o_lo.to_vector<bfloat16>());
      aie::store_v(yh + 64 + i, o_hi.to_vector<bfloat16>());
    }
  }
}

static inline void copy_heads(const bfloat16 *__restrict x, bfloat16 *__restrict y, int heads) {
  for (int i = 0; i < heads * HD; i += 32)
    aie::store_v(y + i, aie::load_v<32>(x + i));
}

// RMSNorm (weight w) then RoPE of one row of `heads` heads.
static __attribute__((noinline)) void qk_heads(const bfloat16 *__restrict x, bfloat16 *__restrict y,
                            const bfloat16 *__restrict w, int heads) {
  for (int h = 0; h < heads; h++) {
    const bfloat16 *__restrict xh = x + h * HD;
    bfloat16 *__restrict yh = y + h * HD;
    AC q = aie::zeros<accfloat, 32>();
    for (int i = 0; i < HD; i += 32) {
      VB v = aie::load_v<32>(xh + i);
      q = aie::mac(q, v, v);
    }
    const Split rs = split(aie::invsqrt(aie::reduce_add(q.to_vector<float>()) *
                                            (1.0f / HD) + EPS));
    for (int i = 0; i < HD; i += 32) {
      VB u = mac_s(aie::zeros<accfloat, 32>(), aie::load_v<32>(xh + i), rs)
                 .to_vector<bfloat16>();
      VB t = aie::mul(u, aie::load_v<32>(w + i)).to_vector<bfloat16>();
      VB ts = pair_swap(t);
      AC o = aie::mul(t, aie::load_v<32>(s_chi + i));
      o = aie::mac(o, t, aie::load_v<32>(s_clo + i));
      o = aie::mac(o, ts, aie::load_v<32>(s_shi + i));
      o = aie::mac(o, ts, aie::load_v<32>(s_slo + i));
      aie::store_v(yh + i, o.to_vector<bfloat16>());
    }
  }
}

static inline void zero_tail(bfloat16 *__restrict y, int from) {
  const VB z = aie::zeros<bfloat16, 32>();
  for (int i = from; i < EL; i += 32)
    aie::store_v(y + i, z);
}

extern "C" {

#ifdef EW_NOP // timing builds only (DE_NOP=1): every row kernel returns at once
#define SET_ROUNDING() return
#else
#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)
#endif

// par[slot] = e, slot = i + shift, dropped outside [0, EW_NPAR): the i-th parameter
// element a core receives (dit_ew.py streams p[-1], p[0], p[0], .., p[n] to a pair).
// For the LN ops the scale vector is stored as bf16(1 + scale), as diffusers rounds it.
void load_param(bfloat16 *__restrict e, bfloat16 *__restrict par, int32_t i, int32_t shift,
                int32_t *__restrict rtp) {
  SET_ROUNDING();
  const int slot = i + shift;
  if (slot < 0 || slot >= EW_NPAR)
    return;
  bfloat16 *__restrict d = par + slot * EL;
  const bool onep = (rtp[R_CNT_LN] | rtp[R_CNT_RES]) && !rtp[R_NORM] &&
                    slot == rtp[R_IDX_SCALE];
  const VB one = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
  for (int k = 0; k < EL; k += 32) {
    VB v = aie::load_v<32>(e + k);
    aie::store_v(d + k, onep ? aie::add(v, one) : v);
  }
}

void ln_mod_row(bfloat16 *__restrict x, bfloat16 *__restrict y, bfloat16 *__restrict par,
                int32_t *__restrict rtp) {
  SET_ROUNDING();
  const int W = rtp[R_W];
  const bool rms = rtp[R_NORM];
  float mean, rstd;
  ln_stats(x, W, mean, rstd, rms);
  ln_apply(x, y, par + rtp[R_IDX_SCALE] * EL, rms ? nullptr : par + rtp[R_IDX_SHIFT] * EL, W,
           mean, rstd);
}

// xo = bf16(x + gate * yb) (the residual stream); z = LN(xo)(1 + scale) + shift.
void res_ln_mod_row(bfloat16 *__restrict x, bfloat16 *__restrict yb, bfloat16 *__restrict z,
                    bfloat16 *__restrict xo, bfloat16 *__restrict par,
                    int32_t *__restrict rtp) {
  SET_ROUNDING();
  const int W = rtp[R_W];
  const bool rms = rtp[R_NORM];
  const bfloat16 *__restrict g = par + rtp[R_IDX_GATE] * EL;
  const VB one = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
  const bool unit = rtp[R_UNIT_GATE];
  AC s = aie::zeros<accfloat, 32>(), q = aie::zeros<accfloat, 32>();
  for (int i = 0; i < W; i += 32) {
    VB r = aie::mac(acc_of(aie::load_v<32>(x + i)), unit ? one : aie::load_v<32>(g + i),
                    aie::load_v<32>(yb + i)).to_vector<bfloat16>();
    aie::store_v(xo + i, r);
    s = aie::add(s, r);
    q = aie::mac(q, r, r);
  }
  const VB zz = aie::zeros<bfloat16, 32>();
  for (int i = W; i < EL; i += 32)
    aie::store_v(xo + i, zz);
  const float inv_w = 1.0f / float(W);
  const float mean = rms ? 0.0f : aie::reduce_add(s.to_vector<float>()) * inv_w;
  const float var = aie::reduce_add(q.to_vector<float>()) * inv_w - mean * mean;
  const float rstd = aie::invsqrt((var > 0.0f ? var : 0.0f) + EPS);
  ln_apply(xo, z, par + rtp[R_IDX_SCALE] * EL, rms ? nullptr : par + rtp[R_IDX_SHIFT] * EL, W,
           mean, rstd);
}

// Two row elements of token rtp[ROW0] + t * rtp[ROW_STEP], RMSNorm per head then RoPE.
// Element A: rtp[HEADS] heads, all q (weight wq). Element B: rtp[B_QH] q heads, then
// rtp[B_KH] k heads (weight wk), the rest of its 24 copied through (Qwen3's v heads
// share the fused q|k|v row). FLUX.2: A = q, B = k (B_QH 0, B_KH 24).
void qk_row(bfloat16 *__restrict a, bfloat16 *__restrict b, bfloat16 *__restrict ao,
            bfloat16 *__restrict bo, bfloat16 *__restrict par, int32_t *__restrict rtp,
            int32_t t) {
  SET_ROUNDING();
  const int g = rtp[R_ROW0] + t * rtp[R_ROW_STEP];
  const float *tab = (const float *)(par + EL);
  const int ha = rtp[R_HEADS], bq = rtp[R_B_QH], bk = rtp[R_B_KH];
  if (rtp[R_ROPE]) {
    qwen_rope_token(tab, g);
    qk_heads_half(a, ao, par, ha);
    qk_heads_half(b, bo, par, bq);
    qk_heads_half(b + bq * HD, bo + bq * HD, par + HD, bk);
  } else {
    rope_token(tab, g, rtp[R_N_TXT], rtp[R_GRID_W]);
    qk_heads(a, ao, par, ha);
    qk_heads(b, bo, par, bq);
    qk_heads(b + bq * HD, bo + bq * HD, par + HD, bk);
  }
  zero_tail(ao, ha * HD);
  copy_heads(b + (bq + bk) * HD, bo + (bq + bk) * HD, EL / HD - bq - bk);
}

void bin_row(bfloat16 *__restrict a, bfloat16 *__restrict b, bfloat16 *__restrict y,
             bfloat16 *__restrict par, int32_t *__restrict rtp) {
  SET_ROUNDING();
  if (rtp[R_BINOP] == BIN_SWIGLU) {
    for (int i = 0; i < EL; i += 32)
      aie::store_v(y + i, aie::mul(silu32(aie::load_v<32>(a + i)), aie::load_v<32>(b + i))
                              .to_vector<bfloat16>());
  } else { // BIN_EULER: dt is par[0:2] as an fp32
    const Split dt = split(*(const float *)par);
    for (int i = 0; i < EL; i += 32)
      aie::store_v(y + i, mac_s(acc_of(aie::load_v<32>(a + i)), aie::load_v<32>(b + i), dt)
                              .to_vector<bfloat16>());
  }
}

void un_row(bfloat16 *__restrict a, bfloat16 *__restrict y, int32_t *__restrict rtp) {
  SET_ROUNDING();
  for (int i = 0; i < EL; i += 32)
    aie::store_v(y + i, silu32(aie::load_v<32>(a + i)));
}

} // extern "C"
