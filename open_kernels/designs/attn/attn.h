#pragma once
//===- attn.h ----------------------------------------------*- C++ -*-===//
//
// Full-attention decode step (one token, one core), after the q/k/v (and gate) GEMVs:
//   q' = rope( rms_HD(q_h) * qn )     NH heads x HD   (qn = effective norm weight)
//   k' = rope( rms_HD(k_h) * kn )     KVH heads x HD, v as is; k', v -> bf16 cache rows
// ATTN_QKNORM_POST swaps the last two steps -- q' = rope( rms_HD(q_h) ) * qn -- which is
// HunYuan's order (query_layernorm AFTER apply_rotary_pos_emb). RoPE is orthogonal and the
// rotary dim is the whole head there, so the RMS is the same either way; what moves is the
// per-dim weight, which does not commute with the pair rotation.
//   for head h (kv head h / (NH/KVH)): s_t = q'_h . K_t / sqrt(HD) over t in [0, pos] (cache rows + new)
//   o_h = softmax(s) V  (online softmax, fp32 accumulators), og_h = o_h [* sigmoid(gate_h)]
// RoPE over the first ROT dims of each head, half-split pairs (i, i + ROT/2); cos/sin for
// position p come from the host in the position record (layout below).
// Reference: open_kernels/model/replica.py attn_decode.
//
// Compile-time knobs (the whole-layer designs pass them from the ModelSpec; the defaults are
// the Qwen3.6-27B point, recipes/catalogue.py's `attn` set): ATTN_NH, ATTN_KVH, ATTN_HD,
// ATTN_ROT, ATTN_GATE (1: a sigmoid output gate arrives with the q heads, as in Qwen3.5/3.6;
// 0: no gate, as in Qwen3 dense / Llama), ATTN_QKNORM, ATTN_QKNORM_POST, ATTN_EPS.
//
// Elements: the attention core's fifo element is ONE cache-row half, E_A = KVH * HD bf16
// bytes (1 KB for the 27B, 2 KB for Qwen3-4B). So a q / k / v / gate element of fp32 heads
// carries kHPE = KVH/2 heads, an og element kHPO = KVH heads, and a cache row is two elements
// (K_t, V_t). meta = two elements: [qn bf16[HD] @0 | kn @HD*2] (per layer) and the position
// record [int32 pos @0 | int32 nf @4 | cos f32[ROT/2] @512 | sin f32[ROT/2] @512 + 2*ROT]
// (ptab row pos; nf = the number of cache rows streamed). pb (int32[4]) = [pos, nf, rows seen]:
// attn_meta fills it, the core loops nf times, attn_step masks rows t >= pos (the whole-layer
// design streams one dummy row at position 0: a zero-length DMA is not an option).

#include "vecmath.h"

#ifndef ATTN_NH
#define ATTN_NH 16
#endif
#ifndef ATTN_KVH
#define ATTN_KVH 2
#endif
#ifndef ATTN_HD
#define ATTN_HD 256
#endif
#ifndef ATTN_ROT
#define ATTN_ROT 64
#endif
#ifndef ATTN_GATE
#define ATTN_GATE 1
#endif
#ifndef ATTN_QKNORM
#define ATTN_QKNORM 1        // 1: rms over the head * qn / kn (Qwen3); 0: RoPE only (Llama)
#endif
#ifndef ATTN_QKNORM_POST
#define ATTN_QKNORM_POST 0   // 1: the norm weight multiplies AFTER RoPE (HunYuan); 0: before (Qwen3)
#endif
#ifndef ATTN_EPS
#define ATTN_EPS 1e-6f       // the qk RMSNorm's epsilon (the model's rms_norm_eps)
#endif
#if ATTN_QKNORM_POST && !ATTN_QKNORM
#error "ATTN_QKNORM_POST needs ATTN_QKNORM"
#endif
#ifndef ATTN_VEXP
#define ATTN_VEXP 0          // 1: batch the online-softmax exponentials over heads (see attn_row_impl)
#endif
#ifndef ATTN_NHL
#define ATTN_NHL ATTN_NH     // heads THIS core owns; < ATTN_NH splits attention over several cores
#endif
static constexpr unsigned kNH = ATTN_NH;
static constexpr unsigned kNHL = ATTN_NHL;   // local head count; oacc / ml / og are indexed by it
static constexpr unsigned kKVH = ATTN_KVH;
static constexpr unsigned kHD = ATTN_HD;
static constexpr unsigned kRot = ATTN_ROT;
static constexpr unsigned kV = 32;
static constexpr unsigned kHPE = kKVH / 2;    // fp32 heads per element
static constexpr unsigned kHPO = kKVH;        // bf16 og heads per element
static_assert(kHD % kV == 0 && kRot % (2 * kV) == 0 && kRot <= kHD && kKVH % 2 == 0 && kNH % kKVH == 0 &&
              kNH % kHPO == 0 && kKVH % kHPE == 0,
              "attn.h: HD a multiple of 32, rotary dim a multiple of 64 within HD, an even kv-head count that divides NH");
static constexpr float kScale = kHD == 256 ? 0.0625f : kHD == 128 ? 0.08838834764831845f
                                : kHD == 64 ? 0.125f : 0.0f;   // 1/sqrt(HD); a new HD adds its constant here
static_assert(kScale > 0.0f, "attn.h: no 1/sqrt(HD) for this head dim");
// ml stride: [m: kMLS][l: kMLS]. With ATTN_VEXP the head count is rounded up to
// a whole vector so BOTH halves are aligned for a 32-float load; without it the
// halves are packed and ml is 2 * NH as before.
#if ATTN_VEXP
static constexpr unsigned kMLS = ((kNHL + kV - 1) / kV) * kV;
#else
static constexpr unsigned kMLS = kNHL;
#endif
static constexpr bool kSplit = (kNHL != kNH);   // compile-time: no h0 arithmetic on the single-core path
// The head offset is a kernel ARGUMENT only when attention is actually split.
// Leaving an unused one in the signature is not free: it perturbs codegen, and
// the single-core families must keep compiling byte-for-byte what they did.
#if ATTN_NHL == ATTN_NH
#define ATTN_H0_PARM
#define ATTN_H0_DECL const int h0 = 0; (void)h0;
#define ATTN_H0_ARG
#else
#define ATTN_H0_PARM , int h0
#define ATTN_H0_DECL
#define ATTN_H0_ARG , h0
#endif
static_assert(kNH % kNHL == 0 && kNHL % kHPO == 0,
              "attn.h: the local head count must divide NH and be a whole number of og elements");

static inline void attn_meta_impl(const uint8_t *__restrict m0, const uint8_t *__restrict m1,
                                  bfloat16 *__restrict qn, bfloat16 *__restrict kn,
                                  float *__restrict cs, int32_t *__restrict pb) {
  const bfloat16 *q = (const bfloat16 *)m0;
  for (unsigned j = 0; j < kHD; j += kV) aie::store_v(qn + j, aie::load_v<kV>(q + j));
  const bfloat16 *k = (const bfloat16 *)(m0 + kHD * 2);
  for (unsigned j = 0; j < kHD; j += kV) aie::store_v(kn + j, aie::load_v<kV>(k + j));
  const float *c = (const float *)(m1 + 512);
  for (unsigned j = 0; j < kRot; j += kV) aie::store_v(cs + j, aie::load_v<kV>(c + j));
  const int32_t *p = (const int32_t *)m1;
  pb[0] = p[0];
  pb[1] = p[1];
  pb[2] = 0;
  pb[3] = 0;
}

// x (fp32[HD]) -> [rms_HD * w] -> rope over [0, ROT) -> dst (fp32[HD]); with ATTN_QKNORM_POST
// the * w moves after the rotation.
__attribute__((noinline)) inline void norm_rope(const float *__restrict x, const bfloat16 *__restrict w,
                             const float *__restrict cs, float *__restrict dst) {
#if !ATTN_QKNORM
  for (unsigned j = 0; j < kHD; j += kV) aie::store_v(dst + j, aie::load_v<kV>(x + j));
#else
  accf32 ss = aie::zeros<accfloat, kV>();
  for (unsigned j = 0; j < kHD; j += kV) {
    v32b h, l;
    split32(aie::load_v<kV>(x + j), h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kHD) + ATTN_EPS);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
  for (unsigned j = 0; j < kHD; j += kV) {
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vs(t, aie::load_v<kV>(x + j), ih, il);
#if ATTN_QKNORM_POST
    aie::store_v(dst + j, t.template to_vector<float>());
#else
    accf32 u = aie::zeros<accfloat, kV>();
    u = mac_vv(u, t.template to_vector<float>(), aie::load_v<kV>(w + j));
    aie::store_v(dst + j, u.template to_vector<float>());
#endif
  }
#endif
  // rope on dims [0, ROT): pairs (a = dst[j], b = dst[j + ROT/2]); cs = [cos ROT/2 | sin ROT/2]
  for (unsigned j = 0; j < kRot / 2; j += kV) {
    const v32f c = aie::load_v<kV>(cs + j);
    const v32f s = aie::load_v<kV>(cs + kRot / 2 + j);
    const v32f a = aie::load_v<kV>(dst + j);
    const v32f b = aie::load_v<kV>(dst + kRot / 2 + j);
    aie::store_v(dst + j, fsub32(fmul32(a, c), fmul32(b, s)));
    aie::store_v(dst + kRot / 2 + j, fadd32(fmul32(b, c), fmul32(a, s)));
  }
#if ATTN_QKNORM_POST
  // the whole head, not just [0, ROT): the unrotated tail is scaled too
  for (unsigned j = 0; j < kHD; j += kV) {
    accf32 u = aie::zeros<accfloat, kV>();
    u = mac_vv(u, aie::load_v<kV>(dst + j), aie::load_v<kV>(w + j));
    aie::store_v(dst + j, u.template to_vector<float>());
  }
#endif
}

static inline void to_bf16_hd(const float *__restrict src, bfloat16 *__restrict dst) {
  for (unsigned j = 0; j < kHD; j += kV) {
    accf32 a;
    a.from_vector(aie::load_v<kV>(src + j));
    aie::store_v(dst + j, a.template to_vector<bfloat16>());
  }
}

// q element e (kHPE heads) -> qs[e*kHPE ..]
static inline void attn_q_impl(const float *__restrict qe, const bfloat16 *__restrict qn,
                               const float *__restrict cs, float *__restrict qs, int e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned i = 0; i < kHPE; ++i) norm_rope(qe + i * kHD, qn, cs, qs + (e * kHPE + i) * kHD);
}
// k element e -> bf16 kout (the cache row half); v element e -> bf16 vout
static inline void attn_k_impl(const float *__restrict ke, const bfloat16 *__restrict kn,
                               const float *__restrict cs, float *__restrict tmp,
                               bfloat16 *__restrict kout, int e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned i = 0; i < kHPE; ++i) {
    norm_rope(ke + i * kHD, kn, cs, tmp);
    to_bf16_hd(tmp, kout + (e * kHPE + i) * kHD);
  }
}
static inline void attn_v_impl(const float *__restrict ve, bfloat16 *__restrict vout, int e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned i = 0; i < kHPE; ++i) to_bf16_hd(ve + i * kHD, vout + (e * kHPE + i) * kHD);
}

static inline void attn_init_impl(float *__restrict oacc, float *__restrict ml) {
  for (unsigned j = 0; j < kNHL * kHD; j += kV) aie::store_v(oacc + j, aie::zeros<float, kV>());
  for (unsigned h = 0; h < kNHL; ++h) { ml[h] = -1e30f; ml[kMLS + h] = 0.f; }
#if ATTN_VEXP
  // The padding lanes are exponentiated with the rest; -1e30 keeps them at
  // exp(-inf) = 0 rather than whatever the stack held.
  for (unsigned h = kNHL; h < kMLS; ++h) { ml[h] = -1e30f; ml[kMLS + h] = 0.f; }
#endif
}

#if ATTN_VEXP
// one position, heads batched through the vector unit.
//
// The straight loop below spends two sexp() per head per position, and sexp is
// SOFTWARE FLOAT ON THE SCALAR UNIT -- vecmath.h says so where it is defined
// ("slow, use for a few dozen values per call"). At NH heads over a context of
// P positions that is 2 * NH * P of them per layer per token: 6.5 million for
// Granite at position 2048 over 40 layers, and it dominates the step.
//
// Same arithmetic, three phases: score every head, do the online-softmax
// update for all of them in one vector pass (vexpN, ~1e-7 relative, the same
// accuracy class as sexp), then accumulate the outputs. Two exponentials per
// vector of 32 heads instead of two per head.
//
// ml is [m: kMLS][l: kMLS] with kMLS the head count rounded up to a vector, so
// both halves stay aligned for a 32-float load: at NH 40 the l half would
// otherwise start 160 bytes in, and attn.h's own trap catalogue has that
// mistake in it already.
__attribute__((noinline)) inline void attn_row_impl(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt,
                                  const float *__restrict qs, float *__restrict oacc,
                                  float *__restrict ml ATTN_H0_PARM) {
  ATTN_H0_DECL
  aie::set_rounding(aie::rounding_mode::conv_even);
  alignas(128) float sv[kMLS], av[kMLS], bv[kMLS];

  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned h = kSplit ? ((unsigned)h0 + hl) : hl;   // folds away when this core owns them all                 // global head: q and the kv mapping
    const unsigned kvh = h / (kNH / kKVH);
    const float *q = qs + h * kHD;
    const bfloat16 *k = Kt + kvh * kHD;
    accf32 d = aie::zeros<accfloat, kV>();
    for (unsigned j = 0; j < kHD; j += kV)
      d = mac_vv(d, aie::load_v<kV>(q + j), aie::load_v<kV>(k + j));
    sv[hl] = aie::reduce_add(d.template to_vector<float>()) * kScale;   // / sqrt(HD)
  }
  for (unsigned h = kNHL; h < kMLS; ++h) sv[h] = -1e30f;

  for (unsigned v = 0; v < kMLS; v += kV) {
    const vfN<kV> s = aie::load_v<kV>(sv + v);
    const vfN<kV> m = aie::load_v<kV>(ml + v);
    const vfN<kV> mn = aie::max(m, s);
    const vfN<kV> a = vexpN<kV>(fsubN<kV>(m, mn));
    const vfN<kV> b = vexpN<kV>(fsubN<kV>(s, mn));
    const vfN<kV> l = aie::load_v<kV>(ml + kMLS + v);
    aie::store_v(ml + v, mn);
    aie::store_v(ml + kMLS + v, faddN<kV>(fmulN<kV>(l, a), b));
    aie::store_v(av + v, a);
    aie::store_v(bv + v, b);
  }

  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned kvh = (kSplit ? ((unsigned)h0 + hl) : hl) / (kNH / kKVH);
    const bfloat16 *v = Vt + kvh * kHD;
    const float a = av[hl], b = bv[hl];
    const bfloat16 bh = (bfloat16)b;
    const bfloat16 bl = (bfloat16)(b - (float)bh);
    float *o = oacc + hl * kHD;
    for (unsigned j = 0; j < kHD; j += kV) {
      accf32 acc;
      acc.from_vector(fscaleN<32>(aie::load_v<kV>(o + j), a));
      acc = aie::mac(acc, aie::load_v<kV>(v + j), bh);
      acc = aie::mac(acc, aie::load_v<kV>(v + j), bl);
      aie::store_v(o + j, acc.template to_vector<float>());
    }
  }
}
#else
// one position: K_t, V_t bf16[KVH * HD]
__attribute__((noinline)) inline void attn_row_impl(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt,
                                  const float *__restrict qs, float *__restrict oacc,
                                  float *__restrict ml ATTN_H0_PARM) {
  ATTN_H0_DECL
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned h = kSplit ? ((unsigned)h0 + hl) : hl;   // folds away when this core owns them all
    const unsigned kvh = h / (kNH / kKVH);
    const float *q = qs + h * kHD;
    const bfloat16 *k = Kt + kvh * kHD;
    const bfloat16 *v = Vt + kvh * kHD;
    accf32 d = aie::zeros<accfloat, kV>();
    for (unsigned j = 0; j < kHD; j += kV)
      d = mac_vv(d, aie::load_v<kV>(q + j), aie::load_v<kV>(k + j));
    const float s = aie::reduce_add(d.template to_vector<float>()) * kScale;   // / sqrt(HD)
    const float m_old = ml[hl];
    const float m_new = (s > m_old) ? s : m_old;
    const float a = sexp(m_old - m_new);
    const float b = sexp(s - m_new);
    ml[hl] = m_new;
    ml[kMLS + hl] = ml[kMLS + hl] * a + b;
    const bfloat16 bh = (bfloat16)b;
    const bfloat16 bl = (bfloat16)(b - (float)bh);
    float *o = oacc + hl * kHD;
    for (unsigned j = 0; j < kHD; j += kV) {
      accf32 acc;
      acc.from_vector(fscaleN<32>(aie::load_v<kV>(o + j), a));
      acc = aie::mac(acc, aie::load_v<kV>(v + j), bh);
      acc = aie::mac(acc, aie::load_v<kV>(v + j), bl);
      aie::store_v(o + j, acc.template to_vector<float>());
    }
  }
}
#endif

// cached row number pb[2] of the nf streamed: rows t >= pos are the position-0 dummy
static inline void attn_step_impl(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt,
                                  const float *__restrict qs, float *__restrict oacc,
                                  float *__restrict ml, int32_t *__restrict pb ATTN_H0_PARM) {
  const int32_t t = pb[2];
  pb[2] = t + 1;
  if (t >= pb[0]) return;
  attn_row_impl(Kt, Vt, qs, oacc, ml ATTN_H0_ARG);
}

// kHPO heads -> one output element: og = o/l [* sigmoid(gate)]; with the gate, g0 holds the
// first kHPE of those heads' gates and g1 the next kHPE (kHPO == 2 * kHPE).
__attribute__((noinline)) inline void attn_fin_impl(const float *__restrict oacc, const float *__restrict ml,
                                 const float *__restrict g0, const float *__restrict g1,
                                 bfloat16 *__restrict og, int hp) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned i = 0; i < kHPO; ++i) {
    const unsigned h = kHPO * hp + i;
    const float inv = 1.0f / ml[kMLS + h];
    const float *o = oacc + h * kHD;
#if ATTN_GATE
    const float *g = (i < kHPE) ? (g0 + i * kHD) : (g1 + (i - kHPE) * kHD);
#endif
    for (unsigned j = 0; j < kHD; j += kV) {
      const v32f on = fscaleN<32>(aie::load_v<kV>(o + j), inv);
#if ATTN_GATE
      const v32f r = fmul32(on, vsigmoidN<32>(aie::load_v<kV>(g + j)));
#else
      const v32f r = on;
#endif
      accf32 a;
      a.from_vector(r);
      aie::store_v(og + i * kHD + j, a.template to_vector<bfloat16>());
    }
  }
}
