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
#ifndef ATTN_NULL
#define ATTN_NULL 0          // PROBE ONLY: consume the cached rows without computing. Wrong answers,
#endif                       // but it prices the fifo traffic against the arithmetic on top of it.
#ifndef ATTN_ABL
#define ATTN_ABL 3           // PROBE ONLY: 1 = score only, 2 = score + softmax, 3 = the real thing.
#endif

#ifndef ATTN_RB
#define ATTN_RB 1            // cached rows per kernel call (see attn_rowb_impl); 1 = one row, as before
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
static constexpr unsigned kQW = kNH * kHD;   // q, stored PRE-SPLIT: [hi bf16[QW] | lo bf16[QW]]
// ATTN_VEXP stores q as that bf16 pair; without it q stays the fp32 vector it always was,
// so a family that does not take this path compiles byte-for-byte what it compiled before.
#if ATTN_VEXP
#define ATTN_QT bfloat16
#else
#define ATTN_QT float
#endif
static constexpr unsigned kHPE = kKVH / 2;    // fp32 heads per element
static constexpr unsigned kHPO = kKVH;        // bf16 og heads per element
static_assert(kHD % kV == 0 && kRot % kV == 0 && kRot <= kHD && kKVH % 2 == 0 && kNH % kKVH == 0 &&
              kNH % kHPO == 0 && kKVH % kHPE == 0,
              "attn.h: HD a multiple of 32, rotary dim a multiple of 32 within HD, an even kv-head count that divides NH");
static constexpr float kScale = kHD == 256 ? 0.0625f : kHD == 128 ? 0.08838834764831845f
                                : kHD == 64 ? 0.125f : 0.0f;   // 1/sqrt(HD); a new HD adds its constant here
static_assert(kScale > 0.0f, "attn.h: no 1/sqrt(HD) for this head dim");
// Where 1/sqrt(HD) goes on the ATTN_VEXP path. At HD 64 and 256 it is a power of two, so
// scaling each bf16 half of q is an exponent shift and the scores come out bit-identical
// (attn_q_impl). At HD 128 it is 2^-3.5: folding it into q would round both halves, so
// there it multiplies the SCORES instead -- one fmulN per position per vector of heads
// (three bf16 macs, ~2^-16 relative), on the vector unit, where the scalar version of
// that same multiply was the 25 ms the fold removed. The HD 64 / 256 builds contain
// neither branch of that #if and compile byte-for-byte what they did.
#define ATTN_SCALE_IN_Q (ATTN_HD == 64 || ATTN_HD == 256)
// ml stride: [m: kMLS][l: kMLS]. With ATTN_VEXP the head count is rounded up to
// a whole vector so BOTH halves are aligned for a 32-float load; without it the
// halves are packed and ml is 2 * NH as before.
#if ATTN_VEXP
static constexpr unsigned kMLS = ((kNHL + kV - 1) / kV) * kV;
#else
static constexpr unsigned kMLS = kNHL;
#endif
// The exponential is batched over heads, so its vector is as wide as the head
// count -- NOT as wide as kV. A core owning 8 heads ran vexpN<32> and threw
// three quarters of the lanes away; fp32 is 8 lanes native here, so those
// quarters are real cycles. kMLS (the ml STRIDE) stays at kV so the l half
// keeps its 128-byte alignment; only the number of lanes touched narrows.
static constexpr unsigned kVE = kNHL >= 32 ? 32 : (kNHL >= 16 ? 16 : 8);
static constexpr unsigned kMLU = ((kNHL + kVE - 1) / kVE) * kVE;
static constexpr unsigned kRB = ATTN_RB;      // cached rows per call
// The block kernel's per-head vectors: kNHL lanes when that is a whole fp32 vector, else
// padded to 8, the narrowest one (a core owning 2 or 4 heads: Gemma3-4B, the 35B, the
// Qwen3.5 sizes). The padding lanes carry -1e30 like attn_row_impl's kMLU ones. At 8+
// heads per core kNL == kNHL and the kernel is byte for byte what it was.
static constexpr unsigned kNL = kNHL >= 8 ? kNHL : 8;
// The head-dim loops of the vector-softmax paths (kHD / 32 iterations: 2, 4, 8) unroll
// fully at HD <= 128. At HD 256 the fully unrolled score and V loops overflowed the core's
// 16 KB of program memory (Gemma3-4B, 2026-09-08: "Overflow of program memory", at RB 4
// and at RB 2), so there they unroll by four. HD 64 / 128 compile what they compiled.
#if ATTN_HD <= 128
#define ATTN_UNROLL_HD AIE_LOOP_UNROLL_FULL
#else
#define ATTN_UNROLL_HD _Pragma("clang loop unroll_count(4)")
#endif
static constexpr unsigned kPV = kRB * kNL;    // the block's score vector: one exp covers it all
// Heads in one og element. Until attention could be split more finely than the og
// element, ACORES was the largest divisor of NH/HPO, so NHL was ALWAYS exactly HPO and
// this was kHPO by construction. A core may now own fewer heads than an og element
// holds; it then writes just its own, and the drain -- already sized NHL * HD * 2 --
// takes exactly those bytes. At kNHL >= kHPO this is kHPO and every family that had
// NHL == HPO compiles what it compiled.
static constexpr unsigned kOGH = kNHL < kHPO ? kNHL : kHPO;
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
// kNHL % kHPO was required while a core had to own WHOLE og elements. It now owns
// kOGH = min(kNHL, kHPO) heads per element, so the requirement is the weaker one that
// its heads tile the element evenly -- which holds trivially when kOGH == kNHL.
static_assert(kNH % kNHL == 0 && kNHL % kOGH == 0,
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
#if ATTN_RB > 1
  // How the nf streamed rows divide into whole blocks and a remainder. When the
  // window carries a dummy row -- only at position 0, where nf is 1 and pos is 0 --
  // every row goes down the one-at-a-time path, which is the only one that masks.
  const int32_t real = (p[0] >= p[1]) ? p[1] : 0;
  pb[4] = real / (int32_t)kRB;
  pb[5] = p[1] - pb[4] * (int32_t)kRB;
#endif
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
  // rope on dims [0, ROT): pairs (a = dst[j], b = dst[j + ROT/2]); cs = [cos ROT/2 | sin ROT/2].
  // 32 pairs a step, then a 16-lane tail when ROT/2 is not a multiple of 32 (Phi-3 rotates
  // 96 of 128 dims: 48 pairs).
  constexpr unsigned kHalf = kRot / 2, kHalfV = kHalf - kHalf % kV;
  for (unsigned j = 0; j < kHalfV; j += kV) {
    const v32f c = aie::load_v<kV>(cs + j);
    const v32f s = aie::load_v<kV>(cs + kHalf + j);
    const v32f a = aie::load_v<kV>(dst + j);
    const v32f b = aie::load_v<kV>(dst + kHalf + j);
    aie::store_v(dst + j, fsub32(fmul32(a, c), fmul32(b, s)));
    aie::store_v(dst + kHalf + j, fadd32(fmul32(b, c), fmul32(a, s)));
  }
#if (ATTN_ROT / 2) % 32
  {
    const v16f c = aie::load_v<16>(cs + kHalfV);
    const v16f s = aie::load_v<16>(cs + kHalf + kHalfV);
    const v16f a = aie::load_v<16>(dst + kHalfV);
    const v16f b = aie::load_v<16>(dst + kHalf + kHalfV);
    aie::store_v(dst + kHalfV, fsubN<16>(fmulN<16>(a, c), fmulN<16>(b, s)));
    aie::store_v(dst + kHalf + kHalfV, faddN<16>(fmulN<16>(b, c), fmulN<16>(a, s)));
  }
#endif
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

// q element e (kHPE heads) -> qs[e*kHPE ..], as the bf16 (hi, lo) pair the scores want.
//
// The dot product is q(fp32) . k(bf16), and mac_vv's way to do that is to split q into two
// bf16 halves and mac twice. q does not change over the context, so that split was being
// recomputed for every cached row: NH * P times per token instead of NH. Hoisting it here
// leaves the inner loop two macs and no split, and the arithmetic is bit-identical.
static inline void attn_q_impl(const float *__restrict qe, const bfloat16 *__restrict qn,
                               const float *__restrict cs, ATTN_QT *__restrict qs, int e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#if !ATTN_VEXP
  for (unsigned i = 0; i < kHPE; ++i) norm_rope(qe + i * kHD, qn, cs, qs + ((unsigned)e * kHPE + i) * kHD);
#else
  alignas(128) float t[kHD];
  for (unsigned i = 0; i < kHPE; ++i) {
    norm_rope(qe + i * kHD, qn, cs, t);
    bfloat16 *qh = qs + ((unsigned)e * kHPE + i) * kHD;
    for (unsigned j = 0; j < kHD; j += kV) {
      vbN<kV> h, l;
      splitN<kV>(aie::load_v<kV>(t + j), h, l);
#if ATTN_SCALE_IN_Q
      // ...and carry 1/sqrt(HD) here too. It used to be a scalar float multiply on
      // every head's score at every position, and scalar float on this core is a
      // software call: that one operation measured 25 ms of a 185 ms decode step at
      // position 2048. kScale is a power of two at this head dim, so scaling each
      // bf16 half is a pure exponent shift and the scores come out bit-identical.
      // (At HD 128 it is not, and the scores are scaled instead: ATTN_SCALE_IN_Q.)
      h = aie::mul(h, (bfloat16)kScale).template to_vector<bfloat16>();
      l = aie::mul(l, (bfloat16)kScale).template to_vector<bfloat16>();
#endif
      aie::store_v(qh + j, h);
      aie::store_v(qh + kQW + j, l);
    }
  }
#endif
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
#if ATTN_VEXP
  aie::set_rounding(aie::rounding_mode::conv_even);   // for every row of this token; the
#endif                                                // non-VEXP row kernel still sets its own
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
                                  const ATTN_QT *__restrict qs, float *__restrict oacc,
                                  float *__restrict ml ATTN_H0_PARM) {
  ATTN_H0_DECL
  // No set_rounding here: it is core-wide state, attn_init_impl sets it once per
  // token, and nothing between that and the last row touches it. It was being
  // written once per position per layer -- 82k times a token, for one value.
  alignas(128) float sv[kMLS];
  // The rescale factors leave phase 2 already split into the bf16 (hi, lo) pairs the
  // macs want, and with an INTEGER flag saying which of the two is one. Doing that
  // per head in phase 3 meant a scalar float subtract and a scalar float compare per
  // head per position, and scalar float on this core is a software call.
  alignas(128) bfloat16 ah[kMLS], al[kMLS], bh[kMLS], bl[kMLS];
  alignas(128) int32_t grew_i[kMLS];

  // Unrolled only when this is the hot path. With ATTN_RB > 1 it runs for the
  // handful of rows that do not fill a block, and the 8x of code it costs is
  // 8x of the core's 16 KB of program memory that the block kernel needs.
#if ATTN_RB == 1 && ATTN_HD <= 128
  AIE_LOOP_UNROLL_FULL      // at HD 256 four heads unrolled over eight head-dim steps overflow
#endif                      // program memory (the 35B's ax, 2026-09-08); the loop stays a loop there
  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned h = kSplit ? ((unsigned)h0 + hl) : hl;   // folds away when this core owns them all                 // global head: q and the kv mapping
    const unsigned kvh = h / (kNH / kKVH);
    const bfloat16 *q = qs + h * kHD;
    const bfloat16 *k = Kt + kvh * kHD;
    // Two accumulators, not one: the hi and lo terms are independent, so this is
    // twice the distance between dependent macs for the scheduler to fill.
    // (aie::reduce_add_v reduces four of these in one call, which is strictly less
    // arithmetic and measured 116.6 -> 141.2 ms: holding four heads' vectors live
    // to feed it spills, and the spill costs more than the reduction saves.)
    accf32 d0 = aie::zeros<accfloat, kV>(), d1 = aie::zeros<accfloat, kV>();
    ATTN_UNROLL_HD
    for (unsigned j = 0; j < kHD; j += kV) {
      const vbN<kV> kj = aie::load_v<kV>(k + j);
      d0 = aie::mac(d0, aie::load_v<kV>(q + j), kj);          // hi
      d1 = aie::mac(d1, aie::load_v<kV>(q + kQW + j), kj);    // lo
    }
    sv[hl] = aie::reduce_add(aie::add(d0, d1).template to_vector<float>());   // 1/sqrt(HD) is in q
  }
  for (unsigned h = kNHL; h < kMLU; ++h) sv[h] = -1e30f;
#if ATTN_ABL < 2
  for (unsigned h = 0; h < kNHL; ++h) ml[h] = sv[h];   // sink, so the scores are not dead code
  return;
#endif

  for (unsigned v = 0; v < kMLU; v += kVE) {
#if ATTN_SCALE_IN_Q
    const vfN<kVE> s = aie::load_v<kVE>(sv + v);
#else
    // q carries no 1/sqrt(HD) at this head dim (ATTN_SCALE_IN_Q): the scores take it here
    const vfN<kVE> s = fmulN<kVE>(aie::load_v<kVE>(sv + v), aie::broadcast<float, kVE>(kScale));
#endif
    const vfN<kVE> m = aie::load_v<kVE>(ml + v);
    const vfN<kVE> mn = aie::max(m, s);
    // mn is m or s, so exactly one of (m - mn), (s - mn) is zero and its exp is
    // exactly 1. One exponential of their SUM therefore carries both: the sum is
    // m - s when s wins and s - m when m does, and the loser's 1.0 is a select
    // rather than a second polynomial.
    const vfN<kVE> e = vexpN<kVE>(faddN<kVE>(fsubN<kVE>(m, mn), fsubN<kVE>(s, mn)));
    const vfN<kVE> one = aie::broadcast<float, kVE>(1.0f);
    const auto grew = aie::gt(s, m);
    const vfN<kVE> a = aie::select(one, e, grew);
    const vfN<kVE> b = aie::select(e, one, grew);
    const vfN<kVE> l = aie::load_v<kVE>(ml + kMLS + v);
    aie::store_v(ml + v, mn);
    aie::store_v(ml + kMLS + v, faddN<kVE>(fmulN<kVE>(l, a), b));
    vbN<kVE> t0, t1;
    splitN<kVE>(a, t0, t1);
    aie::store_v(ah + v, t0);
    aie::store_v(al + v, t1);
    splitN<kVE>(b, t0, t1);
    aie::store_v(bh + v, t0);
    aie::store_v(bl + v, t1);
    aie::store_v(grew_i + v, aie::select(aie::zeros<int32_t, kVE>(),
                                         aie::broadcast<int32_t, kVE>(1), grew));
  }

#if ATTN_ABL < 3
  return;
#endif
  // Unrolled only when this is the hot path. With ATTN_RB > 1 it runs for the
  // handful of rows that do not fill a block, and the 8x of code it costs is
  // 8x of the core's 16 KB of program memory that the block kernel needs.
#if ATTN_RB == 1 && ATTN_HD <= 128
  AIE_LOOP_UNROLL_FULL      // at HD 256 four heads unrolled over eight head-dim steps overflow
#endif                      // program memory (the 35B's ax, 2026-09-08); the loop stays a loop there
  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned kvh = (kSplit ? ((unsigned)h0 + hl) : hl) / (kNH / kKVH);
    const bfloat16 *v = Vt + kvh * kHD;
    const bfloat16 bhh = bh[hl], bll = bl[hl];
    float *o = oacc + hl * kHD;
    // a is exp(m_old - m_new), so it is exactly 1 unless THIS position raised the
    // head's running max -- which over a long context happens O(log P) times, not
    // P. The rescale is not merely a multiply by one either: fscaleN splits o into
    // a bf16 pair and remultiplies, so skipping it is both faster and closer to the
    // fp32 accumulator it is meant to leave alone.
    if (grew_i[hl] == 0) {
      // a is exactly 1: this position did not raise the head's running max, which
      // over a long context is the overwhelming majority. Skipping the rescale is
      // not just a multiply by one saved -- it also leaves the fp32 accumulator
      // alone instead of round-tripping it through a bf16 pair.
      ATTN_UNROLL_HD
      for (unsigned j = 0; j < kHD; j += kV) {
        accf32 acc;
        acc.from_vector(aie::load_v<kV>(o + j));
        const vbN<kV> vj = aie::load_v<kV>(v + j);
        acc = aie::mac(acc, vj, bhh);
        acc = aie::mac(acc, vj, bll);
        aie::store_v(o + j, acc.template to_vector<float>());
      }
    } else {
      const bfloat16 ahh = ah[hl], all = al[hl];
      ATTN_UNROLL_HD
      for (unsigned j = 0; j < kHD; j += kV) {
        accf32 acc = aie::zeros<accfloat, kV>();
        acc = mac_vs<kV>(acc, aie::load_v<kV>(o + j), ahh, all);
        const vbN<kV> vj = aie::load_v<kV>(v + j);
        acc = aie::mac(acc, vj, bhh);
        acc = aie::mac(acc, vj, bll);
        aie::store_v(o + j, acc.template to_vector<float>());
      }
    }
  }
}
#else
// one position: K_t, V_t bf16[KVH * HD]
__attribute__((noinline)) inline void attn_row_impl(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt,
                                  const ATTN_QT *__restrict qs, float *__restrict oacc,
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

#if ATTN_VEXP && ATTN_RB > 1
// kRB cached rows in one call. Same arithmetic as attn_row_impl repeated kRB times,
// with the three things that do not depend on the row hoisted out of it:
//
//   * q is loaded once per head for the whole block, not once per row.
//   * the output accumulator is loaded, rescaled and stored once per block. The
//     rescale factor a is one per block, not one per row -- and it is still exactly
//     1 unless the block raised the head's max.
//   * ONE exponential covers the block. The scores are laid out [row][head], so
//     kRB * kNHL of them are a single vector; vector WIDTH turned out to be nearly
//     free on this core (narrowing the exp from 32 lanes to 8 was worth ~1%), so
//     four rows' exponentials cost about what one row's did.
//
// Every row in a block is real: attn_meta_impl sends any window containing the
// position-0 dummy down the one-row path instead.
__attribute__((noinline)) inline void attn_rowb_impl(const bfloat16 *const *__restrict Kb,
                                   const bfloat16 *const *__restrict Vb,
                                   const ATTN_QT *__restrict qs, float *__restrict oacc,
                                   float *__restrict ml ATTN_H0_PARM) {
  ATTN_H0_DECL
  alignas(128) float sv[kPV], mnv[kPV];
  alignas(128) bfloat16 ph[kPV], pl[kPV];
  alignas(128) bfloat16 ah[kMLS], al[kMLS];
  alignas(128) int32_t grew_i[kMLS];
#if ATTN_NHL < 8
  // the padding lanes [kNHL, kNL) of every row: -1e30 never wins a max and exponentiates to 0
  AIE_LOOP_UNROLL_FULL
  for (unsigned r = 0; r < kRB; ++r) aie::store_v(sv + r * kNL, aie::broadcast<float, kNL>(-1e30f));
#endif

#if ATTN_RB <= 2 && ATTN_HD <= 128
  AIE_LOOP_UNROLL_FULL      // at RB 4 the fully-unrolled body crashes clang (and does not fit);
#endif                      // at HD 256 it overflows program memory even at RB 2 (Gemma3-4B)
  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned h = kSplit ? ((unsigned)h0 + hl) : hl;
    const unsigned kvh = h / (kNH / kKVH);
    const bfloat16 *q = qs + h * kHD;
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < kRB; ++r) {
      const bfloat16 *k = Kb[r] + kvh * kHD;
      accf32 d0 = aie::zeros<accfloat, kV>(), d1 = aie::zeros<accfloat, kV>();
      ATTN_UNROLL_HD
      for (unsigned j = 0; j < kHD; j += kV) {
        const vbN<kV> kj = aie::load_v<kV>(k + j);
        d0 = aie::mac(d0, aie::load_v<kV>(q + j), kj);
        d1 = aie::mac(d1, aie::load_v<kV>(q + kQW + j), kj);
      }
      sv[r * kNL + hl] = aie::reduce_add(aie::add(d0, d1).template to_vector<float>());
    }
  }
#if !ATTN_SCALE_IN_Q
  // q carries no 1/sqrt(HD) at this head dim (ATTN_SCALE_IN_Q): one vector multiply
  // scales the whole block's scores
  aie::store_v(sv, fmulN<kPV>(aie::load_v<kPV>(sv), aie::broadcast<float, kPV>(kScale)));
#endif

  // the block's max per head, then one update of (m, l) for the whole block
  vfN<kNL> smax = aie::load_v<kNL>(sv);
  AIE_LOOP_UNROLL_FULL
  for (unsigned r = 1; r < kRB; ++r) smax = aie::max(smax, aie::load_v<kNL>(sv + r * kNL));
  const vfN<kNL> m = aie::load_v<kNL>(ml);
  const vfN<kNL> mn = aie::max(m, smax);
  AIE_LOOP_UNROLL_FULL
  for (unsigned r = 0; r < kRB; ++r) aie::store_v(mnv + r * kNL, mn);

  alignas(128) float pvf[kPV];
  const vfN<kPV> pv = vexpN<kPV>(fsubN<kPV>(aie::load_v<kPV>(sv), aie::load_v<kPV>(mnv)));
  aie::store_v(pvf, pv);
  vbN<kPV> t0, t1;
  splitN<kPV>(pv, t0, t1);
  aie::store_v(ph, t0);
  aie::store_v(pl, t1);

  // a = exp(m - mn): one of (m - mn) and (smax - mn) is zero, so one exp gives both
  const vfN<kNL> e = vexpN<kNL>(faddN<kNL>(fsubN<kNL>(m, mn), fsubN<kNL>(smax, mn)));
  const auto grew = aie::gt(smax, m);
  const vfN<kNL> a = aie::select(aie::broadcast<float, kNL>(1.0f), e, grew);
  vfN<kNL> lsum = aie::load_v<kNL>(pvf);
  AIE_LOOP_UNROLL_FULL
  for (unsigned r = 1; r < kRB; ++r) lsum = faddN<kNL>(lsum, aie::load_v<kNL>(pvf + r * kNL));
  aie::store_v(ml, mn);
  aie::store_v(ml + kMLS, faddN<kNL>(fmulN<kNL>(aie::load_v<kNL>(ml + kMLS), a), lsum));
  vbN<kNL> u0, u1;
  splitN<kNL>(a, u0, u1);
  aie::store_v(ah, u0);
  aie::store_v(al, u1);
  aie::store_v(grew_i, aie::select(aie::zeros<int32_t, kNL>(),
                                   aie::broadcast<int32_t, kNL>(1), grew));

#if ATTN_RB <= 2 && ATTN_HD <= 128
  AIE_LOOP_UNROLL_FULL      // at RB 4 the fully-unrolled body crashes clang (and does not fit);
#endif                      // at HD 256 it overflows program memory even at RB 2 (Gemma3-4B)
  for (unsigned hl = 0; hl < kNHL; ++hl) {
    const unsigned kvh = (kSplit ? ((unsigned)h0 + hl) : hl) / (kNH / kKVH);
    float *o = oacc + hl * kHD;
    const bfloat16 ahh = ah[hl], all = al[hl];
    // The rescale is its own two-iteration loop rather than a second copy of the
    // accumulation: at RB rows unrolled over RB * kNHL heads, duplicating the
    // accumulation overflowed the core's 16 KB of program memory.
    if (grew_i[hl] != 0) {
      ATTN_UNROLL_HD
      for (unsigned j = 0; j < kHD; j += kV) {
        accf32 acc = aie::zeros<accfloat, kV>();
        acc = mac_vs<kV>(acc, aie::load_v<kV>(o + j), ahh, all);
        aie::store_v(o + j, acc.template to_vector<float>());
      }
    }
    ATTN_UNROLL_HD
    for (unsigned j = 0; j < kHD; j += kV) {
      accf32 acc;
      acc.from_vector(aie::load_v<kV>(o + j));
      AIE_LOOP_UNROLL_FULL
      for (unsigned r = 0; r < kRB; ++r) {
        const vbN<kV> vj = aie::load_v<kV>(Vb[r] + kvh * kHD + j);
        acc = aie::mac(acc, vj, ph[r * kNL + hl]);
        acc = aie::mac(acc, vj, pl[r * kNL + hl]);
      }
      aie::store_v(o + j, acc.template to_vector<float>());
    }
  }
}
#endif

// cached row number pb[2] of the nf streamed: rows t >= pos are the position-0 dummy
static inline void attn_step_impl(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt,
                                  const ATTN_QT *__restrict qs, float *__restrict oacc,
                                  float *__restrict ml, int32_t *__restrict pb ATTN_H0_PARM) {
  const int32_t t = pb[2];
  pb[2] = t + 1;
  if (t >= pb[0]) return;
#if !ATTN_NULL
  attn_row_impl(Kt, Vt, qs, oacc, ml ATTN_H0_ARG);
#else
  (void)Kt; (void)Vt; (void)qs; (void)oacc; (void)ml;
#endif
}

// kHPO heads -> one output element: og = o/l [* sigmoid(gate)]; with the gate, g0 holds the
// first kHPE of those heads' gates and g1 the next kHPE (kHPO == 2 * kHPE).
__attribute__((noinline)) inline void attn_fin_impl(const float *__restrict oacc, const float *__restrict ml,
                                 const float *__restrict g0, const float *__restrict g1,
                                 bfloat16 *__restrict og, int hp) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned i = 0; i < kOGH; ++i) {
    const unsigned h = kOGH * hp + i;
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
