//===- attn_fa.cc ---------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Core kernels of attn_fa.py: causal flash attention for the dense block prefill at head dim 64
// (OPEN-PREFILL-ATTN-FA). One core owns TQ = 32 query rows and walks every key chunk of the
// window (dit_fa's loop), so the online softmax never leaves the core. Unlike dit_fa:
//   - The 32 query rows are the vector lanes. Scores come out transposed (S^T, a key a vector), so
//     the row max, the rescale and the row sum are lane-wise: no horizontal reduction anywhere.
//   - Products are bf16 element-wise MACs into fp32 (a broadcast key or P element times a vector),
//     not the emulated 4x8x8 mmul, whose expanded operands spilled to the stack every iteration.
//   - Q arrives as a bf16 hi / lo pair (AF_QSPLIT): without a q/k norm the scores are unbounded.
//   - The host folds log2(e)/sqrt(hd) into Q, so S is in log2 units. The running max is an
//     integer, so a rescale is an exact power of two; 2^x is a polynomial to ~2^-12 (aie2p's bf16
//     exp2 is Mitchell's, 6% off). O and l accumulate in fp32; l from the bf16 P the product used.
//
// Layouts (bf16 unless noted):
//   Q^T  [64 dims][32 rows] (hi, lo)      K, V  [64 keys][64 dims], the cache's rows
//   S^T  [64 keys][32 rows] fp32          P     [64 keys][2 x 32] bf16, each value doubled, in S^T's slot
//   O    [32 rows][64 dims] fp32          st    m [32] | l [32] (fp32) | p0, status (int32)
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef AF_QSPLIT
#define AF_QSPLIT 1
#endif
#ifndef AF_TAU
#define AF_TAU 16
#endif
#ifndef AF_ABL
#define AF_ABL 0   // ablation (timing only): 1 skips S = QK, 2 the softmax, 4 O += PV
#endif

constexpr int TQ = 32;
constexpr int LKP = 64;
constexpr int HD = 64;
constexpr int ST_M = 0, ST_L = 32, ST_I = 64;
constexpr float NEG = -1.0e30f;

using vf = aie::vector<float, 32>;
using vb = aie::vector<bfloat16, 32>;
using vi = aie::vector<int32_t, 32>;
using af32 = aie::accum<accfloat, 32>;

#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)

// broadcast_elem on bf16 is vextbcst.32: it broadcasts a 32-bit unit, i.e. a PAIR of bf16. So values are
// first doubled (dup: lane 2i and 2i+1 both hold v[i]) and bc(d, i) then broadcasts v[i] of the doubled half.
struct dup {
  vb lo, hi;   // v[0..15] doubled, v[16..31] doubled
};
static inline dup dbl(vb v) {
  const auto z = aie::interleave_zip(v, v, 1);
  return {z.first, z.second};
}
static inline vb bc(const dup &d, int i) {
  return vb(::broadcast_elem((v32bfloat16)(i < 16 ? d.lo : d.hi), i & 15));
}
static inline vb bc16(vb doubled, int i) { return vb(::broadcast_elem((v32bfloat16)doubled, i)); }

static inline vf fsub(vf a, vf b) {
  af32 x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::sub(x, y).template to_vector<float>();
}

static inline vf fadd(vf a, vf b) {
  af32 x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::add(x, y).template to_vector<float>();
}

static inline void split(vf v, vb &h, vb &l) {
  af32 x;
  x.from_vector(v);
  h = x.template to_vector<bfloat16>();
  l = aie::sub(x, h).template to_vector<bfloat16>();
}

static inline vb to_bf(vf v) {
  af32 a;
  a.from_vector(v);
  return a.template to_vector<bfloat16>();
}

// a (fp32) * b (bf16 hi / lo): the three significant cross terms (~2^-16)
static inline vf fmul_hl(vf a, vb bh, vb bl) {
  vb ah, al;
  split(a, ah, al);
  af32 r = aie::mul(ah, bh);
  r = aie::mac(r, ah, bl);
  r = aie::mac(r, al, bh);
  return r.template to_vector<float>();
}

static inline vf fmul(vf a, vf b) {
  vb bh, bl;
  split(b, bh, bl);
  return fmul_hl(a, bh, bl);
}

// The softmax's 2^x straight to bf16, for x <= TAU: x = n + f (|f| <= 1/2, f in bf16), 2^f a degree-3
// polynomial with p(0) = 1 (integers are exact), its linear term from hi / lo halves, then n added to the
// bf16 exponent field; under 2^-125 is zero. ~25 vector ops against exp2v's ~45, error under a bf16 ulp.
#ifndef AF_C1
#define AF_C1 0.69314718f
#define AF_C2 0.24022650f
#define AF_C3 0.05550411f
#endif
static inline vb exp2b(vf x) {
  x = aie::max(x, aie::broadcast<float, 32>(-127.0f));
  const vi n = aie::to_fixed<int32_t>(x, 0);
  af32 a;
  a.from_vector(x);
  a = aie::sub(a, aie::to_float<float>(n, 0));
  const vb f = a.template to_vector<bfloat16>();
  af32 t;
  t.from_vector(aie::broadcast<float, 32>(AF_C2));
  t = aie::mac(t, f, aie::broadcast<bfloat16, 32>((bfloat16)AF_C3));
  const vb u = t.template to_vector<bfloat16>();
  t.from_vector(aie::broadcast<float, 32>(AF_C1));
  t = aie::mac(t, u, f);
  const vb wh = t.template to_vector<bfloat16>();
  const vb wl = aie::sub(t, wh).template to_vector<bfloat16>();
  af32 p;
  p.from_vector(aie::broadcast<float, 32>(1.0f));
  p = aie::mac(p, wh, f);
  p = aie::mac(p, wl, f);
  vi bits = p.template to_vector<float>().template cast_to<int32_t>();
  bits = aie::add(bits, aie::upshift(n, 23));
  bits = aie::select(bits, aie::zeros<int32_t, 32>(), aie::lt(n, aie::broadcast<int32_t, 32>(-125)));
  return to_bf(bits.template cast_to<float>());
}

// 2^x for x <= ~TAU: round to n + f, |f| <= 1/2, 2^f by Horner in bf16 products with fp32 sums
// (the linear term from hi / lo halves), n added to the exponent field; under 2^-125 is zero.
static inline vf exp2v(vf x) {
  x = aie::max(x, aie::broadcast<float, 32>(-127.0f));
  x = aie::min(x, aie::broadcast<float, 32>(64.0f));
  const vi n = aie::to_fixed<int32_t>(x, 0);
  const vf f = fsub(x, aie::to_float<float>(n, 0));
  vb fh, fl;
  split(f, fh, fl);
  af32 t;
  t.from_vector(aie::broadcast<float, 32>(0.0555041087f));
  t = aie::mac(t, fh, aie::broadcast<bfloat16, 32>((bfloat16)0.0096181291f));
  const vb t3 = t.template to_vector<bfloat16>();
  t.from_vector(aie::broadcast<float, 32>(0.2402265070f));
  t = aie::mac(t, t3, fh);
  const vb t2 = t.template to_vector<bfloat16>();
  t.from_vector(aie::broadcast<float, 32>(0.6931471806f));
  t = aie::mac(t, t2, fh);
  const vb t1h = t.template to_vector<bfloat16>();
  const vb t1l = aie::sub(t, t1h).template to_vector<bfloat16>();
  af32 p;
  p.from_vector(aie::broadcast<float, 32>(1.0f));
  p = aie::mac(p, t1h, fh);
  p = aie::mac(p, t1h, fl);
  p = aie::mac(p, t1l, fh);
  vi bits = p.template to_vector<float>().template cast_to<int32_t>();
  bits = aie::add(bits, aie::upshift(n, 23));
  bits = aie::select(bits, aie::zeros<int32_t, 32>(), aie::lt(n, aie::broadcast<int32_t, 32>(-125)));
  return bits.template cast_to<float>();
}

// S^T[j] = K[j] . Q^T. Four accumulators in flight (two keys x hi / lo, or four keys) so no MAC waits on
// the one before it; each Q^T vector load feeds two or four MACs.
static inline void qk_t(const bfloat16 *__restrict qh, const bfloat16 *__restrict ql, const bfloat16 *__restrict k,
                        float *__restrict s) {
#if AF_QSPLIT
  for (int j = 0; j < LKP; j += 2) {
    const bfloat16 *__restrict k0 = k + j * HD;
    const bfloat16 *__restrict k1 = k0 + HD;
    af32 a0 = aie::zeros<accfloat, 32>(), a1 = aie::zeros<accfloat, 32>();
    af32 c0 = aie::zeros<accfloat, 32>(), c1 = aie::zeros<accfloat, 32>();
    for (int half = 0; half < 2; half++) {
      const dup d0 = dbl(aie::load_v<32>(k0 + half * 32)), d1 = dbl(aie::load_v<32>(k1 + half * 32));
#pragma clang loop unroll(full)
      for (int part = 0; part < 2; part++) {
        const vb kk0 = part ? d0.hi : d0.lo, kk1 = part ? d1.hi : d1.lo;
        const bfloat16 *__restrict ph = qh + (half * 32 + part * 16) * TQ;
        const bfloat16 *__restrict pl = ql + (half * 32 + part * 16) * TQ;
#pragma clang loop unroll_count(4)
        for (int d = 0; d < 16; d++) {
          const vb q = aie::load_v<32>(ph + d * TQ), qlo = aie::load_v<32>(pl + d * TQ);
          const vb b0 = bc16(kk0, d), b1 = bc16(kk1, d);
          a0 = aie::mac(a0, b0, q);
          a1 = aie::mac(a1, b1, q);
          c0 = aie::mac(c0, b0, qlo);
          c1 = aie::mac(c1, b1, qlo);
        }
      }
    }
    aie::store_v(s + j * TQ, aie::add(a0, c0).template to_vector<float>());
    aie::store_v(s + (j + 1) * TQ, aie::add(a1, c1).template to_vector<float>());
  }
#else
  (void)ql;
  for (int j = 0; j < LKP; j += 4) {
    af32 a0 = aie::zeros<accfloat, 32>(), a1 = aie::zeros<accfloat, 32>();
    af32 a2 = aie::zeros<accfloat, 32>(), a3 = aie::zeros<accfloat, 32>();
    for (int half = 0; half < 2; half++) {
      const dup d0 = dbl(aie::load_v<32>(k + j * HD + half * 32)), d1 = dbl(aie::load_v<32>(k + (j + 1) * HD + half * 32));
      const dup d2 = dbl(aie::load_v<32>(k + (j + 2) * HD + half * 32)), d3 = dbl(aie::load_v<32>(k + (j + 3) * HD + half * 32));
#pragma clang loop unroll(full)
      for (int part = 0; part < 2; part++) {
        const vb kk0 = part ? d0.hi : d0.lo, kk1 = part ? d1.hi : d1.lo;
        const vb kk2 = part ? d2.hi : d2.lo, kk3 = part ? d3.hi : d3.lo;
        const bfloat16 *__restrict ph = qh + (half * 32 + part * 16) * TQ;
#pragma clang loop unroll_count(4)
        for (int d = 0; d < 16; d++) {
          const vb q = aie::load_v<32>(ph + d * TQ);
          a0 = aie::mac(a0, bc16(kk0, d), q);
          a1 = aie::mac(a1, bc16(kk1, d), q);
          a2 = aie::mac(a2, bc16(kk2, d), q);
          a3 = aie::mac(a3, bc16(kk3, d), q);
        }
      }
    }
    aie::store_v(s + j * TQ, a0.template to_vector<float>());
    aie::store_v(s + (j + 1) * TQ, a1.template to_vector<float>());
    aie::store_v(s + (j + 2) * TQ, a2.template to_vector<float>());
    aie::store_v(s + (j + 3) * TQ, a3.template to_vector<float>());
  }
#endif
}

// O[r] += sum_j P[j][r] V[j], two rows a pass so each V load feeds two MACs. P is stored doubled
// ([key][64]: rows 0..15 then 16..31, each value twice) so a row's value is one 32-bit broadcast.
static inline void pv_t(const bfloat16 *__restrict p, const bfloat16 *__restrict v, float *__restrict o) {
  for (int r = 0; r < TQ; r += 2) {
    float *__restrict o0 = o + r * HD;
    float *__restrict o1 = o0 + HD;
    const bfloat16 *__restrict pr = p + (r / 16) * 32;
    const int i = r & 15;
    af32 a00, a01, a10, a11;
    a00.from_vector(aie::load_v<32>(o0));
    a01.from_vector(aie::load_v<32>(o0 + 32));
    a10.from_vector(aie::load_v<32>(o1));
    a11.from_vector(aie::load_v<32>(o1 + 32));
#pragma clang loop unroll_count(4)
    for (int j = 0; j < LKP; j++) {
      const vb pj = aie::load_v<32>(pr + j * 2 * TQ);
      const vb b0 = bc16(pj, i), b1 = bc16(pj, i + 1);
      const vb v0 = aie::load_v<32>(v + j * HD), v1 = aie::load_v<32>(v + j * HD + 32);
      a00 = aie::mac(a00, b0, v0);
      a01 = aie::mac(a01, b0, v1);
      a10 = aie::mac(a10, b1, v0);
      a11 = aie::mac(a11, b1, v1);
    }
    aie::store_v(o0, a00.template to_vector<float>());
    aie::store_v(o0 + 32, a01.template to_vector<float>());
    aie::store_v(o1, a10.template to_vector<float>());
    aie::store_v(o1 + 32, a11.template to_vector<float>());
  }
}

// the half-split rotation of one token row (dims i and i + 32 pair) by its table row
static inline void rope_row(const float *__restrict x, const float *__restrict t, vf &r0, vf &r1) {
  const vf x0 = aie::load_v<32>(x), x1 = aie::load_v<32>(x + 32);
  const vf c = aie::load_v<32>(t), s = aie::load_v<32>(t + 32);
  r0 = fsub(fmul(x0, c), fmul(x1, s));
  r1 = fadd(fmul(x1, c), fmul(x0, s));
}

alignas(64) static const int32_t k_iota[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

extern "C" {

// the pass's record, the first words of the meta element: p0 = the block's first position
void af_meta(bfloat16 *__restrict e, float *__restrict st) {
  int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
  si[0] = reinterpret_cast<const int32_t *>(e)[0];
}

// a Q element is two cores' Q^T, back to back; keep this core's
void af_capture(bfloat16 *__restrict src, bfloat16 *__restrict dst, int32_t sub) {
  const bfloat16 *__restrict ps = src + sub * HD * TQ;
  for (int j = 0; j < HD * TQ; j += 32)
    aie::store_v(dst + j, aie::load_v<32>(ps + j));
}

void af_init(float *__restrict o, float *__restrict st) {
  const vf z = aie::zeros<float, 32>();
  for (int i = 0; i < TQ * HD; i += 32)
    aie::store_v(o + i, z);
  aie::store_v(st + ST_M, aie::broadcast<float, 32>(NEG));
  aie::store_v(st + ST_L, z);
}

// S^T for one key chunk; status 0 = every key past every row's position (skip), 1 = all visible, 2 = ragged
void af_qk(bfloat16 *__restrict qh, bfloat16 *__restrict ql, bfloat16 *__restrict k, float *__restrict s,
           float *__restrict st, int32_t tok0, int32_t chunk) {
  SET_ROUNDING();
  int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
  const int32_t first = si[0] + tok0;   // the last key row 0 sees
  const int32_t k0 = chunk * LKP;
  const int32_t status = k0 > first + TQ - 1 ? 0 : (k0 + LKP - 1 <= first ? 1 : 2);
  si[1] = status;
  if (status == 0 || (AF_ABL & 1))
    return;
  qk_t(qh, ql, k, s);
}

// mask, lazy rescale, P^T = 2^(S^T - m) in place as bf16, l += rowsum(P)
void af_sm(float *s, float *__restrict st, float *__restrict o, int32_t tok0, int32_t chunk) {
  SET_ROUNDING();
  int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
  const int32_t status = si[1];
  if (status == 0 || (AF_ABL & 2))
    return;
  if (status == 2) {
    // key j of the chunk is visible to row r when j <= p0 + tok0 + r - 64 chunk
    const vi bound = aie::add(aie::load_v<32>(k_iota), aie::broadcast<int32_t, 32>(si[0] + tok0 - chunk * LKP));
    for (int j = 0; j < LKP; j++) {
      float *ps = s + j * TQ;
      aie::store_v(ps, aie::select(aie::load_v<32>(ps), aie::broadcast<float, 32>(NEG),
                                   aie::lt(bound, aie::broadcast<int32_t, 32>(j))));
    }
  }
  // four running maxima, so the compares do not wait on each other
  vf m0 = aie::load_v<32>(s), m1 = aie::load_v<32>(s + TQ), m2 = aie::load_v<32>(s + 2 * TQ),
     m3 = aie::load_v<32>(s + 3 * TQ);
  for (int j = 4; j < LKP; j += 4) {
    m0 = aie::max(m0, aie::load_v<32>(s + j * TQ));
    m1 = aie::max(m1, aie::load_v<32>(s + (j + 1) * TQ));
    m2 = aie::max(m2, aie::load_v<32>(s + (j + 2) * TQ));
    m3 = aie::max(m3, aie::load_v<32>(s + (j + 3) * TQ));
  }
  const vf mx = aie::max(aie::max(m0, m1), aie::max(m2, m3));
  const vf m = aie::load_v<32>(st + ST_M);
  const vf mt = fadd(m, aie::broadcast<float, 32>(float(AF_TAU)));
  const auto up = aie::gt(mx, mt);
  if (!up.empty()) {
    // m moves to round(row max) where a row passed m + TAU; O and l scale by the exact 2^(m - m')
    const vf mn = aie::select(m, aie::to_float<float>(aie::to_fixed<int32_t>(mx, 0), 0), up);
    const vb fac = to_bf(exp2v(fsub(m, mn)));
    const vb zero = aie::zeros<bfloat16, 32>();
    const dup fd = dbl(fac);
    for (int r = 0; r < TQ; r++) {
      const vb b = bc(fd, r);
      float *po = o + r * HD;
      aie::store_v(po, fmul_hl(aie::load_v<32>(po), b, zero));
      aie::store_v(po + 32, fmul_hl(aie::load_v<32>(po + 32), b, zero));
    }
    aie::store_v(st + ST_L, fmul_hl(aie::load_v<32>(st + ST_L), fac, zero));
    aie::store_v(st + ST_M, mn);
  }
  const vf mm = aie::load_v<32>(st + ST_M);
  af32 l, l2 = aie::zeros<accfloat, 32>();
  l.from_vector(aie::load_v<32>(st + ST_L));
  const vb one = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
  // in place: P[j], doubled, is exactly the 128 bytes of S^T[j] this iteration has just read; four keys an
  // iteration so their exponential chains interleave
  bfloat16 *p = reinterpret_cast<bfloat16 *>(s);
  for (int j = 0; j < LKP; j += 4) {
    vb pb[4];
#pragma clang loop unroll(full)
    for (int u = 0; u < 4; u++)
      pb[u] = exp2b(fsub(aie::load_v<32>(s + (j + u) * TQ), mm));
#pragma clang loop unroll(full)
    for (int u = 0; u < 4; u++) {
      const dup pd = dbl(pb[u]);
      aie::store_v(p + (j + u) * 2 * TQ, pd.lo);
      aie::store_v(p + (j + u) * 2 * TQ + 32, pd.hi);
      if (u & 1)
        l2 = aie::mac(l2, pb[u], one);
      else
        l = aie::mac(l, pb[u], one);
    }
  }
  aie::store_v(st + ST_L, aie::add(l, l2).template to_vector<float>());
}

void af_pv(float *__restrict s, bfloat16 *__restrict v, float *__restrict o, float *__restrict st) {
  SET_ROUNDING();
  const int32_t *si = reinterpret_cast<const int32_t *>(st + ST_I);
  if (si[1] == 0 || (AF_ABL & 4))
    return;
  pv_t(reinterpret_cast<const bfloat16 *>(s), v, o);
}

// O /= l, row by row
void af_fin(float *__restrict o, float *__restrict st) {
  SET_ROUNDING();
  const vf l = aie::load_v<32>(st + ST_L);
  // hardware seed, two Newton steps
  vf r = aie::inv(l);
  const vf two = aie::broadcast<float, 32>(2.0f);
  r = fmul(r, fsub(two, fmul(l, r)));
  r = fmul(r, fsub(two, fmul(l, r)));
  vb rh, rl;
  split(r, rh, rl);
  const dup dh = dbl(rh), dl = dbl(rl);
  for (int row = 0; row < TQ; row++) {
    const vb bh = bc(dh, row), bl = bc(dl, row);
    float *po = o + row * HD;
    aie::store_v(po, fmul_hl(aie::load_v<32>(po), bh, bl));
    aie::store_v(po + 32, fmul_hl(aie::load_v<32>(po + 32), bh, bl));
  }
}

// OPEN-PREFILL-GLUE-PREP: the block's rope, scale and Q packing on the array. A raw element is this core's 32 tokens of
// one head, fp32 [32 tokens][64 dims] as the qkv GEMM left them; the table element is [32 tokens][cos 32 | sin 32].
void af_ptab(float *__restrict src, float *__restrict dst) {
  if (AF_ABL & 8)
    return;
  for (int j = 0; j < TQ * HD; j += 32)
    aie::store_v(dst + j, aie::load_v<32>(src + j));
}

// qh / ql hold the scaled rotation as bf16 hi / lo, token rows; o gets them transposed ([64 dims][32 tokens]), hi then lo
void af_prep_q(float *__restrict raw, float *__restrict tbl, bfloat16 *__restrict qh, bfloat16 *__restrict ql,
               bfloat16 *__restrict o) {
  if (AF_ABL & 8)
    return;
  SET_ROUNDING();
  // 1/sqrt(64) and log2(e): the scores come out in the softmax's units
  const vf qs = aie::broadcast<float, 32>(0.125f * 1.4426950408889634f);
  for (int t = 0; t < ((AF_ABL & 16) ? 0 : TQ); t++) {
    vf r0, r1;
    rope_row(raw + t * HD, tbl + t * HD, r0, r1);
    r0 = fmul(r0, qs);
    r1 = fmul(r1, qs);
    vb h0, l0, h1, l1;
    split(r0, h0, l0);
    split(r1, h1, l1);
    aie::store_v(qh + t * HD, h0);
    aie::store_v(qh + t * HD + 32, h1);
    aie::store_v(ql + t * HD, l0);
    aie::store_v(ql + t * HD + 32, l1);
  }
  if (AF_ABL & 32)
    return;
  // [token][dim] -> [dim][token] by five zip rounds per 32x32 block (each round rotates the row and column index bits);
  // block b = {hi, lo} x {dims 0-31, 32-63} ends in its own slot of o, the next slot (the last one qh) is the ping-pong
  for (int b = 0; b < 4; b++) {
    const bfloat16 *__restrict src = (b & 2 ? ql : qh) + (b & 1) * 32;
    bfloat16 *__restrict x = o + b * (TQ * 32);
    bfloat16 *__restrict y = b < 3 ? x + TQ * 32 : qh;
    for (int i = 0; i < 16; i++) {
      const vb lo = aie::load_v<32>(src + i * HD), hi = aie::load_v<32>(src + (i + 16) * HD);
      const auto z = aie::interleave_zip(lo, hi, 1);
      aie::store_v(x + 2 * i * 32, z.first);
      aie::store_v(x + (2 * i + 1) * 32, z.second);
    }
    for (int r = 1; r < 5; r++) {
      const bfloat16 *__restrict from = r & 1 ? x : y;
      bfloat16 *__restrict to = r & 1 ? y : x;
      for (int i = 0; i < 16; i++) {
        const vb lo = aie::load_v<32>(from + i * 32), hi = aie::load_v<32>(from + (i + 16) * 32);
        const auto z = aie::interleave_zip(lo, hi, 1);
        aie::store_v(to + 2 * i * 32, z.first);
        aie::store_v(to + (2 * i + 1) * 32, z.second);
      }
    }
  }
}

// o: K rotated then V, bf16 token rows (the cache's layout), 32 tokens each
void af_prep_kv(float *__restrict rk, float *__restrict rv, float *__restrict tbl, bfloat16 *__restrict o) {
  if (AF_ABL & 8)
    return;
  SET_ROUNDING();
  for (int t = 0; t < TQ; t++) {
    vf r0, r1;
    rope_row(rk + t * HD, tbl + t * HD, r0, r1);
    aie::store_v(o + t * HD, to_bf(r0));
    aie::store_v(o + t * HD + 32, to_bf(r1));
  }
  bfloat16 *__restrict ov = o + TQ * HD;
  for (int j = 0; j < TQ * HD; j += 32)
    aie::store_v(ov + j, to_bf(aie::load_v<32>(rv + j)));
}

} // extern "C"
