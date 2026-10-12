#pragma once
// The verify pass's DeltaNet: S read once a head, each position's S' in dnx.h's per-column op order (bit-identical).

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef VDN_NPOS
#define VDN_NPOS 2
#endif
#ifndef VDN_STRIP_UNROLL
#define VDN_STRIP_UNROLL 2
#endif

namespace vdn {
constexpr unsigned kD = 128, kL = 32, kRows = 128, kElemRows = 17;
constexpr unsigned BLK = 0, BLK_BYTES = 2 * kRows * kL * 4;   // half a head's S: [quarter plane][row][32 f32]
constexpr unsigned RS = 1664, R_KH = 0, R_KL = 256, R_QH = 512, R_QL = 768, R_V = 1024, R_SC = 1536, R_SCL = 1600;
constexpr unsigned REC = BLK + BLK_BYTES;   // per position: k, q as bf16 hi / lo, v, [decay, beta] hi and lo
// T pass 1 sums, TN the next position's, DLT delta, O o (raw: the post core scales it)
constexpr unsigned T = REC + VDN_NPOS * RS, TN = T + 256, DLT = TN + 256, O = DLT + 256;
constexpr unsigned BYTES = O + VDN_NPOS * 512;

using vf = aie::vector<float, kL>;
using vb = aie::vector<bfloat16, kL>;
using acc = aie::accum<accfloat, kL>;

inline __attribute__((always_inline)) void split(const vf &v, vb &h, vb &l) {
  acc a;
  a.from_vector(v);
  h = a.template to_vector<bfloat16>();
  l = aie::sub(a, h).template to_vector<bfloat16>();
}
inline __attribute__((always_inline)) acc mac3(acc a, const vb &h, const vb &l, bfloat16 sh, bfloat16 sl) {
  a = aie::mac(a, h, sh);
  a = aie::mac(a, h, sl);
  a = aie::mac(a, l, sh);
  return a;
}
// mac3 with the scalar negated: a - x y is a + x (-y) exactly, and the split of -s is minus the split of s
inline __attribute__((always_inline)) acc msc3(acc a, const vb &h, const vb &l, bfloat16 sh, bfloat16 sl) {
  a = aie::msc(a, h, sh);
  a = aie::msc(a, h, sl);
  a = aie::msc(a, l, sh);
  return a;
}
// dnx_row_update's S'[i] = decay * S[i] + k[i] * delta, in its order
inline __attribute__((always_inline)) vf upd(const vf &x, bfloat16 dh, bfloat16 dl, const vb &deh, const vb &del,
                                             bfloat16 kh, bfloat16 kl) {
  vb h, l;
  split(x, h, l);
  acc s = mac3(aie::zeros<accfloat, kL>(), h, l, dh, dl);
  return mac3(s, deh, del, kh, kl).template to_vector<float>();
}
}  // namespace vdn

inline __attribute__((noinline)) void vdn_rec_impl(const float *__restrict e, uint8_t *__restrict tab, unsigned p) {
  using namespace vdn;
  aie::set_rounding(aie::rounding_mode::conv_even);
  uint8_t *__restrict r = tab + REC + p * RS;
  bfloat16 *__restrict kh = (bfloat16 *)(r + R_KH);
  bfloat16 *__restrict kl = (bfloat16 *)(r + R_KL);
  bfloat16 *__restrict qh = (bfloat16 *)(r + R_QH);
  bfloat16 *__restrict ql = (bfloat16 *)(r + R_QL);
  float *__restrict v = (float *)(r + R_V);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kD; j += kL) {
    vb h, l;
    split(aie::load_v<kL>(e + j), h, l);
    aie::store_v(kh + j, h);
    aie::store_v(kl + j, l);
    split(aie::load_v<kL>(e + kD + j), h, l);
    aie::store_v(qh + j, h);
    aie::store_v(ql + j, l);
    aie::store_v(v + j, aie::load_v<kL>(e + 2 * kD + j));
  }
  vb h, l;
  split(aie::load_v<kL>(e + 384), h, l);                   // [decay, beta, ...]
  aie::store_v((bfloat16 *)(r + R_SC), h);
  aie::store_v((bfloat16 *)(r + R_SCL), l);
}

// element ei: quarter plane ei / 8, rows 17 (ei % 8) ..; position 0's pass 1 rides on the copy
inline __attribute__((noinline)) void vdn_ld_impl(const float *__restrict e, uint8_t *__restrict tab, unsigned ei) {
  using namespace vdn;
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned qq = ei >> 3, r0 = (ei & 7u) * kElemRows;
  const unsigned nr = r0 + kElemRows <= kRows ? kElemRows : kRows - r0;
  float *__restrict dst = (float *)(tab + BLK) + (qq * kRows + r0) * kL;
  float *__restrict t = (float *)(tab + T) + qq * kL;
  const bfloat16 *__restrict kh = (const bfloat16 *)(tab + REC + R_KH) + r0;
  const bfloat16 *__restrict kl = (const bfloat16 *)(tab + REC + R_KL) + r0;
  acc a;
  if (r0 == 0)
    a = aie::zeros<accfloat, kL>();
  else
    a.from_vector(aie::load_v<kL>(t));
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < nr; ++i) {
    const vf s = aie::load_v<kL>(e + i * kL);
    aie::store_v(dst + i * kL, s);
    vb h, l;
    split(s, h, l);
    a = mac3(a, h, l, kh[i], kl[i]);
  }
  aie::store_v(t, a.template to_vector<float>());
}

namespace vdn {
// restrict parameters: inlined, they stay noalias scopes, so loads need not wait for earlier stores
inline __attribute__((always_inline)) void rows2(const bfloat16 *__restrict rk, const bfloat16 *__restrict sc,
                                                 const bfloat16 *__restrict dlh, float *__restrict blk,
                                                 float *__restrict o, float *__restrict tn, float *__restrict y0,
                                                 float *__restrict y1) {
  const bfloat16 ddh = sc[0], ddl = sc[kL];
  const bfloat16 k0h = rk[R_KH / 2], k1h = rk[R_KH / 2 + 1], k0l = rk[R_KL / 2], k1l = rk[R_KL / 2 + 1];
  const bfloat16 q0h = rk[R_QH / 2], q1h = rk[R_QH / 2 + 1], q0l = rk[R_QL / 2], q1l = rk[R_QL / 2 + 1];
  const bfloat16 n0h = rk[(RS + R_KH) / 2], n1h = rk[(RS + R_KH) / 2 + 1];
  const bfloat16 n0l = rk[(RS + R_KL) / 2], n1l = rk[(RS + R_KL) / 2 + 1];
#pragma clang loop unroll_count(VDN_STRIP_UNROLL)
  for (unsigned qq = 0; qq < 2; ++qq) {
    const unsigned c = qq * kL;
    float *__restrict b = blk + qq * kRows * kL;
    const vb deh = aie::load_v<kL>(dlh + c), del = aie::load_v<kL>(dlh + 64 + c);
    const vf x0 = aie::load_v<kL>(b), x1 = aie::load_v<kL>(b + kL);
    acc oa, ta;
    oa.from_vector(aie::load_v<kL>(o + c));
    ta.from_vector(aie::load_v<kL>(tn + c));
    const vf s0 = upd(x0, ddh, ddl, deh, del, k0h, k0l);
    const vf s1 = upd(x1, ddh, ddl, deh, del, k1h, k1l);
    vb sh, sl;
    split(s0, sh, sl);
    oa = mac3(oa, sh, sl, q0h, q0l);
    ta = mac3(ta, sh, sl, n0h, n0l);
    split(s1, sh, sl);
    oa = mac3(oa, sh, sl, q1h, q1l);
    ta = mac3(ta, sh, sl, n1h, n1l);
    aie::store_v(y0 + c, s0);
    aie::store_v(y1 + c, s1);
    aie::store_v(b, s0);
    aie::store_v(b + kL, s1);
    aie::store_v(o + c, oa.template to_vector<float>());
    aie::store_v(tn + c, ta.template to_vector<float>());
  }
}
}  // namespace vdn

// the last position's TN reads T's bytes as k': its result is never used
inline __attribute__((noinline)) void vdn_row_impl(uint8_t *__restrict tab, float *__restrict y, unsigned p, unsigned r2,
                                                   unsigned h) {
  using namespace vdn;
  aie::set_rounding(aie::rounding_mode::conv_even);
  const uint8_t *__restrict rp = tab + REC + p * RS;
  const bfloat16 *__restrict sc = (const bfloat16 *)(rp + R_SC);
  bfloat16 *__restrict dlh = (bfloat16 *)(tab + DLT);
  float *__restrict o = (float *)(tab + O) + (p * 2 + h) * 64;
  float *__restrict tn = (float *)(tab + TN);
  if (r2 == 0) {
    const float *__restrict t = (const float *)(tab + T);
    const float *__restrict v = (const float *)(rp + R_V) + h * 64;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 64; j += kL) {
      vb th, tl;
      split(aie::load_v<kL>(t + j), th, tl);
      acc a;
      a.from_vector(aie::load_v<kL>(v + j));
      a = msc3(a, th, tl, sc[0], sc[kL]);
      vb uh, ul;
      split(a.template to_vector<float>(), uh, ul);
      acc d = mac3(aie::zeros<accfloat, kL>(), uh, ul, sc[1], sc[kL + 1]);
      vb dh, dl;
      split(d.template to_vector<float>(), dh, dl);
      aie::store_v(dlh + j, dh);
      aie::store_v(dlh + 64 + j, dl);
      aie::store_v(o + j, aie::zeros<float, kL>());
      aie::store_v(tn + j, aie::zeros<float, kL>());
    }
  }
  rows2((const bfloat16 *)rp + 2 * r2, sc, dlh, (float *)(tab + BLK) + 2 * r2 * kL, o, tn, y, y + 64);
  if (r2 == kRows / 2 - 1) {
    float *__restrict t = (float *)(tab + T);
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 64; j += kL) aie::store_v(t + j, aie::load_v<kL>(tn + j));
  }
}
