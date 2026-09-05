#pragma once
//===- vecmath.h -------------------------------------------*- C++ -*-===//
//
// fp32 vector arithmetic and transcendentals for AIE2P, built on bf16 MACs.
//
// AIE2P has no fp32 vector multiplier (aie::mul<float> returns zero, silently)
// and its tanh / inv / invsqrt / exp2 are LUT linear approximations (~1e-2).
// Everything here splits fp32 operands into bf16 hi + lo halves and
// accumulates the three significant cross terms (~2^-16 relative), and the
// transcendentals are polynomial / Newton on top of that (~1e-7).
//
// 32-lane variants (v32f) and 16-lane variants (v16f) are provided.
// The transcendentals are `inline` + noinline (COMDAT): one copy per core program
// however many translation units use them (16 KB program memory).

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <unsigned N> using vfN = aie::vector<float, N>;
template <unsigned N> using vbN = aie::vector<bfloat16, N>;
template <unsigned N> using accN = aie::accum<accfloat, N>;

using v32f = vfN<32>;
using v32b = vbN<32>;
using accf32 = accN<32>;
using v16f = vfN<16>;
using v16b = vbN<16>;
using accf16 = accN<16>;

// fp32 vector -> (hi, lo) bf16 halves, hi + lo == v to ~16 mantissa bits.
template <unsigned N>
static inline void splitN(const vfN<N> &v, vbN<N> &h, vbN<N> &l) {
  accN<N> a;
  a.from_vector(v);
  h = a.template to_vector<bfloat16>();
  l = aie::sub(a, h).template to_vector<bfloat16>();
}
static inline void split32(const v32f &v, v32b &h, v32b &l) { splitN<32>(v, h, l); }
static inline void split16(const v16f &v, v16b &h, v16b &l) { splitN<16>(v, h, l); }

// acc += a(fp32 vec) * s(bf16 hi/lo scalar)
template <unsigned N>
static inline accN<N> mac_vs(accN<N> acc, const vfN<N> &a, bfloat16 sh, bfloat16 sl) {
  vbN<N> ah, al;
  splitN<N>(a, ah, al);
  acc = aie::mac(acc, ah, sh);
  acc = aie::mac(acc, ah, sl);
  acc = aie::mac(acc, al, sh);
  return acc;
}
// acc += a(fp32 vec) * b(bf16 vec)
template <unsigned N>
static inline accN<N> mac_vv(accN<N> acc, const vfN<N> &a, const vbN<N> &b) {
  vbN<N> ah, al;
  splitN<N>(a, ah, al);
  acc = aie::mac(acc, ah, b);
  acc = aie::mac(acc, al, b);
  return acc;
}

template <unsigned N>
static inline vfN<N> fmulN(const vfN<N> &a, const vfN<N> &b) {
  vbN<N> ah, al, bh, bl;
  splitN<N>(a, ah, al);
  splitN<N>(b, bh, bl);
  accN<N> acc = aie::mul(ah, bh);
  acc = aie::mac(acc, ah, bl);
  acc = aie::mac(acc, al, bh);
  return acc.template to_vector<float>();
}
template <unsigned N>
static inline vfN<N> faddN(const vfN<N> &a, const vfN<N> &b) {
  accN<N> x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::add(x, y).template to_vector<float>();
}
template <unsigned N>
static inline vfN<N> fsubN(const vfN<N> &a, const vfN<N> &b) {
  accN<N> x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::sub(x, y).template to_vector<float>();
}
// a * s for fp32 vector a and fp32 scalar s
template <unsigned N>
static inline vfN<N> fscaleN(const vfN<N> &a, float s) {
  const bfloat16 sh = (bfloat16)s;
  const bfloat16 sl = (bfloat16)(s - (float)sh);
  accN<N> acc = aie::zeros<accfloat, N>();
  return mac_vs<N>(acc, a, sh, sl).template to_vector<float>();
}

// exp(x) to ~1e-7 relative: 2^(x*log2e) = 2^n * 2^f, |f| <= 0.5, degree-6 poly.
template <unsigned N>
__attribute__((noinline)) inline vfN<N> vexpN(vfN<N> x) {
  x = aie::max(x, aie::broadcast<float, N>(-87.0f));
  x = aie::min(x, aie::broadcast<float, N>(88.0f));
  const vfN<N> t = fmulN<N>(x, aie::broadcast<float, N>(1.44269504f));
  const aie::vector<int32_t, N> n = aie::to_fixed<int32_t>(t, 0);      // round (conv_even)
  const vfN<N> nf = aie::to_float<float>(n, 0);
  const vfN<N> f = fsubN<N>(t, nf);                                     // [-0.5, 0.5]
  vfN<N> p = aie::broadcast<float, N>(1.54035304e-4f);
  p = faddN<N>(fmulN<N>(p, f), aie::broadcast<float, N>(1.33335581e-3f));
  p = faddN<N>(fmulN<N>(p, f), aie::broadcast<float, N>(9.61812911e-3f));
  p = faddN<N>(fmulN<N>(p, f), aie::broadcast<float, N>(5.55041087e-2f));
  p = faddN<N>(fmulN<N>(p, f), aie::broadcast<float, N>(2.40226507e-1f));
  p = faddN<N>(fmulN<N>(p, f), aie::broadcast<float, N>(6.93147181e-1f));
  p = faddN<N>(fmulN<N>(p, f), aie::broadcast<float, N>(1.0f));
  const aie::vector<int32_t, N> bits =
      aie::upshift(aie::add(n, aie::broadcast<int32_t, N>(127)), 23);
  const vfN<N> scale = bits.template cast_to<float>();                  // exact power of two
  accN<N> s;
  s.from_vector(scale);
  const vbN<N> sb = s.template to_vector<bfloat16>();                   // exact
  vbN<N> ph, pl;
  splitN<N>(p, ph, pl);
  accN<N> r = aie::mul(ph, sb);
  r = aie::mac(r, pl, sb);
  return r.template to_vector<float>();
}

// 1/d to fp32: hardware inv seed + two Newton steps r = r (2 - d r).
template <unsigned N>
__attribute__((noinline)) inline vfN<N> vrecipN(const vfN<N> &d) {
  vfN<N> r = aie::inv(d);
  const vfN<N> two = aie::broadcast<float, N>(2.0f);
  r = fmulN<N>(r, fsubN<N>(two, fmulN<N>(d, r)));
  r = fmulN<N>(r, fsubN<N>(two, fmulN<N>(d, r)));
  return r;
}

template <unsigned N>
__attribute__((noinline)) inline vfN<N> vsigmoidN(const vfN<N> &x) {
  // (aie::neg on a float vector fails to legalize (G_FNEG) in some kernels;
  // 0 - x through the accumulator is exact.)
  const vfN<N> e = vexpN<N>(fsubN<N>(aie::zeros<float, N>(), x));
  return vrecipN<N>(faddN<N>(e, aie::broadcast<float, N>(1.0f)));
}
template <unsigned N>
inline __attribute__((noinline)) vfN<N> vsiluN(const vfN<N> &x) { return fmulN<N>(x, vsigmoidN<N>(x)); }

// ---- scalar transcendentals (software float on the scalar unit: slow, use
// for a few dozen values per call). No libm dependency.
static inline float sexp(float x) {
  if (x > 88.f) x = 88.f;
  if (x < -87.f) x = -87.f;
  const float t = x * 1.44269504f;
  const int n = (int)(t + (t >= 0.f ? 0.5f : -0.5f));
  const float f = (t - (float)n) * 0.69314718f;      // |f| <= 0.347
  float p = 1.f + f * (1.f + f * (0.5f + f * (0.166666667f + f * (0.0416666667f + f * (0.00833333333f + f * 0.00138888889f)))));
  union { float f; uint32_t u; } s;
  s.u = (uint32_t)(n + 127) << 23;
  return p * s.f;
}
static inline float slog(float x) {                 // natural log, x > 0
  union { float f; uint32_t u; } s;
  s.f = x;
  int e = (int)((s.u >> 23) & 0xFF) - 127;
  s.u = (s.u & 0x007FFFFFu) | 0x3F800000u;           // mantissa in [1, 2)
  float m = s.f;
  if (m > 1.41421356f) { m *= 0.5f; e += 1; }
  const float y = (m - 1.f) / (m + 1.f);
  const float y2 = y * y;
  const float l = 2.f * y * (1.f + y2 * (0.333333333f + y2 * (0.2f + y2 * (0.142857143f + y2 * 0.111111111f))));
  return l + (float)e * 0.69314718f;
}
static inline float ssoftplus(float u) { return u > 20.f ? u : slog(1.f + sexp(u)); }
static inline float ssigmoid(float u) { return 1.f / (1.f + sexp(-u)); }

// scalar 1/sqrt(x) to fp32: hardware seed + two Newton steps
static inline float srsqrt(float x) {
  float inv = aie::invsqrt(x);
  inv = inv * (1.5f - 0.5f * x * inv * inv);
  inv = inv * (1.5f - 0.5f * x * inv * inv);
  return inv;
}

// 32-lane names used by dn_glue.h
static inline v32f fmul32(const v32f &a, const v32f &b) { return fmulN<32>(a, b); }
static inline v32f fadd32(const v32f &a, const v32f &b) { return faddN<32>(a, b); }
static inline v32f fsub32(const v32f &a, const v32f &b) { return fsubN<32>(a, b); }
static inline v32f vexp32(const v32f &x) { return vexpN<32>(x); }
static inline v32f vrecip32(const v32f &d) { return vrecipN<32>(d); }
static inline v32f vsigmoid32(const v32f &x) { return vsigmoidN<32>(x); }
