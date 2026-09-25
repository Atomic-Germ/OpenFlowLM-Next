#pragma once
// Optional fp32-accuracy arithmetic on AIE2P's bf16 multiplier. Unlike the
// legacy two-part, three-product approximation, retain three bf16 components
// and all products through the third significance level. Used where an f32
// result will be rounded to bf16 again: small errors can cross a rounding bin.
// Distinct names prevent COMDAT collisions with legacy vecmath functions.
#include "vecmath.h"

template <unsigned N>
static inline void precise_splitN(const vfN<N> &v, vbN<N> &h, vbN<N> &l, vbN<N> &t) {
  accN<N> a;
  a.from_vector(v);
  h = a.template to_vector<bfloat16>();
  a = aie::sub(a, h);
  l = a.template to_vector<bfloat16>();
  t = aie::sub(a, l).template to_vector<bfloat16>();
}

template <unsigned N>
static inline vfN<N> precise_mulN(const vfN<N> &a, const vfN<N> &b) {
  vbN<N> ah, al, at, bh, bl, bt;
  precise_splitN<N>(a, ah, al, at);
  precise_splitN<N>(b, bh, bl, bt);
  accN<N> acc = aie::mul(ah, bh);
  acc = aie::mac(acc, ah, bl);
  acc = aie::mac(acc, al, bh);
  acc = aie::mac(acc, al, bl);
  acc = aie::mac(acc, ah, bt);
  acc = aie::mac(acc, at, bh);
  return acc.template to_vector<float>();
}

template <unsigned N>
__attribute__((noinline)) inline vfN<N> precise_expN(vfN<N> x) {
  x = aie::max(x, aie::broadcast<float, N>(-87.0f));
  x = aie::min(x, aie::broadcast<float, N>(88.0f));
  const vfN<N> t = precise_mulN<N>(x, aie::broadcast<float, N>(1.44269504f));
  const aie::vector<int32_t, N> n = aie::to_fixed<int32_t>(t, 0);
  const vfN<N> f = fsubN<N>(t, aie::to_float<float>(n, 0));
  vfN<N> p = aie::broadcast<float, N>(1.54035304e-4f);
  p = faddN<N>(precise_mulN<N>(p, f), aie::broadcast<float, N>(1.33335581e-3f));
  p = faddN<N>(precise_mulN<N>(p, f), aie::broadcast<float, N>(9.61812911e-3f));
  p = faddN<N>(precise_mulN<N>(p, f), aie::broadcast<float, N>(5.55041087e-2f));
  p = faddN<N>(precise_mulN<N>(p, f), aie::broadcast<float, N>(2.40226507e-1f));
  p = faddN<N>(precise_mulN<N>(p, f), aie::broadcast<float, N>(6.93147181e-1f));
  p = faddN<N>(precise_mulN<N>(p, f), aie::broadcast<float, N>(1.0f));
  const aie::vector<int32_t, N> bits = aie::upshift(aie::add(n, aie::broadcast<int32_t, N>(127)), 23);
  return precise_mulN<N>(p, bits.template cast_to<float>());
}

template <unsigned N>
__attribute__((noinline)) inline vfN<N> precise_recipN(const vfN<N> &d) {
  vfN<N> r = aie::inv(d);
  const vfN<N> two = aie::broadcast<float, N>(2.0f);
  r = precise_mulN<N>(r, fsubN<N>(two, precise_mulN<N>(d, r)));
  r = precise_mulN<N>(r, fsubN<N>(two, precise_mulN<N>(d, r)));
  return r;
}

template <unsigned N>
__attribute__((noinline)) inline vfN<N> precise_siluN(const vfN<N> &x) {
  const vfN<N> e = precise_expN<N>(fsubN<N>(aie::zeros<float, N>(), x));
  return precise_mulN<N>(x, precise_recipN<N>(faddN<N>(e, aie::broadcast<float, N>(1.0f))));
}
