#pragma once
#include "vecmath.h"
using namespace aie::operators;
#include "fp32_add_rne.h"
#include "fp32_mul_rne.h"

struct LnIntegerLanes {
  using V = aie::vector<uint32_t,32>;
  static V v(uint32_t x) { return aie::broadcast<uint32_t,32>(x); }
  static V choose(aie::mask<32> c,V a,V b) { return aie::select(b,a,c); }
  static V mul(V a,V b) { return aie::mul(a,b).to_vector<uint32_t>(0); }
};
__attribute__((noinline)) inline v32f ln_add_rne(v32f a,v32f b) {
  return fp32_add_rne<LnIntegerLanes>(a.cast_to<uint32_t>(),b.cast_to<uint32_t>()).cast_to<float>();
}
static inline v32f ln_neg(v32f a) {
  return (a.cast_to<uint32_t>() ^ LnIntegerLanes::v(0x80000000)).cast_to<float>();
}
static inline v32f ln_sub_rne(v32f a,v32f b) { return ln_add_rne(a,ln_neg(b)); }
__attribute__((noinline)) inline v32f ln_mul_rne(v32f a,v32f b) {
  return fp32_mul_rne<LnIntegerLanes>(a.cast_to<uint32_t>(),b.cast_to<uint32_t>()).cast_to<float>();
}
static inline void ln_two_sum(v32f &sum,v32f &low,v32f term) {
  const auto next=ln_add_rne(sum,term);
  const auto recovered=ln_sub_rne(next,sum);
  const auto error=ln_add_rne(ln_sub_rne(sum,ln_sub_rne(next,recovered)),ln_sub_rne(term,recovered));
  low=ln_add_rne(low,error);sum=next;
}
