#pragma once
#include "ln_rne.h"
#include "fp32_product_chain.h"
#include "sigmoid_series.h"

struct PostCarryOps {
  using V = v32f;
  static V v(float a) { return aie::broadcast<float,32>(a); }
  static V add(V a,V b) { return ln_add_rne(a,b); }
  static V sub(V a,V b) { return ln_sub_rne(a,b); }
  static V mul(V a,V b) { return ln_mul_rne(a,b); }
  static V high(V a) { return (a.cast_to<uint32_t>() & LnIntegerLanes::v(0xfffff000)).cast_to<float>(); }
};

__attribute__((noinline)) inline v32f post_sigmoid(v32f z) {
  const auto small = aie::le(aie::abs(z),PostCarryOps::v(.5f));
  const auto bounded = aie::select(PostCarryOps::v(0.f),z,small);
  const auto series = sigmoid_series<PostCarryOps>(bounded);
  const auto exp = precise_expN<32>(ln_neg(z));
  const auto outside = precise_recipN<32>(faddN<32>(exp,PostCarryOps::v(1.f)));
  return aie::select(outside,series,small);
}

__attribute__((noinline)) inline v32f post_product(v32f o,v32f inv,v32f w,v32f z,v32f r) {
  return fp32_product_chain<PostCarryOps>(o,inv,w,z,r);
}
