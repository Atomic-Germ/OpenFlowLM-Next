#pragma once
// Compensate all nine BF16 component products before rounding to FP32.
// Separate names keep the legacy activation and other core operators intact.
#include "vecmath_precise.h"
#include "gemv_tab.h"
using namespace aie::operators;
#include "fp32_add_rne.h"

struct ActivationAddLanes {
  using V=aie::vector<uint32_t,32>;
  static V v(uint32_t x) { return aie::broadcast<uint32_t,32>(x); }
  static V choose(aie::mask<32> c,V a,V b) { return aie::select(b,a,c); }
};
__attribute__((noinline)) inline v32f act_add(v32f a,v32f b) {
  return fp32_add_rne<ActivationAddLanes>(a.cast_to<uint32_t>(),b.cast_to<uint32_t>()).cast_to<float>();
}

__attribute__((noinline)) inline v32f act_mul_carry(v32f a,v32f b) {
  v32b ah,al,at,bh,bl,bt;
  precise_splitN<32>(a,ah,al,at);precise_splitN<32>(b,bh,bl,bt);
  accf32 sum=aie::zeros<accfloat,32>();
  v32f low=aie::zeros<float,32>();
  q4_product_add(sum,low,ah,bh);
  q4_product_add(sum,low,ah,bl);q4_product_add(sum,low,al,bh);
  q4_product_add(sum,low,al,bl);q4_product_add(sum,low,ah,bt);q4_product_add(sum,low,at,bh);
  q4_product_add(sum,low,al,bt);q4_product_add(sum,low,at,bl);q4_product_add(sum,low,at,bt);
  return aie::add(sum,low).to_vector<float>();
}
static inline v32f act_v(float a) { return aie::broadcast<float,32>(a); }
static inline v32f act_neg(v32f a) { return (a.cast_to<uint32_t>()^ActivationAddLanes::v(0x80000000)).cast_to<float>(); }

#if DENSE_ACT_SERIES
#include "sigmoid_series.h"
struct ActivationSeries {
  using V=v32f;
  static V v(float x) { return act_v(x); }
  static V mul(V a,V b) { return act_mul_carry(a,b); }
  static V add(V a,V b) { return act_add(a,b); }
};
__attribute__((noinline)) inline v32f act_sigmoid_series(v32f g) {
  return sigmoid_series<ActivationSeries>(g);
}
#endif

__attribute__((noinline)) inline v32f act_exp_carry(v32f x) {
  x=aie::min(aie::max(x,act_v(-87.f)),act_v(88.f));
  const auto t=act_mul_carry(x,act_v(1.44269504f));
  const auto n=aie::to_fixed<int32_t>(t,0);
  const auto f=act_add(t,act_neg(aie::to_float<float>(n,0)));
  auto p=act_v(1.54035304e-4f);
  // Keep Horner evaluation in a loop: six unrolled call sites exceed the
  // 16 KiB program budget when linked with corrected GEMV and segment carry.
  static const float coefficients[]={1.33335581e-3f,9.61812911e-3f,5.55041087e-2f,
                                      2.40226507e-1f,6.93147181e-1f,1.f};
#pragma clang loop unroll(disable)
  for (unsigned i=0;i<6;++i)
    p=act_add(act_mul_carry(p,f),act_v(coefficients[i]));
  const auto bits=aie::upshift(aie::add(n,aie::broadcast<int32_t,32>(127)),23);
  return act_mul_carry(p,bits.cast_to<float>());
}
#if DENSE_ACT_SERIES
__attribute__((noinline)) inline v32f act_sigmoid_carry(v32f g) {
  const auto d=act_add(act_exp_carry(act_neg(g)),act_v(1.f));
  auto r=aie::inv(d);
#pragma clang loop unroll(disable)
  for(unsigned i=0;i<2;++i)
    r=act_mul_carry(r,act_add(act_v(2.f),act_neg(act_mul_carry(d,r))));
  const auto small=aie::le(aie::abs(g),act_v(.5f));
  // Mask speculative polynomial lanes outside its interval before evaluation.
  return aie::select(r,act_sigmoid_series(aie::select(act_v(0.f),g,small)),small);
}
#endif
__attribute__((noinline)) inline v32f act_gated_silu_carry(v32f g,v32f u) {
#if DENSE_ACT_SERIES
  const auto r=act_sigmoid_carry(g);
#else
  const auto d=act_add(act_exp_carry(act_neg(g)),act_v(1.f));
  auto r=aie::inv(d);
  r=act_mul_carry(r,act_add(act_v(2.f),act_neg(act_mul_carry(d,r))));
  r=act_mul_carry(r,act_add(act_v(2.f),act_neg(act_mul_carry(d,r))));
#endif
  return act_mul_carry(act_mul_carry(g,r),u);
}
