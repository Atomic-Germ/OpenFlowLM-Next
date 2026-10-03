#pragma once
#include "fp32_scale_carry.h"

// Five finite normal-range factors with product residuals retained until the
// final FP32 rounding. O supplies IEEE add/sub/mul/high, as for scale carry.
// This is not an overflow/underflow-safe general product implementation.
template<class O>
static inline typename O::V fp32_product_chain(typename O::V x, typename O::V a,
                                              typename O::V b, typename O::V c,
                                              typename O::V d) {
  auto high = O::mul(x,a);
  auto low = fp32_product_low<O>(x,a,high);
  const typename O::V factors[] = {b,c,d};
#pragma clang loop unroll(disable)
  for (unsigned i=0;i<3;++i) {
    const auto next = O::mul(high,factors[i]);
    low = O::add(O::mul(low,factors[i]),fp32_product_low<O>(high,factors[i],next));
    high = next;
  }
  return O::add(high,low);
}
