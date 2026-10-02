#pragma once

// Compensated finite, normal-range FP32 products. O provides IEEE add/sub/mul
// and high(x), which clears the lower twelve significand bits. Splitting into
// two twelve-bit parts makes each component product exact. This is not an
// overflow/underflow-safe replacement for the general IEEE multiplier.
template <class O>
static inline typename O::V fp32_product_low(typename O::V a, typename O::V b,
                                            typename O::V product) {
  const auto ah=O::high(a), al=O::sub(a,ah);
  const auto bh=O::high(b), bl=O::sub(b,bh);
  const auto e1=O::sub(product,O::mul(ah,bh));
  const auto e2=O::sub(e1,O::mul(al,bh));
  const auto e3=O::sub(e2,O::mul(ah,bl));
  return O::sub(O::mul(al,bl),e3);
}

template <class O>
static inline typename O::V fp32_scale_carry(typename O::V x, typename O::V w,
                                            typename O::V inv) {
  const auto weighted=O::mul(x,w);
  const auto weighted_low=fp32_product_low<O>(x,w,weighted);
  const auto result=O::mul(weighted,inv);
  const auto result_low=fp32_product_low<O>(weighted,inv,result);
  return O::add(result,O::add(result_low,O::mul(weighted_low,inv)));
}
