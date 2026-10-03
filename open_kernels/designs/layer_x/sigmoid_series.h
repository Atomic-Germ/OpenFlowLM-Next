#pragma once
// Odd Taylor series for sigmoid(x)-1/2 through x^9, for |x| <= 1/2.
// The first omitted term is at most 1.1e-9 on this interval. O supplies
// compensated products and exact additions (or scalar IEEE operations in tests).
template<class O> static inline typename O::V sigmoid_series(typename O::V x) {
  const auto square=O::mul(x,x);
  auto p=O::v(31.f/1451520.f);
  static const float c[]={-17.f/80640.f,1.f/480.f,-1.f/48.f,1.f/4.f};
#pragma clang loop unroll(disable)
  for(unsigned i=0;i<4;++i)p=O::add(O::mul(p,square),O::v(c[i]));
  return O::add(O::v(.5f),O::mul(x,p));
}
