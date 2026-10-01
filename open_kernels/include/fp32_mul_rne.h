#pragma once
#include <stdint.h>

// IEEE binary32 product with integer lanes. Four exact 12-bit products retain
// all 48 significand bits; normalization and sticky shifts round once to even.
// O supplies broadcast, select and the low 32 bits of an integer multiply.
template <class O>
static inline typename O::V fp32_mul_rne(typename O::V a, typename O::V b) {
  const auto zero=O::v(0), one=O::v(1), hidden=O::v(0x800000);
  const auto aa=a & O::v(0x7fffffff), bb=b & O::v(0x7fffffff);
  const auto sign=(a^b) & O::v(0x80000000);
  auto ea=(aa>>23) & O::v(255), eb=(bb>>23) & O::v(255);
  auto ma=(aa & O::v(0x7fffff)) | O::choose(ea==zero,zero,hidden);
  auto mb=(bb & O::v(0x7fffff)) | O::choose(eb==zero,zero,hidden);
  // Bias exponents temporarily to allow subnormal normalization unsigned.
  ea=O::choose(ea==zero,one,ea)+O::v(32);
  eb=O::choose(eb==zero,one,eb)+O::v(32);
#pragma clang loop unroll(full)
  for (unsigned s=16;s;s>>=1) {
    const auto ca=ma<O::v(1u<<(24-s)), cb=mb<O::v(1u<<(24-s));
    ma=O::choose(ca,ma<<s,ma);ea=ea-O::choose(ca,O::v(s),zero);
    mb=O::choose(cb,mb<<s,mb);eb=eb-O::choose(cb,O::v(s),zero);
  }
  const auto mask=O::v(4095);
  const auto p0=O::mul(ma&mask,mb&mask);
  const auto p1=O::mul(ma>>12,mb&mask)+O::mul(ma&mask,mb>>12)+(p0>>12);
  const auto lo=(p0&mask)|((p1&mask)<<12);
  const auto hi=O::mul(ma>>12,mb>>12)+(p1>>12);
  const auto wide=hi>=hidden;
  auto m=O::choose(wide,(hi<<3)|(lo>>21)|O::choose((lo&O::v(0x1fffff))!=zero,one,zero),
                       (hi<<4)|(lo>>20)|O::choose((lo&O::v(0xfffff))!=zero,one,zero));
  // Encoded result exponent plus 256, to keep underflow positive.
  auto e=ea+eb+O::v(65)+O::choose(wide,one,zero);
  auto gap=O::choose(e<O::v(257),O::v(257)-e,zero);
  gap=O::choose(gap>O::v(31),O::v(31),gap);
#pragma clang loop unroll(full)
  for (unsigned s=16;s;s>>=1) {
    const auto jam=(m>>s)|O::choose((m&O::v((1u<<s)-1))!=zero,one,zero);
    m=O::choose((gap&O::v(s))!=zero,jam,m);
  }
  e=O::choose(e<O::v(257),O::v(257),e);
  const auto low=m&O::v(7);
  auto mant=(m>>3)+O::choose((low>O::v(4))|((low==O::v(4))&(((m>>3)&one)!=zero)),one,zero);
  const auto carry=mant>=O::v(1u<<24);
  mant=O::choose(carry,mant>>1,mant);e=e+O::choose(carry,one,zero);
  auto exponent=O::choose(mant<hidden,zero,e-O::v(256));
  auto result=sign|(exponent<<23)|(mant&O::v(0x7fffff));
  const auto inf=O::v(0x7f800000);
  result=O::choose(e>=O::v(511),sign|inf,result);
  result=O::choose((aa==zero)|(bb==zero),sign,result);
  result=O::choose((aa==inf)|(bb==inf),sign|inf,result);
  result=O::choose(((aa==zero)&(bb==inf))|((bb==zero)&(aa==inf)),O::v(0x7fc00000),result);
  result=O::choose(bb>inf,b|O::v(0x400000),result);
  return O::choose(aa>inf,a|O::v(0x400000),result);
}
