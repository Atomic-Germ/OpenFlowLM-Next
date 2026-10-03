#include "dense_activation_carry.h"
extern "C" void activation_probe(const float *input,float *output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto g=aie::load_v<32>(input), u=aie::load_v<32>(input+32);
  const auto e=act_exp_carry(act_neg(g));
  const auto d=act_add(e,act_v(1.f));
  const auto r0=aie::inv(d);
  const auto r1=act_mul_carry(r0,act_add(act_v(2.f),act_neg(act_mul_carry(d,r0))));
  const auto r2=act_mul_carry(r1,act_add(act_v(2.f),act_neg(act_mul_carry(d,r1))));
  const auto s=act_mul_carry(g,r2), gu=act_mul_carry(g,u);
  aie::store_v(output,e);aie::store_v(output+32,d);
  aie::store_v(output+64,r0);aie::store_v(output+96,r1);aie::store_v(output+128,r2);
  aie::store_v(output+160,s);aie::store_v(output+192,act_mul_carry(s,u));
  aie::store_v(output+224,gu);aie::store_v(output+256,act_mul_carry(gu,r2));
  aie::store_v(output+288,act_gated_silu_carry(g,u));
}
