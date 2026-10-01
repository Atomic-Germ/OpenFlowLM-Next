#include "ln.h"
#include "vecmath_precise.h"

extern "C" void ln_stream_xn(const float *__restrict saved, const float *__restrict sums,
                             const bfloat16 *__restrict w, bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float inv = srsqrt(aie::reduce_add(aie::load_v<kV>(sums)) * (1.0f / kN) + LN_EPS);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    accf32 weight(aie::load_v<kV>(w + j));
    const v32f t = precise_mulN<kV>(aie::load_v<kV>(saved + j), weight.template to_vector<float>());
    accf32 o(precise_mulN<kV>(t, aie::broadcast<float, kV>(inv)));
#if LN_RESIDUAL_RNE
    // BF16-product accumulation may turn -0 into +0. Preserve the exact
    // residual zero's sign, including the sign of the normalization weight.
    const auto bits = aie::load_v<kV>(saved + j).cast_to<uint32_t>();
    const auto zero = aie::eq(aie::bit_and(uint32_t(0x7fffffff), bits), uint32_t(0));
    const auto sign = aie::bit_and(uint32_t(0x80000000), aie::bit_xor(bits, weight.to_vector<float>().cast_to<uint32_t>()));
    const auto z = aie::downshift(sign, 16).pack<uint16_t>().cast_to<bfloat16>();
    aie::store_v(xn + j, aie::select(o.template to_vector<bfloat16>(), z, zero));
#else
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
#endif
  }
}
