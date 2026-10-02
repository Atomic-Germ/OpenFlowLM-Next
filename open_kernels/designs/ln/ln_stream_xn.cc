#include "ln.h"
#include "vecmath_precise.h"
#if LN_NORM_RNE
#include "ln_rne.h"
#endif

extern "C" void ln_stream_xn(const float *__restrict saved, const float *__restrict sums,
                             const bfloat16 *__restrict w, bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#if LN_NORM_RNE
  auto sum = aie::load_v<kV>(sums), low = ln_neg(aie::load_v<kV>(sums+kV));
#pragma clang loop unroll(disable)
  for (unsigned shift=16;shift;shift>>=1) {
    const auto other = aie::shuffle_down_rotate(sum,shift);
    const auto other_low = aie::shuffle_down_rotate(low,shift);
    ln_two_sum(sum,low,other);
    low=ln_add_rne(low,other_low);
  }
  const float total = ln_add_rne(sum,low)[0];
#else
  const float total = aie::reduce_add(aie::load_v<kV>(sums));
#endif
  const float mean = total * (1.0f / kN) + LN_EPS;
  const float inv = srsqrt(mean);
#if LN_TRACE_STATS
  float traced_weighted = 0, traced_output = 0;
#endif
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    accf32 weight(aie::load_v<kV>(w + j));
#if LN_NORM_RNE
    const v32f t = ln_mul_rne(aie::load_v<kV>(saved+j),weight.to_vector<float>());
#if LN_SCALE_CARRY
    accf32 o(ln_scale_carry(aie::load_v<kV>(saved+j),weight.to_vector<float>(),
                          aie::broadcast<float,kV>(inv)));
#else
    accf32 o(ln_mul_rne(t,aie::broadcast<float,kV>(inv)));
#endif
#else
    const v32f t = precise_mulN<kV>(aie::load_v<kV>(saved + j), weight.template to_vector<float>());
    accf32 o(precise_mulN<kV>(t, aie::broadcast<float, kV>(inv)));
#endif
#if LN_TRACE_STATS
    if (j == (LN_TRACE_INDEX / kV) * kV) {
      traced_weighted = t[ LN_TRACE_INDEX % kV ];
      traced_output = o.to_vector<float>()[ LN_TRACE_INDEX % kV ];
    }
#endif
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
#if LN_TRACE_STATS
  // Diagnostic build only: replace xn with FP32 statistics, never inference.
  float *trace = reinterpret_cast<float *>(xn);
  for (unsigned j = 0; j < kN/2; j += kV)
    aie::store_v(trace+j, aie::zeros<float,kV>());
  aie::store_v(trace, aie::load_v<kV>(sums));
  aie::store_v(trace+kV, aie::load_v<kV>(sums+kV));
  trace[64] = total; trace[65] = mean; trace[66] = inv;
  trace[67] = traced_weighted; trace[68] = traced_output;
#endif
}
