// vecmath_probe.cc -- what aie2p's vector tanh and bf16 inv / invsqrt return, for vecmath_probe.py.
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {
// out[0:n] = tanh(in) (hardware, fp32 in -> bf16 out); out[n:2n] = inv(bf16(|in|)+0.25);
// out[2n:3n] = invsqrt(bf16(|in|)+0.25).
void vecmath_hw(float *__restrict in, bfloat16 *__restrict out, int32_t n) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> x = aie::load_v<16>(in + i);
    aie::store_v(out + i, aie::tanh<bfloat16>(x));
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::add(aie::abs(x), aie::broadcast<float, 16>(0.25f)));
    aie::vector<bfloat16, 16> b = a.template to_vector<bfloat16>();
    aie::store_v(out + n + i, aie::inv(b));
    aie::store_v(out + 2 * n + i, aie::invsqrt(b));
  }
}
}
