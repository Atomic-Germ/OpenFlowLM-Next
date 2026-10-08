// exp2_probe.cc -- what aie::exp2<bfloat16>(fp32) returns on aie2p, for exp2_probe.py.
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {
// out[i] = exp2(in[i]) through the hardware bf16 exp2, N values.
void exp2_hw(float *__restrict in, bfloat16 *__restrict out, int32_t n) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < n; i += 16)
    aie::store_v(out + i, aie::exp2<bfloat16>(aie::load_v<16>(in + i)));
}
}
