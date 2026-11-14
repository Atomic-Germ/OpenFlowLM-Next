#include "attn.h"
extern "C" void kv_v(const float *in, bfloat16 *out, int e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for(unsigned j=0;j<(kHPE*kHD);j+=kV) {
    v32b hi,lo; splitN<kV>(aie::load_v<kV>(in+j),hi,lo);
    aie::store_v(out+e*(kHPE*kHD)+j,hi);
#if !KV_PAIR
    lo=aie::zeros<bfloat16,kV>();
#endif
    aie::store_v(out+kKVH*kHD+e*(kHPE*kHD)+j,lo);
  }
}
