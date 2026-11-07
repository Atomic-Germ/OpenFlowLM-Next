#include "attn.h"
extern "C" void attn_pair_k(const float *in, const bfloat16 *kn, const float *cs,
                      float *tmp, bfloat16 *out, int e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for(unsigned i=0;i<kHPE;++i) {
    norm_rope(in+i*kHD,kn,cs,tmp);
    for(unsigned j=0;j<kHD;j+=kV) {
      v32b hi,lo; splitN<kV>(aie::load_v<kV>(tmp+j),hi,lo);
      aie::store_v(out+(e*kHPE+i)*kHD+j,hi);
#if !ATTN_KV_PAIR
      lo=aie::zeros<bfloat16,kV>();
#endif
      aie::store_v(out+kKVH*kHD+(e*kHPE+i)*kHD+j,lo);
    }
  }
}
