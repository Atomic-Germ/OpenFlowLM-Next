#include "attn.h"
// One EA-byte FP32 output element: HPE heads and one gate input element.
extern "C" void attn_pair_fin(const float *oacc, const float *ml, const float *gate,
                              float *out, int hp) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for(unsigned i=0;i<kHPE;++i) {
    const unsigned h=kHPE*hp+i;
#if ATTN_INTFP
    const float inv=srecip_rn(ml[kMLS+h]);
#else
    const float inv=1.0f/ml[kMLS+h];
#endif
    const float *o=oacc+h*kHD;
    const float *g=gate+i*kHD;
    for(unsigned j=0;j<kHD;j+=kV) {
      const v32f on=fscaleN<32>(aie::load_v<kV>(o+j),inv);
      const v32f r=fmul32(on,vsigmoidN<32>(aie::load_v<kV>(g+j)));
      aie::store_v(out+i*kHD+j,r);
    }
  }
}
