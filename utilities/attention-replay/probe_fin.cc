#include "attn.h"
// Only the final store differs. The control calls the production BF16 finish
// and promotes its result; the FP32 experiment retains the same pre-store math.
extern "C" void probe_fin(const float *oacc, const float *ml, const float *g0,
                          const float *g1, float *out, int hp) {
#if !REPLAY_FP32
  alignas(128) bfloat16 temp[kKVH*kHD];
  attn_fin_impl(oacc, ml, g0, g1, temp, hp);
  for(unsigned j=0;j<kOGH*kHD;j+=kV) {
    accf32 a; a.from_vector(aie::load_v<kV>(temp+j));
    aie::store_v(out+j,a.to_vector<float>());
  }
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  for(unsigned i=0;i<kOGH;++i) {
    const unsigned h=kOGH*hp+i;
#if ATTN_INTFP
    const float inv=srecip_rn(ml[kMLS+h]);
#else
    const float inv=1.0f/ml[kMLS+h];
#endif
    const float *o=oacc+h*kHD;
    const float *g=(i<kHPE)?g0+i*kHD:g1+(i-kHPE)*kHD;
    for(unsigned j=0;j<kHD;j+=kV) {
      const v32f on=fscaleN<32>(aie::load_v<kV>(o+j),inv);
      const v32f r=fmul32(on,vsigmoidN<32>(aie::load_v<kV>(g+j)));
      aie::store_v(out+i*kHD+j,r);
    }
  }
#endif
}
