#include "dn_glue.h"

// Each call consumes 512 channels of all three FP32 history rows. The two
// weight elements still contain the unchanged 1024-channel BF16 tap raster.
extern "C" void history_conv(const float *__restrict q,
    const float *__restrict s0, const float *__restrict s1, const float *__restrict s2,
    const bfloat16 *__restrict w01, const bfloat16 *__restrict w23,
    float *__restrict ns0, float *__restrict ns1, float *__restrict ns2,
    float *__restrict qk, float *__restrict vt, int tile, int base, int half) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned t = tile + base;
  const unsigned start = half * 512;
  float *dst = (t < kKeyTiles ? qk + t * kTile : vt) + start;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 512; j += kV) {
    const auto x = aie::load_v<kV>(q+j);
    const auto a0 = aie::load_v<kV>(s0+j);
    const auto a1 = aie::load_v<kV>(s1+j);
    const auto a2 = aie::load_v<kV>(s2+j);
    accf32 a = aie::zeros<accfloat,kV>();
    a = mac_vv(a, a0, aie::load_v<kV>(w01+start+j));
    a = mac_vv(a, a1, aie::load_v<kV>(w01+kTile+start+j));
    a = mac_vv(a, a2, aie::load_v<kV>(w23+start+j));
    a = mac_vv(a, x, aie::load_v<kV>(w23+kTile+start+j));
    const auto af = a.template to_vector<float>();
    aie::store_v(dst+j, fmul32(af,vsigmoid32(af)));
    aie::store_v(ns0+j,a1); aie::store_v(ns1+j,a2); aie::store_v(ns2+j,x);
  }
  if (t < kKeyTiles) {
#pragma clang loop unroll(disable)
    for (unsigned hh=0; hh<512/kHD; ++hh) {
      float *hp=dst+hh*kHD;
      accf32 ss=aie::zeros<accfloat,kV>();
#pragma clang loop unroll(disable)
      for (unsigned j=0; j<kHD; j+=kV) {
        v32b ch,cl; split32(aie::load_v<kV>(hp+j),ch,cl);
        ss=aie::mac(ss,ch,ch);ss=aie::mac(ss,ch,cl);ss=aie::mac(ss,ch,cl);
      }
      const float inv=srsqrt(aie::reduce_add(ss.template to_vector<float>())+1e-6f);
      const bfloat16 ih=(bfloat16)inv,il=(bfloat16)(inv-(float)ih);
#pragma clang loop unroll(disable)
      for (unsigned j=0; j<kHD; j+=kV) {
        accf32 o=aie::zeros<accfloat,kV>();
        o=mac_vs(o,aie::load_v<kV>(hp+j),ih,il);
        aie::store_v(hp+j,o.template to_vector<float>());
      }
    }
  }
}
