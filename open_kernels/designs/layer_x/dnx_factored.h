#pragma once
// Compensated old-state readout plus rank-one update. Shared noinline helpers
// avoid duplicate program-memory use across the full-layer entry points.
// Reuse triple-BF16 state update and delta.
#define dnx_delta_head triple_delta_head
#define update_slice triple_update_slice
#define dnx_row_half triple_row_half
#define dnx_ofin_half triple_ofin_half
#include "dnx_precise.h"
#undef dnx_delta_head
#undef update_slice
#undef dnx_row_half
#undef dnx_ofin_half
static constexpr unsigned DS_O_LOW=768,DS_DOT=896;
struct compensated {
  vf hi,lo;
  static compensated zero() { return {aie::zeros<float,kV>(),aie::zeros<float,kV>()}; }
  // TwoSum: recover the rounding error without a magnitude ordering assumption.
  void add(vf x) {
    vf s=aie::add(hi,x),b=aie::sub(s,hi);
    vf e=aie::add(aie::sub(hi,aie::sub(s,b)),aie::sub(x,b));
    lo=aie::add(lo,e);hi=s;
  }
};
// BF16 products are exactly representable in FP32. Keep their sum in two parts.
inline __attribute__((noinline)) void compensated_mac(compensated &a,vf x,float scalar) {
  vb h,l,r,sh,sl,sr;split3(x,h,l,r);
  split3(aie::broadcast<float,kV>(scalar),sh,sl,sr);
  a.add(aie::mul(h,sh).template to_vector<float>());
  a.add(aie::mul(h,sl).template to_vector<float>());
  a.add(aie::mul(l,sh).template to_vector<float>());
  a.add(aie::mul(l,sl).template to_vector<float>());
  a.add(aie::mul(h,sr).template to_vector<float>());
  a.add(aie::mul(r,sh).template to_vector<float>());
  a.add(aie::mul(l,sr).template to_vector<float>());
  a.add(aie::mul(r,sl).template to_vector<float>());
  a.add(aie::mul(r,sr).template to_vector<float>());
}
static inline void dnx_delta_head(float *__restrict ds) {
  triple_delta_head(ds);
  compensated dot=compensated::zero();
#pragma clang loop unroll(disable)
  for(unsigned i=0;i<kD;++i) compensated_mac(dot,aie::broadcast<float,kV>(ds[i]),ds[128+i]);
  ds[DS_DOT]=dot.hi[0];ds[DS_DOT+1]=dot.lo[0];
#pragma clang loop unroll(disable)
  for(unsigned j=0;j<kD;j+=kV) aie::store_v(ds+DS_O_LOW+j,aie::zeros<float,kV>());
}
static inline void update_slice(float *__restrict S,float *__restrict ds,unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for(unsigned j=0;j<kD;j+=kV) {
    compensated out={aie::load_v<kV>(ds+DS_O+j),aie::load_v<kV>(ds+DS_O_LOW+j)};
    vf delta=aie::load_v<kV>(ds+DS_DELTA+j);
#pragma clang loop unroll(disable)
    for(unsigned i=0;i<kRowsX;++i) {
      unsigned row=blk*kRowsX+i;
      float k=row<kD?ds[row]:0.f,q=row<kD?ds[128+row]:0.f;
      vf old=aie::load_v<kV>(S+i*kD+j);
      compensated_mac(out,old,q);
      accf16 a=aie::zeros<accfloat,kV>();a=mac3(a,old,ds[384]);a=mac3(a,delta,k);
      aie::store_v(S+i*kD+j,a.template to_vector<float>());
    }
    aie::store_v(ds+DS_O+j,out.hi);aie::store_v(ds+DS_O_LOW+j,out.lo);
  }
}
static inline void dnx_row_half(float *__restrict S,float *__restrict ds,float *__restrict ye,
                                unsigned blk,unsigned i,unsigned hf) {
  if(i==0 && hf==0) update_slice(S,ds,blk);
#pragma clang loop unroll(full)
  for(unsigned j=0;j<kHalf;j+=kV) aie::store_v(ye+j,aie::load_v<kV>(S+i*kD+hf*kHalf+j));
}
static inline void dnx_ofin_half(const float *__restrict ds,float *__restrict ye,unsigned hf) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for(unsigned j=0;j<kHalf;j+=kV) {
    unsigned off=hf*kHalf+j;
    compensated out=compensated::zero();
    compensated_mac(out,aie::load_v<kV>(ds+DS_O+off),ds[384]);
    compensated_mac(out,aie::load_v<kV>(ds+DS_O_LOW+off),ds[384]);
    vf delta=aie::load_v<kV>(ds+DS_DELTA+off);
    compensated_mac(out,delta,ds[DS_DOT]);compensated_mac(out,delta,ds[DS_DOT+1]);
    accf16 a=aie::zeros<accfloat,kV>();
    a=mac3(a,aie::add(out.hi,out.lo),0.088388347648318f);
    aie::store_v(ye+j,a.template to_vector<float>());
  }
}
