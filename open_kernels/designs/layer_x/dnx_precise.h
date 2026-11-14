#pragma once
// Triple-BF16 products for the validated factored recurrence.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
static constexpr unsigned kD=128,kV=16,kHalf=64,kRowsX=DNX_ROWS;
static constexpr unsigned DS_VEC=0,DS_T=512,DS_O=640,DS_DELTA=1088;
using vf=aie::vector<float,kV>;
using vb=aie::vector<bfloat16,kV>;
using accf16=aie::accum<accfloat,kV>;
static inline void split3(vf v,vb &h,vb &l,vb &r) {
  accf16 a; a.from_vector(v);
  h=a.template to_vector<bfloat16>();
  a=aie::sub(a,h); l=a.template to_vector<bfloat16>();
  r=aie::sub(a,l).template to_vector<bfloat16>();
}
// All nine products; no native FP32 vector multiplication or scalar soft-float.
inline __attribute__((noinline)) accf16 mac3(accf16 acc,vf a,float scalar) {
  vb ah,al,ar,sh,sl,sr;
  split3(a,ah,al,ar); split3(aie::broadcast<float,kV>(scalar),sh,sl,sr);
  acc=aie::mac(acc,ah,sh); acc=aie::mac(acc,ah,sl); acc=aie::mac(acc,al,sh);
  acc=aie::mac(acc,al,sl); acc=aie::mac(acc,ah,sr); acc=aie::mac(acc,ar,sh);
  acc=aie::mac(acc,al,sr); acc=aie::mac(acc,ar,sl); acc=aie::mac(acc,ar,sr);
  return acc;
}
static inline void dnx_pass1_slice(const float *__restrict S,float *__restrict ds,unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for(unsigned j=0;j<kD;j+=kV) {
    accf16 a=aie::zeros<accfloat,kV>();
    if(blk) a.from_vector(aie::load_v<kV>(ds+DS_T+j));
#pragma clang loop unroll(disable)
    for(unsigned i=0;i<kRowsX;++i) {
      unsigned row=blk*kRowsX+i;
      float k=row<kD?ds[row]:0.f;
      a=mac3(a,aie::load_v<kV>(S+i*kD+j),k);
    }
    aie::store_v(ds+DS_T+j,a.template to_vector<float>());
  }
}
static inline void dnx_delta_head(float *__restrict ds) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for(unsigned j=0;j<kD;j+=kV) {
    accf16 a; a.from_vector(aie::load_v<kV>(ds+256+j));
    a=mac3(a,aie::load_v<kV>(ds+DS_T+j),-ds[384]);
    accf16 d=aie::zeros<accfloat,kV>(); d=mac3(d,a.template to_vector<float>(),ds[385]);
    aie::store_v(ds+DS_DELTA+j,d.template to_vector<float>());
    aie::store_v(ds+DS_O+j,aie::zeros<float,kV>());
  }
}
static inline void update_slice(float *__restrict S,float *__restrict ds,unsigned blk) {
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for(unsigned j=0;j<kD;j+=kV) {
    accf16 out; out.from_vector(aie::load_v<kV>(ds+DS_O+j));
    vf delta=aie::load_v<kV>(ds+DS_DELTA+j);
#pragma clang loop unroll(disable)
    for(unsigned i=0;i<kRowsX;++i) {
      unsigned row=blk*kRowsX+i;
      float k=row<kD?ds[row]:0.f, q=row<kD?ds[128+row]:0.f;
      accf16 a=aie::zeros<accfloat,kV>();
      a=mac3(a,aie::load_v<kV>(S+i*kD+j),ds[384]);
      a=mac3(a,delta,k);
      vf sn=a.template to_vector<float>();
      aie::store_v(S+i*kD+j,sn); out=mac3(out,sn,q);
    }
    aie::store_v(ds+DS_O+j,out.template to_vector<float>());
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
    accf16 a=aie::zeros<accfloat,kV>();
    a=mac3(a,aie::load_v<kV>(ds+DS_O+hf*kHalf+j),0.088388347648318f);
    aie::store_v(ye+j,a.template to_vector<float>());
  }
}
