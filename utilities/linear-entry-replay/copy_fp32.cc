#include "dn_glue.h"
extern "C" void linear_copy_fp32(const float *__restrict src, float *__restrict dst, int i) {
#pragma clang loop unroll(disable)
  for (unsigned j=0; j<1024; j+=kV)
    aie::store_v(dst+(unsigned)i*1024+j,aie::load_v<kV>(src+j));
}
