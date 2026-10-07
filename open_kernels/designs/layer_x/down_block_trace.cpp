#include "gemv_q4.h"
extern "C" void down_block_trace(const uint8_t *w, const uint8_t *tab,
                                  float *ms, float *out, int32_t kt) {
#if GEMV_Q4_PRODUCT_TRACE
#pragma clang loop unroll(disable)
  for (unsigned i=1120;i<2560;i+=32) aie::store_v(out+i,aie::zeros<float,32>());
#endif
  gemv_q4_tile(w, tab, 4096, kt, kt==0, kt==15, ms, 0, out
#if GEMV_Q4_PRODUCT_TRACE
              , out+1120
#endif
              );
  aie::store_v(out+1024, aie::load_v<32>(ms));
  const float *cp=(const float *)(tab+4*4096+4096/4+4096/16);
  aie::store_v(out+1056, aie::load_v<32>(cp));
  const bfloat16 *hi=(const bfloat16 *)(tab+2*4096+4096/8)+kt*8;
  const bfloat16 *lo=hi+4096/32;
  const bfloat16 *tail=(const bfloat16 *)(tab+4*4096+4096/4)+kt*8;
  aie::accum<accfloat,32> sums;
  sums.from_vector(aie::concat(aie::load_v<8>(hi),aie::load_v<8>(lo),
                               aie::load_v<8>(tail),aie::zeros<bfloat16,8>()));
  aie::store_v(out+1088,sums.to_vector<float>());
}
