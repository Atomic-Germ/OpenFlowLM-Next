#include "attn.h"
#include <algorithm>
// Emit raw Q hi/lo, O accumulator, padded [max,sum] softmax state in EA-byte blocks.
extern "C" void probe_debug(const ATTN_QT *q, const float *o, const float *ml,
                            uint8_t *out, int block) {
  constexpr unsigned qb=2*kQW*sizeof(bfloat16)/kEA;
  constexpr unsigned ob=kNHL*kHD*sizeof(float)/kEA;
  const uint32_t *src;
  unsigned count=kEA/4;
  if(block<qb) src=reinterpret_cast<const uint32_t*>(q)+block*count;
  else if(block<qb+ob) src=reinterpret_cast<const uint32_t*>(o)+(block-qb)*count;
  else {
    const unsigned start=(block-qb-ob)*count;
    src=reinterpret_cast<const uint32_t*>(ml)+start;
    count=(2*kMLS>start)?std::min(count,2*kMLS-start):0;
  }
  for(unsigned j=0;j<kEA/4;j+=32) {
    auto v=aie::zeros<uint32_t,32>();
    if(j<count) v=aie::load_v<32>(src+j);
    aie::store_v(reinterpret_cast<uint32_t*>(out)+j,v);
  }
}
