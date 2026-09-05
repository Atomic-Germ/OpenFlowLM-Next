#include "dnx.h"
extern "C" {
void dnx_vcopy(const float *__restrict e, float *__restrict ds) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 512; j += kV)
    aie::store_v(ds + DS_VEC + j, aie::load_v<kV>(e + j));
}
}
