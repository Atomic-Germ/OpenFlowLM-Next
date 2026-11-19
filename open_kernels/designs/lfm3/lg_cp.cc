// Row k of a core-local [rows][128] f32 buffer from a fifo element (the conv taps: they are held while the
// state rows and B / C / u stream, so the fifo never has to hold a whole element's inputs at once).
#include "vecmath.h"

extern "C" {
void lg_cp(const float *__restrict src, float *__restrict dst, int32_t k) {
  float *__restrict d = dst + k * 128;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 128; j += 32) aie::store_v(d + j, aie::load_v<32>(src + j));
}
}
