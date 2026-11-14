#include "add_rne.h"
extern "C" void refine_sum(float *__restrict high, const float *__restrict low) {
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; ++j) {
    uint32_t h, l;
    __builtin_memcpy(&h, high + j, 4);
    __builtin_memcpy(&l, low + j, 4);
    const uint32_t sum = add_rne(h, l);
    __builtin_memcpy(high + j, &sum, 4);
  }
}
