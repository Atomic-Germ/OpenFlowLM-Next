#include "moe_batch.h"
extern "C" {
void mb_silu(const float *__restrict u, const float *__restrict g, float *__restrict h) {
  mb_silu_tile(u, g, reinterpret_cast<bfloat16 *>(h));
}
}
