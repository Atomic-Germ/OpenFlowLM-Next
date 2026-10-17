// One KV cache row as the cache lays it out: K then V.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void dxl_kvpack(const bfloat16 *__restrict k, const bfloat16 *__restrict v, bfloat16 *__restrict o) {
  for (unsigned j = 0; j < DXL_KVW; j += 32) aie::store_v(o + j, aie::load_v<32>(k + j));
  for (unsigned j = 0; j < DXL_KVW; j += 32) aie::store_v(o + DXL_KVW + j, aie::load_v<32>(v + j));
}
}
