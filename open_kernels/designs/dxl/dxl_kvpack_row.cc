// Row cnt[0] of the L rows' KV, then the next: one core packs them all where the south ports run short (dxl.py KV_ONE).
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void dxl_kvpack_row(const bfloat16 *__restrict k, const bfloat16 *__restrict v, bfloat16 *__restrict rows,
                    int32_t *__restrict cnt, int32_t first) {
  if (first) cnt[0] = 0;
  bfloat16 *__restrict o = rows + (unsigned)cnt[0] * 2u * DXL_KVW;
  for (unsigned j = 0; j < DXL_KVW; j += 32) aie::store_v(o + j, aie::load_v<32>(k + j));
  for (unsigned j = 0; j < DXL_KVW; j += 32) aie::store_v(o + DXL_KVW + j, aie::load_v<32>(v + j));
  cnt[0] += 1;
}
}
