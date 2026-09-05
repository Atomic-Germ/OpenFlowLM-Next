// cache-bust: 1788363878 (IRON hashes only this file, not included headers)
#include "dn_glue.h"
extern "C" {
// tv = v-tile index (0..3), i = head within the tile (0..7): head h = tv*8 + i.
void glue_emit_fn(const float *__restrict qk, const float *__restrict vt, const float *__restrict decay,
                  const float *__restrict beta, float *__restrict rec, int tv, int i) {
  glue_emit(qk, vt, decay, beta, rec, (unsigned)(tv * 8 + i));
}
}
