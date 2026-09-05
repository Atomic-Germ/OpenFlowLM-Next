// cache-bust: 1788363877 (IRON hashes only this file, not included headers)
#include "dn_glue.h"
extern "C" {
// tile from a rolled runtime loop; the accumulator starts on tile 0.
void glue_ab(const bfloat16 *__restrict W, const bfloat16 *__restrict xn, float *__restrict acc,
             int tile) {
  glue_ab_tile(W, xn, acc, (unsigned)tile, tile == 0);
}
}
