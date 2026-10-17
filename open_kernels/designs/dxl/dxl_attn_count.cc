// A loop count through memory: llc spent 9+ minutes scheduling the constant skip loops this replaces.
#include <stdint.h>

extern "C" {
void dxl_attn_count(int32_t *__restrict pb, int32_t n) {
  pb[7] = n;
}
}
