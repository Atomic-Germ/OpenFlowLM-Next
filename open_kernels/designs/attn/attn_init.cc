#include "attn.h"
extern "C" {
void attn_init(float *__restrict oacc, float *__restrict ml) { attn_init_impl(oacc, ml); }
}
