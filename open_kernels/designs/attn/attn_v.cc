#include "attn.h"
extern "C" {
void attn_v(const float *__restrict vh, bfloat16 *__restrict vout, int h) { attn_v_impl(vh, vout, h); }
}
