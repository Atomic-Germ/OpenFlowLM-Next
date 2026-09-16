#include "moe_batch.h"
extern "C" {
void mb_zero_ug(float *__restrict c) { mb_zero_n(c, 1024); }
}
