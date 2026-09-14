#include "moe_batch.h"
extern "C" {
void mb_zero_y(float *__restrict c) { mb_zero_n(c, 512); }
}
