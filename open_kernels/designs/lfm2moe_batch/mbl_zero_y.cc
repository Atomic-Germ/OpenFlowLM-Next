#include "mbl.h"
extern "C" {
void mbl_zero_y(float *__restrict c) { mbl_zero_n(c, 64 * MBL_NT); }
}
