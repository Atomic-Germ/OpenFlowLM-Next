#include "dnx.h"
extern "C" {
void dnx_ofin(const float *__restrict ds, float *__restrict ye, int32_t hf) {
  dnx_ofin_half(ds, ye, (unsigned)hf);
}
}
