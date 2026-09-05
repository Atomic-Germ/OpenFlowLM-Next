#include "dnx.h"
extern "C" {
void dnx_delta(float *__restrict ds) {
  dnx_delta_head(ds);
}
}
