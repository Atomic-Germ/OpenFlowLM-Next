// The job count of this dispatch's mode into nj[0].
#include "dxl_job.h"

extern "C" {
void dxl_njobs(const int32_t *__restrict table, const int32_t *__restrict rtp, int32_t *__restrict nj) {
  nj[0] = table[(unsigned)rtp[0] * (1u + DXL_JMAX * F_N)];
}
}
