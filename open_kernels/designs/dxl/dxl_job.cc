// Job j of this dispatch's mode (rtp[0]) into jp, the row every other job kernel reads: its
// first four fields are the loop counts [slices, bands, chunks per band per slice, bands drained].
#include "dxl_job.h"

extern "C" {
void dxl_job(const int32_t *__restrict table, const int32_t *__restrict rtp, int32_t j, int32_t *__restrict jp) {
  const int32_t *r = dxl_job_row(table, rtp, j);
  for (unsigned i = 0; i < F_N; ++i) jp[i] = r[i];
}
}
