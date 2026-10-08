#pragma once
// Must match recipes/dxl.py job_table: per mode (rtp[0]: 0 verify, 1 draft) a job count, then FIELDS-order rows.
#include "dxl_gemv.h"

#ifndef DXL_JMAX
#define DXL_JMAX 32
#endif
enum { F_S, F_BT, F_CPS, F_NDRAIN, F_KS, F_MODE, F_OFF, F_B0, F_S0, F_KT, F_DMODE, F_G0, F_N };

static inline const int32_t *dxl_job_row(const int32_t *__restrict table, const int32_t *__restrict rtp, int32_t j) {
  return table + (unsigned)rtp[0] * (1u + DXL_JMAX * F_N) + 1u + (unsigned)j * F_N;
}
