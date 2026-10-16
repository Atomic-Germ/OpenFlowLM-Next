#pragma once
// Must match recipes/dxl.py job_table: per mode (rtp[0]: 0 verify, 1 draft) a job count, then FIELDS-order rows.
#include "dxl_gemv.h"

#ifndef DXL_JMAX
#define DXL_JMAX 32
#endif
enum { F_S, F_BT, F_CPS, F_NDRAIN, F_KS, F_MODE, F_OFF, F_B0, F_S0, F_KT, F_DMODE, F_G0, F_NPREP, F_N };

#ifndef DXL_TABS
#define DXL_TABS 1       // slices whose tables a core keeps (recipes/dxl.py TABS)
#endif
#ifndef DXL_TABK
#define DXL_TABK 512
#endif
// Where slice s of this job's tables sits in tab: a job on at most DXL_TABS slices keeps one table a slice, so the next job on the
// same activation skips its prep; the LoRA's z rows (MODE 2) have a region after them; every other job rebuilds slot 0.
static inline unsigned dxl_tab_off(const int32_t *__restrict jp, unsigned s) {
  if (DXL_TABS == 1) return 0;
  if (jp[F_MODE] == 2) return DXL_TABS * kL * gemv_q4_tab_bytes(DXL_TABK);
  return (unsigned)jp[F_S] <= DXL_TABS ? s * kL * gemv_q4_tab_bytes((unsigned)jp[F_KS]) : 0u;
}

static inline const int32_t *dxl_job_row(const int32_t *__restrict table, const int32_t *__restrict rtp, int32_t j) {
  return table + (unsigned)rtp[0] * (1u + DXL_JMAX * F_N) + 1u + (unsigned)j * F_N;
}
