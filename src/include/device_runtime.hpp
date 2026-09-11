/// \file device_runtime.hpp
/// \brief Selects the NPU runtime backend (XRT or HRX) at build time and exposes
///        it to the rest of the codebase under a single neutral alias namespace
///        `oflm_rt`, so device-facing code stays backend-agnostic.
/// \note  The backend is chosen via the OFLM_USE_HRX build flag wired in
///        src/CMakeLists.txt (0 = XRT default, 1 = HRX):
///          - OFLM_USE_HRX defined   -> HRX amdxdna runtime  (namespace hrx)
///          - OFLM_USE_HRX undefined -> Xilinx Run Time (XRT) (namespace xrt)
///        Both backends expose the same device-facing surface used across OFLM
///        (device, bo, ext::bo, kernel, hw_context, run, runlist, xclbin,
///        info::device), so referencing them through `oflm_rt::` keeps a single
///        source tree building against either runtime.
#pragma once

#if defined(OFLM_USE_HRX)
#include "hrx_cpp/hrx_cpp.hpp"
namespace oflm_rt = hrx;
#else
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_device.h"
#include "xrt/experimental/xrt_ext.h"
namespace oflm_rt = xrt;
#endif
