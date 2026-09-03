/// \file npu_utils_matmul.hpp
/// \brief NPU2 BF16 matmul backend for the open embedding engine.
#pragma once

#include <cstdint>
#include <string>
#include <memory>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace open_embedding {

class NpuMatmul {
public:
    NpuMatmul();
    ~NpuMatmul();
    NpuMatmul(const NpuMatmul&) = delete;
    NpuMatmul& operator=(const NpuMatmul&) = delete;

    bool init(const std::string& asset_dir, const std::string& device_id);
    bool enabled() const;

    int m_pad_for(int K, int N, int M) const;

    bool matmul_bf16(int M, int K, int N,
                     const uint16_t* a,
                     const uint16_t* b,
                     float* c);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace open_embedding
