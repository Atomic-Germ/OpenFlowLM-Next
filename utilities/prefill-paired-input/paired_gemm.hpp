// Private diagnostic: run an unchanged BF16-input projection twice.
#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36::paired_gemm {
struct Workspace {
    std::vector<float> low, output;

    template<class Dispatch>
    const float* run(const float* x, size_t inputs, size_t outputs, Dispatch dispatch) {
        low.resize(inputs);
        for (size_t i = 0; i < inputs; ++i) {
            const float hi = bf16_to_f32(f32_to_bf16(x[i]));
            if (!std::isfinite(x[i]) || !std::isfinite(hi))
                throw std::runtime_error("paired GEMM requires finite BF16-range inputs");
            low[i] = x[i] - hi;
        }
        // Copy before the second dispatch reuses its mapped (or split-sum) output.
        // Never dirty a device output buffer: read_back's CLFLUSH can overwrite it.
        const float* high_output = dispatch(x);
        output.assign(high_output, high_output + outputs);
        const float* low_output = dispatch(low.data());
        for (size_t i = 0; i < outputs; ++i) output[i] += low_output[i];
        return output.data();
    }
};
}  // namespace open_qwen36::paired_gemm
