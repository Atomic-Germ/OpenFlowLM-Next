// Preserve BF16 high/low input products and host-owned sums for paired prefill.
#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36::prefill_precision {
struct InputWorkspace {
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
}  // namespace open_qwen36::prefill_precision

namespace open_qwen36::prefill_precision {
// Match the validated order exactly: (W_hi X_hi + W_hi X_lo) +
// (W_lo X_hi + W_lo X_lo). Each dispatch may overwrite the same mapped BO.
struct Workspace {
    InputWorkspace high, low;
    std::vector<float> output;
    template<class Dispatch>
    const float* run(const float* x, size_t inputs, size_t outputs, Dispatch dispatch) {
        const float* h = high.run(x, inputs, outputs,
            [&](const float* input) { return dispatch(false, input); });
        output.assign(h, h + outputs);
        const float* l = low.run(x, inputs, outputs,
            [&](const float* input) { return dispatch(true, input); });
        for (size_t i = 0; i < outputs; ++i) output[i] += l[i];
        return output.data();
    }
};
}
