// Preserve BF16 high/low input products and host-owned sums for paired prefill.
#pragma once
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36::prefill_precision {

// OFLM_OPEN_FOLD_PROFILE: exclusive, non-overlapping windows around the host
// fold. Dispatch time is never inside a window. Null in production.
struct FoldPhaseTiming {
    double split_ms = 0;       ///< input high/low splits, one per weight half
    double copy_ms = 0;        ///< intermediate output copies
    double accumulate_ms = 0;  ///< elementwise sums
    long long calls = 0;       ///< completed four-product folds
};

namespace detail {
using FoldClock = std::chrono::steady_clock;
template<class Clock>
inline double fold_ms(typename Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
}

struct InputWorkspace {
    std::vector<float> low, output;

    template<class Clock = detail::FoldClock, class Dispatch>
    const float* run(const float* x, size_t inputs, size_t outputs, Dispatch dispatch,
                     FoldPhaseTiming* timing = nullptr) {
        typename Clock::time_point t{};
        if (timing) t = Clock::now();
        low.resize(inputs);
        for (size_t i = 0; i < inputs; ++i) {
            const float hi = bf16_to_f32(f32_to_bf16(x[i]));
            if (!std::isfinite(x[i]) || !std::isfinite(hi))
                throw std::runtime_error("paired GEMM requires finite BF16-range inputs");
            low[i] = x[i] - hi;
        }
        if (timing) timing->split_ms += detail::fold_ms<Clock>(t);
        // Copy before the second dispatch reuses its mapped (or split-sum) output.
        // Never dirty a device output buffer: read_back's CLFLUSH can overwrite it.
        const float* high_output = dispatch(x);
        if (timing) t = Clock::now();
        output.assign(high_output, high_output + outputs);
        if (timing) timing->copy_ms += detail::fold_ms<Clock>(t);
        const float* low_output = dispatch(low.data());
        // Dispatch residence is measured separately from host folding.
        if (timing) t = Clock::now();
        for (size_t i = 0; i < outputs; ++i) output[i] += low_output[i];
        if (timing) timing->accumulate_ms += detail::fold_ms<Clock>(t);
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
    template<class Clock = detail::FoldClock, class Dispatch>
    const float* run(const float* x, size_t inputs, size_t outputs, Dispatch dispatch,
                     FoldPhaseTiming* timing = nullptr) {
        // The inner fold already booked its own split/copy/accumulate; these
        // windows are the outer fold's own copy and final sum only.
        const float* h = high.run<Clock>(x, inputs, outputs,
            [&](const float* input) { return dispatch(false, input); }, timing);
        typename Clock::time_point t{};
        if (timing) t = Clock::now();
        output.assign(h, h + outputs);
        if (timing) timing->copy_ms += detail::fold_ms<Clock>(t);
        const float* l = low.run<Clock>(x, inputs, outputs,
            [&](const float* input) { return dispatch(true, input); }, timing);
        if (timing) t = Clock::now();
        for (size_t i = 0; i < outputs; ++i) output[i] += l[i];
        if (timing) { timing->accumulate_ms += detail::fold_ms<Clock>(t); ++timing->calls; }
        return output.data();
    }
};
}
