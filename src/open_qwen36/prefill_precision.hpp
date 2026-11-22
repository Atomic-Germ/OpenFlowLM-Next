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
    double split_ms = 0;       ///< input high/low splits (the duplicate one is PR4B's target)
    double copy_ms = 0;        ///< intermediate output copies
    double accumulate_ms = 0;  ///< elementwise sums
    long long calls = 0;       ///< completed four-product folds
};

namespace detail {
using FoldClock = std::chrono::steady_clock;
inline double fold_ms(FoldClock::time_point t) {
    return std::chrono::duration<double, std::milli>(FoldClock::now() - t).count();
}
}

struct InputWorkspace {
    std::vector<float> low, output;

    template<class Dispatch>
    const float* run(const float* x, size_t inputs, size_t outputs, Dispatch dispatch,
                     FoldPhaseTiming* timing = nullptr) {
        auto t = detail::FoldClock::now();
        low.resize(inputs);
        for (size_t i = 0; i < inputs; ++i) {
            const float hi = bf16_to_f32(f32_to_bf16(x[i]));
            if (!std::isfinite(x[i]) || !std::isfinite(hi))
                throw std::runtime_error("paired GEMM requires finite BF16-range inputs");
            low[i] = x[i] - hi;
        }
        if (timing) timing->split_ms += detail::fold_ms(t);
        // Copy before the second dispatch reuses its mapped (or split-sum) output.
        // Never dirty a device output buffer: read_back's CLFLUSH can overwrite it.
        const float* high_output = dispatch(x);
        t = detail::FoldClock::now();
        output.assign(high_output, high_output + outputs);
        if (timing) timing->copy_ms += detail::fold_ms(t);
        const float* low_output = dispatch(low.data());
        // Clock AFTER the dispatch: the wait belongs to the NPU, not the fold.
        t = detail::FoldClock::now();
        for (size_t i = 0; i < outputs; ++i) output[i] += low_output[i];
        if (timing) timing->accumulate_ms += detail::fold_ms(t);
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
    const float* run(const float* x, size_t inputs, size_t outputs, Dispatch dispatch,
                     FoldPhaseTiming* timing = nullptr) {
        // The inner fold already booked its own split/copy/accumulate; these
        // windows are the outer fold's own copy and final sum only.
        const float* h = high.run(x, inputs, outputs,
            [&](const float* input) { return dispatch(false, input); }, timing);
        auto t = detail::FoldClock::now();
        output.assign(h, h + outputs);
        if (timing) timing->copy_ms += detail::fold_ms(t);
        const float* l = low.run(x, inputs, outputs,
            [&](const float* input) { return dispatch(true, input); }, timing);
        t = detail::FoldClock::now();
        for (size_t i = 0; i < outputs; ++i) output[i] += l[i];
        if (timing) { timing->accumulate_ms += detail::fold_ms(t); ++timing->calls; }
        return output.data();
    }
};
}
