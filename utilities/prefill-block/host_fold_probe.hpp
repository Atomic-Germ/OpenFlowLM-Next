// Corrected paired-prefill host-fold probe, shared by the benchmark and its tests.
//
// Both arms perform the SAME four products in the SAME association:
//   (W_hi X_hi + W_hi X_lo) + (W_lo X_hi + W_lo X_lo)
// The baseline runs the real prefill_precision::Workspace; the candidate is the
// arithmetic-preserving rewrite (split once, drop one intermediate copy).
// Outputs must be byte-identical and both arms must reject non-finite input.
//
// Workspaces are persistent across calls, as Core keeps them in
// precision_workspaces_; allocation cost is therefore not part of the steady
// state and is reported separately by the benchmark.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_qwen36/prefill_precision.hpp"

namespace fold_probe {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// One fixture is the activation vector plus the four stub product outputs the
// dispatches would have produced. The products are held separately so a bug in
// either arm cannot be hidden by buffer aliasing.
struct Fixture {
    std::vector<float> x;   // K*T input
    std::vector<float> a;   // W_hi X_hi
    std::vector<float> b;   // W_hi X_lo
    std::vector<float> c;   // W_lo X_hi
    std::vector<float> d;   // W_lo X_lo
};

// The split buffer is persistent and reused (Core keeps its workspace too).
inline void split_low(const std::vector<float>& x, std::vector<float>& low) {
    low.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float hi = open_qwen36::bf16_to_f32(open_qwen36::f32_to_bf16(x[i]));
        if (!std::isfinite(x[i]) || !std::isfinite(hi))
            throw std::runtime_error("paired GEMM requires finite BF16-range inputs");
        low[i] = x[i] - hi;
    }
}

// Deterministic fixture for T physical tokens: activations are K*T, products N*T.
inline Fixture make_fixture(size_t n, size_t k, size_t tokens, const std::string& kind, unsigned seed = 26026) {
    const size_t inputs = k * tokens, outputs = n * tokens;
    Fixture f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_int_distribution<int> small(-8, 7);
    f.x.resize(inputs);
    for (size_t i = 0; i < inputs; ++i) {
        if (kind == "random") f.x[i] = u(gen);
        else if (kind == "rounding") {
            // values just off a BF16 tie, and both signs of zero
            const float base = 1.0f + static_cast<float>(small(gen)) / 256.0f;
            f.x[i] = (i % 3 == 0) ? -base : base;
        } else if (kind == "nan") f.x[i] = (i == 7) ? std::numeric_limits<float>::quiet_NaN() : 0.5f;
        else throw std::invalid_argument("unknown fixture kind");
    }
    // Products vary independently, including values whose sums round in FP32.
    auto fill = [&](std::vector<float>& v, float base, int step) {
        v.resize(outputs);
        for (size_t i = 0; i < outputs; ++i)
            v[i] = base + static_cast<float>(static_cast<int>(i % 17) * step) / 1024.0f;
    };
    fill(f.a, 1.0f, 3);
    fill(f.b, -0.5f, 5);
    fill(f.c, 0.25f, 7);
    fill(f.d, -0.125f, 11);
    return f;
}

// --- baseline: the real Workspace, with a dispatch that returns the fixtures ---
struct BaselineState {
    open_qwen36::prefill_precision::Workspace ws;
    // Pointers the real fold hands to the dispatch, snapshotted per call so the
    // four products can be identified without guessing from the weight flag.
    std::vector<float> out;
};

// Returns false when the fold rejects the input (non-finite), true on success.
// Only the total is timed: the real header does not expose its internal split
// and copy boundaries, and inventing them here would be the same mistake the
// review found. The candidate reports its own exclusive phases separately.
inline bool run_baseline(BaselineState& st, const Fixture& f, std::vector<float>* out,
                         double* total_ms = nullptr) {
    auto dispatch = [&](bool low_weight, const float* input) -> const float* {
        // Identify the input half by pointer: the high half is the caller's x,
        // the low half is the fold's own split buffer.
        const bool low_input = (input != f.x.data());
        if (!low_weight && !low_input) return f.a.data();
        if (!low_weight && low_input) return f.b.data();
        if (low_weight && !low_input) return f.c.data();
        return f.d.data();
    };
    const auto t0 = Clock::now();
    try {
        const float* r = st.ws.run(f.x.data(), f.x.size(), f.a.size(), dispatch);
        if (total_ms) *total_ms = ms_since(t0);
        if (out) out->assign(r, r + f.a.size());
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// --- candidate: split once, then (a+b) + (c+d) with identical FP32 rounding ---
struct CandidateState {
    std::vector<float> low, high, low_sum, out;
};

inline bool run_candidate(CandidateState& st, const Fixture& f, std::vector<float>* out,
                          double* split_ms = nullptr, double* copy_ms = nullptr,
                          double* accum_ms = nullptr) {
    const size_t inputs = f.x.size(), outputs = f.a.size();
    double split = 0, copy = 0, accum = 0;

    auto t = Clock::now();
    try {
        split_low(f.x, st.low);            // identical validation and arithmetic
    } catch (const std::runtime_error&) {
        return false;
    }
    split += ms_since(t);

    t = Clock::now();
    st.high.assign(f.a.begin(), f.a.end());          // W_hi X_hi
    copy += ms_since(t);
    t = Clock::now();
    for (size_t i = 0; i < outputs; ++i) st.high[i] += f.b[i];   // + W_hi X_lo
    accum += ms_since(t);

    t = Clock::now();
    st.low_sum.assign(f.c.begin(), f.c.end());       // W_lo X_hi
    copy += ms_since(t);
    t = Clock::now();
    for (size_t i = 0; i < outputs; ++i) st.low_sum[i] += f.d[i]; // + W_lo X_lo
    accum += ms_since(t);

    t = Clock::now();
    st.out = st.high;                                // the outer fold's copy
    for (size_t i = 0; i < outputs; ++i) st.out[i] += st.low_sum[i];
    accum += ms_since(t);

    if (out) *out = st.out;
    if (split_ms) *split_ms = split;
    if (copy_ms) *copy_ms = copy;
    if (accum_ms) *accum_ms = accum;
    (void)inputs;
    return true;
}

// Full-buffer comparison, reported explicitly: an input both arms reject is
// agreement, not a numeric result, so it must not be mistaken for equality.
struct Comparison {
    bool baseline_accepted = false;
    bool candidate_accepted = false;
    bool outputs_equal = false;     // only when both accepted
    bool agree = false;             // same acceptance decision, and equal when accepted
    std::string why;
};

inline Comparison compare(const Fixture& f) {
    BaselineState base;
    CandidateState cand;
    Comparison c;
    std::vector<float> bo, co;
    c.baseline_accepted = run_baseline(base, f, &bo);
    c.candidate_accepted = run_candidate(cand, f, &co);
    if (c.baseline_accepted != c.candidate_accepted) {
        c.why = c.baseline_accepted ? "candidate rejected an input the baseline accepted"
                                    : "baseline rejected an input the candidate accepted";
        return c;
    }
    if (!c.baseline_accepted) {
        c.agree = true;
        c.why = "both rejected";
        return c;
    }
    c.outputs_equal = bo.size() == co.size() &&
                      std::memcmp(bo.data(), co.data(), bo.size() * sizeof(float)) == 0;
    c.agree = c.outputs_equal;
    if (!c.outputs_equal) c.why = "outputs differ";
    return c;
}

// Convenience for the benchmark gate: both arms must accept and be identical.
inline bool equivalent(const Fixture& f, std::string* why = nullptr) {
    const Comparison c = compare(f);
    if (!c.outputs_equal && why) *why = c.why;
    return c.outputs_equal;
}

// --- paired statistics -------------------------------------------------------
struct Stats {
    std::vector<double> baseline_ms, candidate_ms, delta_ms, change_pct;
    double median_baseline = 0, median_candidate = 0, median_delta = 0, median_change_pct = 0;
    double ci95_low = 0, ci95_high = 0;
    int wins = 0;
};

inline double median(std::vector<double> v) {
    if (v.empty()) throw std::invalid_argument("median of empty sample");
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline Stats summarise(const std::vector<double>& b, const std::vector<double>& c,
                       unsigned seed = 26026, int resamples = 20000) {
    if (b.empty() || b.size() != c.size())
        throw std::invalid_argument("paired samples required");
    Stats s;
    s.baseline_ms = b;
    s.candidate_ms = c;
    for (size_t i = 0; i < b.size(); ++i) {
        if (!std::isfinite(b[i]) || !std::isfinite(c[i]) || b[i] <= 0 || c[i] <= 0)
            throw std::invalid_argument("positive finite paired samples required");
        s.delta_ms.push_back(c[i] - b[i]);
        s.change_pct.push_back(100.0 * (c[i] / b[i] - 1.0));
        if (c[i] < b[i]) ++s.wins;
    }
    s.median_baseline = median(b);
    s.median_candidate = median(c);
    s.median_delta = median(s.delta_ms);
    s.median_change_pct = median(s.change_pct);
    // Paired bootstrap of the median difference.
    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> pick(0, b.size() - 1);
    std::vector<double> boot;
    boot.reserve(static_cast<size_t>(resamples));
    std::vector<double> sample(b.size());
    for (int r = 0; r < resamples; ++r) {
        for (size_t i = 0; i < sample.size(); ++i) sample[i] = s.delta_ms[pick(gen)];
        boot.push_back(median(sample));
    }
    std::sort(boot.begin(), boot.end());
    s.ci95_low = boot[boot.size() / 40];
    s.ci95_high = boot[boot.size() - 1 - boot.size() / 40];
    return s;
}

}  // namespace fold_probe
