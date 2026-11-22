// Estimate the ceiling of two arithmetic-preserving host-fold changes, using the
// real production fold for the baseline and a bit-identical re-association for
// the candidate. Diagnostic only: nothing here is production code.
//
// Baseline (prefill_precision::Workspace::run):
//   high.run: split x (K*T), copy A (N*T), A += B
//   outer:    copy (A+B) (N*T)
//   low.run:  split x AGAIN (K*T), copy C (N*T), C += D
//   outer:    (A+B) += (C+D)
// Candidate:
//   split x once, copy A, A += B, copy C, C += D, (A+B) += (C+D)
// Every element's association is unchanged: ((A+B) + (C+D)) in both.
//
//   g++ -O3 -DNDEBUG -std=c++20 -fopenmp -Isrc utilities/prefill-block/host_fold_ceiling.cpp -o /tmp/ceiling
//   /tmp/ceiling 7
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "open_qwen36/prefill_precision.hpp"

namespace {

struct Shape { const char* name; size_t n, k; };
const Shape kLinear[] = {{"qkvz", 16384, 5120}, {"out", 5120, 6144},
                         {"ffn_up", 34816, 5120}, {"ffn_down", 5120, 17408}};
const Shape kFull[] = {{"attn_in", 14336, 5120}, {"out", 5120, 5120},
                       {"ffn_up", 34816, 5120}, {"ffn_down", 5120, 17408}};
constexpr size_t T = 256;

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Baselines are timed through the real header; the candidate mirrors its
// statements exactly, minus the second split and the intermediate copy.
struct Buffers {
    std::vector<float> x, lo, a, b, c, d, out;
};

void prepare(Buffers& b, const Shape& s) {
    b.x.assign(s.k * T, 1.0f);
    b.out.assign(s.n * T, 0.0f);
    b.a.assign(s.n * T, 1.0f);
    b.b.assign(s.n * T, 2.0f);
    b.c.assign(s.n * T, 3.0f);
    b.d.assign(s.n * T, 4.0f);
}

double baseline(const Shape& s, Buffers& b) {
    using namespace open_qwen36::prefill_precision;
    Workspace ws;
    auto dispatch = [&](bool low, const float*) { return low ? b.b.data() : b.a.data(); };
    const auto t = Clock::now();
    const float* r = ws.run(b.x.data(), s.k * T, s.n * T, dispatch);
    const double e = ms(t);
    if (!r || !std::isfinite(r[0])) std::abort();
    return e;
}

double candidate(const Shape& s, Buffers& b) {
    const size_t inputs = s.k * T, outputs = s.n * T;
    const auto t = Clock::now();
    b.lo.resize(inputs);
    for (size_t i = 0; i < inputs; ++i) {
        const float hi = open_qwen36::bf16_to_f32(open_qwen36::f32_to_bf16(b.x[i]));
        b.lo[i] = b.x[i] - hi;
    }
    // W_hi X_hi, then W_hi X_lo, exactly as InputWorkspace::run does.
    b.out.assign(b.a.begin(), b.a.end());
    for (size_t i = 0; i < outputs; ++i) b.out[i] += b.b[i];
    // W_lo X_hi + W_lo X_lo kept as its own sum so the association is unchanged.
    // c is the W_lo X_hi stub output, d the W_lo X_lo one; copy then add, as the
    // baseline's InputWorkspace::run does.
    std::vector<float> lo_sum(outputs);
    std::copy(b.c.begin(), b.c.end(), lo_sum.begin());
    for (size_t i = 0; i < outputs; ++i) lo_sum[i] += b.d[i];
    for (size_t i = 0; i < outputs; ++i) b.out[i] = (b.out[i] + lo_sum[i]);
    const double e = ms(t);
    if (!std::isfinite(b.out[0])) std::abort();
    return e;
}

template <class Fn>
double block(const Shape* set, size_t layers, std::vector<Buffers>& bufs, Fn fn) {
    double total = 0;
    for (size_t i = 0; i < 4; ++i) total += fn(set[i], bufs[i]) * static_cast<double>(layers);
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 7;
    if (reps < 5) { std::fprintf(stderr, "use at least five repetitions\n"); return 2; }
    std::vector<Buffers> lin(4), full(4);
    for (size_t i = 0; i < 4; ++i) { prepare(lin[i], kLinear[i]); prepare(full[i], kFull[i]); }
    double best_base = 1e30, best_cand = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double base = block(kLinear, 48, lin, baseline) + block(kFull, 16, full, baseline);
        const double cand = block(kLinear, 48, lin, candidate) + block(kFull, 16, full, candidate);
        best_base = std::min(best_base, base);
        best_cand = std::min(best_cand, cand);
    }
    std::printf("host fold, 256-token block, minimum of %d\n", reps);
    std::printf("  baseline  (2 splits, 1 intermediate copy) : %8.1f ms\n", best_base);
    std::printf("  candidate (1 split,  no intermediate copy) : %8.1f ms\n", best_cand);
    std::printf("  ceiling saved                              : %8.1f ms  (%.1f%% of the fold)\n",
                best_base - best_cand, 100.0 * (best_base - best_cand) / best_base);
    return 0;
}
