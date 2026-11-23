// Time the REAL paired-prefill host fold (prefill_precision.hpp) with a stubbed
// dispatch, so CPU split/copy/accumulate cost is separable from every NPU
// dispatch, mapping and sync. Diagnostic only: it changes no production source
// and is not a model or end-to-end measurement.
//
//   g++ -O3 -DNDEBUG -std=c++20 -fopenmp -Isrc utilities/prefill-block/host_fold_bench.cpp -o /tmp/fold
//   /tmp/fold 7
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "open_qwen36/prefill_precision.hpp"

namespace {

struct Shape { const char* name; size_t n, k; };

// Qwen3.8-27B block route at T=256: four projections per layer kind.
// linear: fused qkv+z, out, fused up+gate, down
// full:   fused q+k+v+gate, out, fused up+gate, down
const Shape kLinear[] = {{"qkvz", 16384, 5120}, {"out", 5120, 6144},
                         {"ffn_up", 34816, 5120}, {"ffn_down", 5120, 17408}};
const Shape kFull[] = {{"attn_in", 14336, 5120}, {"out", 5120, 5120},
                       {"ffn_up", 34816, 5120}, {"ffn_down", 5120, 17408}};
constexpr size_t T = 256;

// Stub dispatch: returns a preallocated device-like buffer and does no work.
// Only the fold's split/copy/accumulate is timed; the real dispatch's cost is
// the separate part0 column. The outputs are distinct so neither the copy nor
// the add can be folded away.
struct Scratch {
    std::vector<float> a, b;
};

const float* dispatch(bool low, Scratch& s) {
    return low ? s.b.data() : s.a.data();
}

double one(const Shape& s, std::vector<float>& x) {
    using namespace open_qwen36::prefill_precision;
    const size_t inputs = s.k * T, outputs = s.n * T;
    x.resize(inputs);
    Scratch scratch{std::vector<float>(outputs, 1.0f), std::vector<float>(outputs, 2.0f)};
    Workspace ws;
    auto fn = [&](bool low, const float*) { return dispatch(low, scratch); };
    const auto t = std::chrono::steady_clock::now();
    const float* r = ws.run(x.data(), inputs, outputs, fn);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    if (!r || !std::isfinite(r[0])) std::abort();
    return ms;
}

double block(const Shape* set, size_t layers, std::vector<float>& x, bool verbose) {
    double total = 0;
    for (size_t i = 0; i < 4; ++i) {
        const double ms = one(set[i], x);
        if (verbose) std::printf("  %-8s %6zu x %5zu : %8.3f ms/fold x %zu layers\n",
                                 set[i].name, set[i].n, set[i].k, ms, layers);
        total += ms * static_cast<double>(layers);
    }
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 7;
    if (reps < 5) { std::fprintf(stderr, "use at least five repetitions\n"); return 2; }
    std::printf("T=%zu reps=%d host fold only (split + copy + accumulate)\n", T, reps);
    double best_linear = 1e30, best_full = 1e30;
    std::vector<float> x;
    for (int rep = 0; rep < reps; ++rep) {
        const bool verbose = rep == 0;
        const double lin = block(kLinear, 48, x, verbose);
        const double full = block(kFull, 16, x, verbose);
        best_linear = std::min(best_linear, lin);
        best_full = std::min(best_full, full);
        std::printf("  rep %d: linear %8.1f ms  full %8.1f ms  block %8.1f ms\n",
                    rep, lin, full, lin + full);
    }
    std::printf("model 256-token block host fold (minimum of %d): %.1f ms\n",
                reps, best_linear + best_full);
    return 0;
}
