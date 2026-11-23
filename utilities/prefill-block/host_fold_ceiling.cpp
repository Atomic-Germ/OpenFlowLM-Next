// Paired host-fold benchmark with numerically equivalent, allocation-comparable
// arms. Fails closed if the outputs are not byte-identical.
//
//   g++ -O3 -DNDEBUG -std=c++20 -fopenmp -I. -Isrc \
//       utilities/prefill-block/host_fold_ceiling.cpp -o models/host_fold_ceiling
//   ./host_fold_ceiling 7
//
// Both arms perform the same four products in the same association; the
// candidate removes only a duplicate input split; all output copies remain.
// Workspaces persist across calls in both arms, as Core's do. One warm-up pair is
// discarded, then alternating paired measurements are retained and summarised.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "host_fold_probe.hpp"

namespace {

// Qwen3.8-27B block route at T=256. Linear layers: fused qkv+z, out, up+gate,
// down. Full layers: fused q+k+v+gate, out, up+gate, down.
struct Shape { const char* name; size_t n, k; };
const Shape kLinear[] = {{"qkvz", 16384, 5120}, {"out", 5120, 6144},
                         {"ffn_up", 34816, 5120}, {"ffn_down", 5120, 17408}};
const Shape kFull[] = {{"attn_in", 14336, 5120}, {"out", 5120, 6144},
                       {"ffn_up", 34816, 5120}, {"ffn_down", 5120, 17408}};
constexpr size_t T = 256;
constexpr size_t LINEAR_LAYERS = 48, FULL_LAYERS = 16;

struct ShapeState {
    std::vector<fold_probe::Fixture> linear_fixtures, full_fixtures;
    fold_probe::BaselineState linear_base, full_base;
    fold_probe::CandidateState linear_cand, full_cand;
};

// Build fixtures once. `kind` is normally "random"; the caller may select the
// rounding fixture for the numerical gate.
ShapeState state(const std::string& kind) {
    ShapeState s;
    for (const auto& sh : kLinear) s.linear_fixtures.push_back(fold_probe::make_fixture(sh.n, sh.k, T, kind));
    for (const auto& sh : kFull) s.full_fixtures.push_back(fold_probe::make_fixture(sh.n, sh.k, T, kind));
    return s;
}

// Byte-exact gate over every projection of both layer kinds. Returns false and
// prints the first mismatch, so a benchmark can never publish a timing for
// arms that compute different answers.
bool validate(const ShapeState& s) {
    auto check = [](const std::vector<fold_probe::Fixture>& fixtures, const char* kind) {
        for (size_t i = 0; i < fixtures.size(); ++i) {
            std::string why;
            if (!fold_probe::equivalent(fixtures[i], &why)) {
                std::fprintf(stderr, "fold validation FAILED (%s projection %zu): %s\n", kind, i, why.c_str());
                return false;
            }
        }
        return true;
    };
    return check(s.linear_fixtures, "linear") && check(s.full_fixtures, "full");
}

// One whole 256-token block: every projection of every layer, with each arm's
// workspace persistent across the whole measurement (as Core keeps them).
double block_ms(bool baseline, ShapeState& s) {
    double total = 0;
    auto run = [&](const std::vector<fold_probe::Fixture>& fixtures, size_t layers,
                   fold_probe::BaselineState& base, fold_probe::CandidateState& cand) {
        for (size_t i = 0; i < fixtures.size(); ++i) {
            for (size_t layer = 0; layer < layers; ++layer) {
                if (baseline) {
                    double t = 0;
                    if (!fold_probe::run_baseline(base, fixtures[i], nullptr, &t)) std::abort();
                    total += t;
                } else {
                    // The candidate times its whole call; a per-phase total would
                    // double-count the clock reads inside the fold.
                    const auto t0 = fold_probe::Clock::now();
                    if (!fold_probe::run_candidate(cand, fixtures[i], nullptr)) std::abort();
                    total += fold_probe::ms_since(t0);
                }
            }
        }
    };
    run(s.linear_fixtures, LINEAR_LAYERS, s.linear_base, s.linear_cand);
    run(s.full_fixtures, FULL_LAYERS, s.full_base, s.full_cand);
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--describe") {
        std::printf("synthetic split-only candidate; three output copies retained; no production speedup prediction\n");
        for (const auto& s : kLinear) std::printf("linear %s %zu %zu %zu\n", s.name, s.n, s.k, T);
        for (const auto& s : kFull) std::printf("full %s %zu %zu %zu\n", s.name, s.n, s.k, T);
        return 0;
    }
    const int reps = argc > 1 ? std::atoi(argv[1]) : 7;
    if (reps < 5) { std::fprintf(stderr, "use at least five measured pairs (after warm-up)\n"); return 2; }
    const std::string kind = argc > 2 ? argv[2] : "random";
    ShapeState s = state(kind);
    if (!validate(s)) {
        std::fprintf(stderr, "refusing to benchmark numerically different arms\n");
        return 3;
    }
    // One warm-up pair, discarded: persistent workspaces reach steady state.
    (void)block_ms(true, s);
    (void)block_ms(false, s);
    std::vector<double> base, cand;
    for (int r = 0; r < reps; ++r) {
        // Alternate order so drift does not favour one arm.
        if (r % 2 == 0) {
            base.push_back(block_ms(true, s));
            cand.push_back(block_ms(false, s));
        } else {
            cand.push_back(block_ms(false, s));
            base.push_back(block_ms(true, s));
        }
    }
    const fold_probe::Stats st = fold_probe::summarise(base, cand);
    std::printf("host fold, T=%zu, fixture=%s, %d measured pairs (one warm-up pair excluded)\n",
                T, kind.c_str(), reps);
    std::printf("  baseline median  %.1f ms   candidate median %.1f ms\n",
                st.median_baseline, st.median_candidate);
    std::printf("  paired median difference %.1f ms (%.2f%%), wins %d/%zu\n",
                st.median_delta, st.median_change_pct, st.wins, st.delta_ms.size());
    std::printf("  bootstrap 95%% CI of median difference [%.1f, %.1f] ms\n",
                st.ci95_low, st.ci95_high);
    std::printf("samples_baseline");
    for (double v : st.baseline_ms) std::printf(" %.3f", v);
    std::printf("\nsamples_candidate");
    for (double v : st.candidate_ms) std::printf(" %.3f", v);
    std::printf("\nsamples_delta");
    for (double v : st.delta_ms) std::printf(" %.3f", v);
    std::printf("\n");
    return 0;
}
