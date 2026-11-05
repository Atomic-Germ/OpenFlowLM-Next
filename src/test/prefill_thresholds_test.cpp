// Traces: OPEN-PREFILL-BATCH route eligibility. Host-only: no XRT, no device, no weights.
//
// The problem this exists for was a documentation bug: `engine.hpp` said the block route
// starts at 64 while `Engine::prefill()` used 96. Both numbers are real, but they answer
// different questions -- 96 is the dense sequential-to-block-loop crossover, 64 belongs to
// the layer-major schedule the MoE families measured. A reader of the public header would
// have predicted the wrong route for a 64-95 token prompt.
//
// Testing this by searching for "96" in a comment would pass on the old header too. These
// assertions are on the constants and on the eligibility function, so the numbers cannot
// drift apart again without a failure here.
#include "open_qwen36/prefill_thresholds.hpp"

#include <iostream>

using open_qwen36::prefill::block_route_eligible;
using open_qwen36::prefill::kDenseBlockLoop;
using open_qwen36::prefill::kLayerMajor;

static int failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

// ---------------------------------------------------------------- the constants

static void test_the_dense_crossover_is_96() {
    // The production dense value. The baseline JSON records the same number; a test in
    // specs/open-engine asserts the two agree, which is why this cannot be changed alone.
    CHECK(kDenseBlockLoop == 96);
}

static void test_the_layer_major_crossover_is_a_different_number() {
    // The distinction the documentation lost. If these ever become equal the file comment
    // and the header both need rewriting, so the equality is worth failing on.
    CHECK(kLayerMajor == 64);
    CHECK(kLayerMajor != kDenseBlockLoop);
    CHECK(kLayerMajor < kDenseBlockLoop);
}

// ---------------------------------------------------------------- the boundary

static void test_95_is_stepped_and_96_is_eligible() {
    // The exact range the old header described wrongly: 64..95 is BELOW the dense
    // crossover and is stepped one token at a time.
    CHECK(!block_route_eligible(0));
    CHECK(!block_route_eligible(63));
    CHECK(!block_route_eligible(64));   // the old documented default: NOT the dense one
    CHECK(!block_route_eligible(95));   // one below
    CHECK(block_route_eligible(96));    // exactly the crossover
    CHECK(block_route_eligible(97));
    CHECK(block_route_eligible(4096));
}

static void test_eligibility_is_a_threshold_not_an_interval() {
    // Monotone: once eligible, larger prompts stay eligible. A gap here would mean the
    // policy was not a threshold at all.
    bool seen = false;
    for (size_t n = 0; n < 300; ++n) {
        const bool ok = block_route_eligible(n);
        if (ok) seen = true;
        else CHECK(!seen && n < kDenseBlockLoop);
        if (n == kDenseBlockLoop) CHECK(ok);
    }
}

static void test_an_override_moves_the_boundary_without_moving_the_constant() {
    // OFLM_OPEN_GEMM_BLOCK_MIN is a diagnostic A/B knob: it changes the effective boundary
    // for that run and never the documented default.
    CHECK(!block_route_eligible(120, 256));
    CHECK(block_route_eligible(256, 256));
    CHECK(block_route_eligible(96, 64));   // a layer-major-shaped boundary, if a caller asks
    CHECK(kDenseBlockLoop == 96);          // the constant itself is untouched
}

int main() {
    test_the_dense_crossover_is_96();
    test_the_layer_major_crossover_is_a_different_number();
    test_95_is_stepped_and_96_is_eligible();
    test_eligibility_is_a_threshold_not_an_interval();
    test_an_override_moves_the_boundary_without_moving_the_constant();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "prefill thresholds: all checks passed\n";
    return 0;
}
