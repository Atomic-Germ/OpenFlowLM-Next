// Traces: OPEN-PREFILL-BATCH (route selection). Host-only: no XRT, no weights, no device.
//
// The bug this exists for: OFLM_OPEN_LAYER_MAJOR=1 only set a boolean, and
// Core::layer_major_preferred() then re-decided with the automatic dense rule. On a set
// whose manifest says `block_loop_preferred` the operator's request lost, while the log
// claimed layer-major was forced. A test that greps the source for the environment
// variable cannot see that -- both the assignment and the string were present and the
// route was still the block loop. So this exercises the decision function itself.
#include "open_qwen36/layer_major_policy.hpp"

#include <iostream>
#include <string>

using open_qwen36::layer_major::Facts;
using open_qwen36::layer_major::Force;
using open_qwen36::layer_major::parse;
using open_qwen36::layer_major::preferred;

static int failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

/// A dense set: capable, no routed experts.
static Facts dense(bool block_loop_preferred) {
    Facts f;
    f.capable = true;
    f.has_routed_experts = false;
    f.block_loop_preferred = block_loop_preferred;
    return f;
}

static Facts moe() {
    Facts f;
    f.capable = true;
    f.has_routed_experts = true;
    return f;
}

static Facts incapable() { return Facts{}; }

// ---------------------------------------------------------------- parsing the operator

static void test_unset_is_auto_not_a_default() {
    // `Auto` is a third state: an unset variable must not be indistinguishable from "0",
    // or the automatic policy could never run and every dense set would silently change.
    CHECK(parse(nullptr) == Force::Auto);
    CHECK(parse("0") == Force::Off);
    CHECK(parse("1") == Force::On);
    CHECK(parse("true") == Force::On);
    CHECK(parse("2") == Force::On);
    // An empty value stays ON, which is what the old `!= "0"` read did: a set-but-empty
    // variable must not flip anyone's route as a side effect of this change.
    CHECK(parse("") == Force::On);
}

// ---------------------------------------------------------------- the automatic policy

static void test_unset_with_block_loop_preferred_is_the_block_loop() {
    // The shipped Qwen3.8-27B case: the manifest says the block loop was measured better.
    CHECK(preferred(Force::Auto, dense(true)) == false);
}

static void test_unset_without_block_loop_preferred_keeps_layer_major() {
    // A dense set that never declared the preference keeps what it shipped with.
    CHECK(preferred(Force::Auto, dense(false)) == true);
}

static void test_unset_leaves_the_moe_answer_alone() {
    // Non-goal: do not move the MoE default. Routed experts still get layer-major, and the
    // manifest flag does not override that.
    CHECK(preferred(Force::Auto, moe()) == true);
    Facts m = moe();
    m.block_loop_preferred = true;
    CHECK(preferred(Force::Auto, m) == true);
}

// ---------------------------------------------------------------- the explicit override

static void test_force_on_wins_over_block_loop_preferred() {
    // This is the bug, stated as a test. Before the fix this returned false and the block
    // loop ran while the log said layer-major was forced.
    CHECK(preferred(Force::On, dense(true)) == true);
}

static void test_force_on_also_wins_where_layer_major_was_already_preferred() {
    CHECK(preferred(Force::On, dense(false)) == true);
    CHECK(preferred(Force::On, moe()) == true);
}

static void test_force_off_disables_layer_major_everywhere() {
    // Not "unless it would have been picked anyway": an explicit 0 means 0.
    CHECK(preferred(Force::Off, dense(false)) == false);
    CHECK(preferred(Force::Off, dense(true)) == false);
    CHECK(preferred(Force::Off, moe()) == false);
}

static void test_force_on_cannot_create_capability() {
    // A set with no block route has no layer-major kernels to run, so the request cannot be
    // honoured. It must report false rather than pretend, and the runtime says so in its
    // log -- "cannot be honoured here" -- instead of claiming a forced schedule.
    CHECK(preferred(Force::On, incapable()) == false);
    CHECK(preferred(Force::Auto, incapable()) == false);
    Facts no_route_capable_false;
    no_route_capable_false.capable = false;
    no_route_capable_false.has_routed_experts = true;
    no_route_capable_false.block_loop_preferred = true;
    CHECK(preferred(Force::On, no_route_capable_false) == false);
}

// ---------------------------------------------------------------- the full matrix

static void test_every_force_and_manifest_combination_is_decided() {
    // No case may be left to a default: 3 forces x 2 manifest flags x 2 capabilities.
    for (int capable = 0; capable <= 1; ++capable) {
        for (int prefers_block = 0; prefers_block <= 1; ++prefers_block) {
            Facts f;
            f.capable = capable == 1;
            f.block_loop_preferred = prefers_block == 1;
            // layer-major is only reachable on a capable set, whatever is asked for
            CHECK(preferred(Force::On, f) == (capable == 1));
            CHECK(preferred(Force::Off, f) == false);
            if (capable == 1) {
                CHECK(preferred(Force::Auto, f) == (prefers_block == 0));
            } else {
                CHECK(preferred(Force::Auto, f) == false);
            }
        }
    }
}

int main() {
    test_unset_is_auto_not_a_default();
    test_unset_with_block_loop_preferred_is_the_block_loop();
    test_unset_without_block_loop_preferred_keeps_layer_major();
    test_unset_leaves_the_moe_answer_alone();
    test_force_on_wins_over_block_loop_preferred();
    test_force_on_also_wins_where_layer_major_was_already_preferred();
    test_force_off_disables_layer_major_everywhere();
    test_force_on_cannot_create_capability();
    test_every_force_and_manifest_combination_is_decided();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "layer-major policy: all checks passed\n";
    return 0;
}
