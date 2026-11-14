// Traces: OPEN-PREFILL-BATCH route telemetry. Host-only: no XRT, no device, no weights.
//
// The bug: `why=` was recomputed from static state with `fallback` as the final `else`. A
// block route that threw, was rewound and then stepped produced exactly the same static
// state as one the policy never chose -- so a recovered crash could be reported as
// `not-preferred`, i.e. as a decision. The fix carries the attempt explicitly and gives it
// priority; this test pins the whole matrix, including the priority itself.
#include "open_qwen36/prefill_route_reason.hpp"

#include <cstring>
#include <iostream>
#include <string>

using open_qwen36::prefill_route::Attempt;
using open_qwen36::prefill_route::Policy;
using open_qwen36::prefill_route::reason;

static int failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

static bool is(const char* got, const char* want) { return std::strcmp(got, want) == 0; }

// ---------------------------------------------------------------- the policy reasons

static void test_a_policy_decision_reads_as_a_policy_decision() {
    // No block was attempted, so each static rule names itself.
    CHECK(is(reason(Attempt::NotAttempted, Policy::Disabled), "disabled"));
    CHECK(is(reason(Attempt::NotAttempted, Policy::BelowCrossover), "below-crossover"));
    CHECK(is(reason(Attempt::NotAttempted, Policy::Mrope), "mrope"));
    CHECK(is(reason(Attempt::NotAttempted, Policy::NoBlockProgram), "no-block-program"));
    // The one the bug report is about: a set that CAN run the route but whose schedule
    // policy prefers the block loop. Nothing failed, so this must survive unchanged.
    CHECK(is(reason(Attempt::NotAttempted, Policy::NotPreferred), "not-preferred"));
}

static void test_every_policy_without_an_attempt_is_distinct() {
    const char* seen[] = {
        reason(Attempt::NotAttempted, Policy::Disabled),
        reason(Attempt::NotAttempted, Policy::BelowCrossover),
        reason(Attempt::NotAttempted, Policy::Mrope),
        reason(Attempt::NotAttempted, Policy::NoBlockProgram),
        reason(Attempt::NotAttempted, Policy::NotPreferred),
    };
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = i + 1; j < 5; ++j)
            CHECK(std::strcmp(seen[i], seen[j]) != 0);
    // And none of them may be the failure word.
    for (const char* s : seen) CHECK(!is(s, "fallback"));
}

// ---------------------------------------------------------------- the dynamic reason

static void test_a_recovered_failure_is_fallback_whatever_the_policy_says() {
    // The priority rule. A recovered failure outranks every static cause, including the one
    // the old code would have printed: `not-preferred`.
    CHECK(is(reason(Attempt::FailedRecovered, Policy::NotPreferred), "fallback"));
    CHECK(is(reason(Attempt::FailedRecovered, Policy::Enabled), "fallback"));
    CHECK(is(reason(Attempt::FailedRecovered, Policy::NoBlockProgram), "fallback"));
    CHECK(is(reason(Attempt::FailedRecovered, Policy::Mrope), "fallback"));
    CHECK(is(reason(Attempt::FailedRecovered, Policy::BelowCrossover), "fallback"));
    CHECK(is(reason(Attempt::FailedRecovered, Policy::Disabled), "fallback"));
}

static void test_the_failure_reason_is_not_a_policy_name() {
    // Read by an operator scanning a log; `fallback` must not collide with a decision.
    for (Policy p : {Policy::Disabled, Policy::BelowCrossover, Policy::Mrope,
                     Policy::NoBlockProgram, Policy::NotPreferred, Policy::Enabled}) {
        CHECK(!is(reason(Attempt::FailedRecovered, p), "not-preferred"));
        CHECK(!is(reason(Attempt::FailedRecovered, p), "disabled"));
    }
}

// ---------------------------------------------------------------- the impossible state

static void test_running_sequentially_with_nothing_to_say_is_not_called_fallback() {
    // Enabled + no attempt means the block route should have run. The OLD code reported this
    // as `fallback` by falling off the end of its conditional chain, which is how a genuine
    // bug could masquerade as a handled failure. It is neither a policy nor a recovery, so
    // it says so and a test can notice.
    CHECK(is(reason(Attempt::NotAttempted, Policy::Enabled), "unexpected-sequential"));
    CHECK(!is(reason(Attempt::NotAttempted, Policy::Enabled), "fallback"));
}

// ---------------------------------------------------------------- the matrix

static void test_every_pair_is_decided_and_only_one_path_yields_fallback() {
    const Policy all[] = {Policy::Enabled, Policy::Disabled, Policy::BelowCrossover,
                          Policy::Mrope, Policy::NoBlockProgram, Policy::NotPreferred};
    int fallbacks = 0;
    for (Attempt a : {Attempt::NotAttempted, Attempt::FailedRecovered}) {
        for (Policy p : all) {
            const char* r = reason(a, p);
            CHECK(r != nullptr && *r != '\0');
            if (is(r, "fallback")) {
                ++fallbacks;
                CHECK(a == Attempt::FailedRecovered);
            }
        }
    }
    // Exactly the six FailedRecovered cells, no more: `fallback` may never appear for a
    // request that did not attempt a block route.
    CHECK(fallbacks == 6);
}

int main() {
    test_a_policy_decision_reads_as_a_policy_decision();
    test_every_policy_without_an_attempt_is_distinct();
    test_a_recovered_failure_is_fallback_whatever_the_policy_says();
    test_the_failure_reason_is_not_a_policy_name();
    test_running_sequentially_with_nothing_to_say_is_not_called_fallback();
    test_every_pair_is_decided_and_only_one_path_yields_fallback();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "prefill route reason: all checks passed\n";
    return 0;
}
