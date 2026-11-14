// Traces: the think-preamble capacity fix. Host-only: no XRT, no device, no weights.
//
// `Qwen3_5VL::generate` steps a think preamble before its first emitted token, so the
// largest prompt that can still generate is MAX_L - (preamble + 1), not MAX_L - 1. The bug
// was found by guessing that headroom four times (2, 3, 4, 5) and watching the boundary move
// each time, because the preamble was also not being counted in `total_tokens`.
//
// Those were NPU runs. This is the arithmetic behind them, and it is testable anywhere --
// which is the point of extracting it: a semantic regression here must fail CI, not wait for
// somebody to boot the device at context 512.
#include "AutoModel/generation_headroom.hpp"

#include <cstddef>
#include <iostream>
#include <limits>

using oflm::generation::largest_generating_prompt;
using oflm::generation::may_generate;
using oflm::generation::minimum_headroom;
using oflm::generation::preamble_rows;

static int failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

// ---------------------------------------------------------------- the preamble itself

static void test_each_branch_steps_its_own_number_of_rows() {
    // The ids are in the header's comment; this pins the counts the guard depends on.
    CHECK(preamble_rows(true) == 2);
    CHECK(preamble_rows(false) == 4);
    CHECK(minimum_headroom(true) == 3);
    CHECK(minimum_headroom(false) == 5);
}

// ---------------------------------------------------------------- the measured boundary

static void test_the_boundary_matches_the_hardware_observation_at_context_512() {
    // Not thinking: prompt 507 generates and ends at exactly MAX_L, 508 is refused.
    CHECK(largest_generating_prompt(512, false) == 507);
    CHECK(may_generate(507, 512, false));
    CHECK(!may_generate(508, 512, false));
    // Thinking: prompt 509 is the last that generates, 510 is refused.
    CHECK(largest_generating_prompt(512, true) == 509);
    CHECK(may_generate(509, 512, true));
    CHECK(!may_generate(510, 512, true));
}

static void test_the_refusal_is_one_past_the_last_legal_prompt() {
    // No gap and no overlap between "generates" and "refused".
    for (std::size_t max_l : {size_t(512), size_t(1024), size_t(4096), size_t(32768)}) {
        for (bool thinking : {false, true}) {
            const std::size_t last = largest_generating_prompt(max_l, thinking);
            CHECK(may_generate(last, max_l, thinking));
            CHECK(!may_generate(last + 1, max_l, thinking));
            CHECK(!may_generate(last + 2, max_l, thinking));
        }
    }
}

static void test_the_documented_range_from_the_bug_report_is_covered() {
    // The report's own claim: non-thinking is `ctx_len - 5`, thinking is `ctx_len - 3`.
    for (std::size_t max_l : {size_t(512), size_t(2048), size_t(8192)}) {
        CHECK(largest_generating_prompt(max_l, false) == max_l - 5);
        CHECK(largest_generating_prompt(max_l, true) == max_l - 3);
    }
}

// ---------------------------------------------------------------- the hostile inputs

static void test_a_context_too_small_for_the_preamble_refuses_rather_than_wrapping() {
    // The documented contract, stated exactly: `largest_generating_prompt` returns 0 when
    // the context cannot hold the preamble plus one row, so a caller comparing a token count
    // is never handed a wrapped negative. At MAX_L == need there is nothing to give.
    CHECK(largest_generating_prompt(0, false) == 0);
    CHECK(largest_generating_prompt(4, false) == 0);
    // A context of exactly `need` fits the preamble plus the emit row, so an EMPTY prompt is
    // the one case that generates. This is why `count == 0` is not the refusal test.
    CHECK(largest_generating_prompt(5, false) == 0);
    CHECK(may_generate(0, 5, false));
    CHECK(!may_generate(1, 5, false));
    // Thinking needs 3, so MAX_L 3 gives room for prompt 0 and nothing more.
    CHECK(largest_generating_prompt(3, true) == 0);
    CHECK(may_generate(0, 3, true));
    CHECK(!may_generate(1, 3, true));
    // From need + 1 upward the boundary grows one for one.
    for (std::size_t max_l : {size_t(6), size_t(7), size_t(64)}) {
        CHECK(largest_generating_prompt(max_l, false) == max_l - 5);
    }
}

static void test_the_predicate_cannot_overflow_near_size_max() {
    // The old inline form was `total + need > MAX_L`; with a huge total that sum wraps. The
    // helper subtracts instead, so every huge prompt is simply refused.
    const std::size_t huge = std::numeric_limits<std::size_t>::max();
    for (bool thinking : {false, true}) {
        CHECK(!may_generate(huge, 512, thinking));
        CHECK(!may_generate(huge - 1, 512, thinking));
        CHECK(largest_generating_prompt(huge, thinking) == huge - static_cast<std::size_t>(
                                                                 minimum_headroom(thinking)));
    }
}

// ---------------------------------------------------------------- the full table

static void test_every_prompt_count_at_a_small_context_is_decided_correctly() {
    const std::size_t max_l = 64;
    for (std::size_t prompt = 0; prompt <= 80; ++prompt) {
        for (bool thinking : {false, true}) {
            const std::size_t need = static_cast<std::size_t>(minimum_headroom(thinking));
            const bool expected = prompt + need <= max_l;
            CHECK(may_generate(prompt, max_l, thinking) == expected);
        }
    }
}

int main() {
    test_each_branch_steps_its_own_number_of_rows();
    test_the_boundary_matches_the_hardware_observation_at_context_512();
    test_the_refusal_is_one_past_the_last_legal_prompt();
    test_the_documented_range_from_the_bug_report_is_covered();
    test_a_context_too_small_for_the_preamble_refuses_rather_than_wrapping();
    test_the_predicate_cannot_overflow_near_size_max();
    test_every_prompt_count_at_a_small_context_is_decided_correctly();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "generation headroom: all checks passed\n";
    return 0;
}
