// Traces: bring-up diagnostics range validation. Host-only: no XRT, no device, no weights.
//
// The CLI parses `--dump-act`, `--dump-state`, `--dump-global` and `--state-canary` offsets
// as `long long` and used to check only the LENGTH (`n <= 0`). A negative offset then cast to
// `size_t` became SIZE_MAX, which the Core's `off + n > size` test wrapped and accepted.
//
// The fix rejects the negative before the cast, in one place the four parsers share. This
// test exercises that shared predicate directly rather than through a process launch: the
// four call sites are a `diagnostic_range_ok(off, n) || n == 0` guard each, so the predicate
// is the whole contract. It lives here rather than in cli.cpp's anonymous namespace for
// that reason -- see the `#if` in cli.cpp.
#include "open_qwen36/diagnostic_range.hpp"

#include <cstdint>
#include <iostream>
#include <limits>

using open_qwen36::diagnostic::range_ok;
using open_qwen36::diagnostic::to_size;

static int failures = 0;
static constexpr size_t kMax = std::numeric_limits<size_t>::max();

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

// ---------------------------------------------------------------- what must be rejected

static void test_a_negative_offset_is_refused_before_any_cast() {
    // The four options, one shape each: a negative offset with a legal length.
    for (long long off : {-1LL, -2LL, -1024LL}) {
        CHECK(!range_ok(off, 16));
    }
    // -1 is the interesting one: it is the value that becomes SIZE_MAX.
    CHECK(static_cast<size_t>(-1LL) == kMax);
}

static void test_a_negative_length_is_refused() {
    CHECK(!range_ok(0, -1LL));
    CHECK(!range_ok(0, -16LL));
    // Both negative is not a cancellation.
    CHECK(!range_ok(-16LL, -16LL));
    // Zero length is NOT a negative one: `range_ok` is about signedness, and the CLI's own
    // rule that a diagnostics dump must ask for something lives one level up
    // (`diagnostic_range_ok(off, n) == range_ok(off, n) && n != 0`). Keeping the two apart is
    // what lets Core accept an empty slice while the CLI refuses an empty dump.
    CHECK(range_ok(0, 0));
}

static void test_the_parser_rejects_every_negative_combination() {
    // What the old guard accepted: off < 0 with n > 0. Every one of these used to reach the
    // Core as a huge size_t.
    for (long long off : {-1LL, -8LL, -1000000LL}) {
        for (long long n : {1LL, 16LL, 4096LL, 1LL << 40}) {
            CHECK(!range_ok(off, n));
        }
    }
}

// ---------------------------------------------------------------- what must be accepted

static void test_valid_ranges_still_parse_normally() {
    CHECK(range_ok(0, 1));
    CHECK(range_ok(0, 4096));
    CHECK(range_ok(4096, 16));
    CHECK(range_ok(1LL << 40, 1));
    // The largest offset a `long long` can carry is legal at this layer; whether it fits the
    // buffer is the Core's question, not the parser's.
    CHECK(range_ok(std::numeric_limits<long long>::max(), 1));
}

static void test_the_cast_is_exact_for_every_accepted_value() {
    for (long long off : {0LL, 1LL, 4096LL, 1LL << 40, std::numeric_limits<long long>::max()}) {
        CHECK(to_size(off) == static_cast<size_t>(off));
        CHECK(to_size(off) <= kMax);
    }
    for (long long n : {1LL, 2LL, 4096LL, 1LL << 40}) {
        CHECK(to_size(n) == static_cast<size_t>(n));
    }
}

// ---------------------------------------------------------------- the pair

static void test_every_accepted_pair_survives_the_cast_unchanged() {
    // No accepted value may change when converted, or the guard would be measuring one
    // number and the caller using another.
    for (long long off = 0; off < 64; ++off) {
        for (long long n = 1; n < 64; ++n) {
            CHECK(range_ok(off, n));
            CHECK(to_size(off) == static_cast<size_t>(off));
            CHECK(to_size(n) == static_cast<size_t>(n));
        }
    }
}

static void test_no_negative_ever_reaches_size_max() {
    // The end-to-end property: for every value the parser refuses, the cast it prevents
    // would have produced a number no real buffer can hold.
    for (long long off = -1; off > -4096; --off) {
        CHECK(!range_ok(off, 1));
        const size_t would_be = static_cast<size_t>(off);
        CHECK(would_be > static_cast<size_t>(1) << 40);
    }
}

int main() {
    test_a_negative_offset_is_refused_before_any_cast();
    test_a_negative_length_is_refused();
    test_the_parser_rejects_every_negative_combination();
    test_valid_ranges_still_parse_normally();
    test_the_cast_is_exact_for_every_accepted_value();
    test_every_accepted_pair_survives_the_cast_unchanged();
    test_no_negative_ever_reaches_size_max();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "diagnostic range: all checks passed\n";
    return 0;
}
