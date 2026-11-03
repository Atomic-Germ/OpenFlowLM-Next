// Traces: bring-up diagnostics range validation. Host-only: no XRT, no device, no weights.
//
// The bug: `if (off + n > size)` on two `size_t`s. With `off = SIZE_MAX, n = 2` the sum wraps
// to 1, the check passes, and the accessor syncs and memcpy's from a byte that does not
// exist. A negative CLI offset reaches that state trivially, because
// `static_cast<size_t>(-1)` IS `SIZE_MAX`, and the parsers only rejected a non-positive
// length. These cases are the whole contract, exercised directly -- the Core accessors are
// one `require_range` call each, and a test that needed a device could not run in CI.
#include "open_qwen36/buffer_range.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

using open_qwen36::range_ok;
using open_qwen36::require_range;

static int failures = 0;
static constexpr size_t kMax = std::numeric_limits<size_t>::max();

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

/// `require_range` accepted -- the case a caller would carry on from.
static bool accepts(size_t off, size_t n, size_t size) {
    try {
        require_range(off, n, size, "test", "test");
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

// ---------------------------------------------------------------- the legal cases

static void test_an_in_range_read_is_accepted() {
    CHECK(accepts(0, 16, 64));
    CHECK(accepts(8, 16, 64));
    CHECK(accepts(48, 16, 64));       // ends exactly at the last byte + 1
    CHECK(accepts(0, 64, 64));       // the whole buffer
}

static void test_the_exact_last_byte_is_accepted() {
    CHECK(accepts(63, 1, 64));
    CHECK(accepts(64 - 1, 1, 64));
}

static void test_a_zero_length_range_is_legal_but_only_inside() {
    // The contract, stated where it can be read: [size, size) is a valid empty range, which
    // is the C++ container convention, and it is what the accessors rely on -- nothing to
    // sync, nothing to copy, but an offset past the end is still a caller bug.
    CHECK(accepts(0, 0, 64));
    CHECK(accepts(64, 0, 64));
    CHECK(!accepts(65, 0, 64));
    CHECK(!accepts(kMax, 0, 64));
    // A zero-length range in a zero-length buffer exists and is legal.
    CHECK(accepts(0, 0, 0));
}

// ---------------------------------------------------------------- one byte too far

static void test_one_byte_past_the_end_is_rejected() {
    CHECK(!accepts(63, 2, 64));       // ends at 65, one past
    CHECK(!accepts(64, 1, 64));       // starts at the end
    CHECK(!accepts(0, 65, 64));       // one longer than the buffer
}

// ---------------------------------------------------------------- the wrap cases

static void test_a_huge_offset_is_rejected_rather_than_wrapping() {
    // This is the bug. `SIZE_MAX + 2` is 1, so the old `off + n > size` said "in range".
    CHECK(!accepts(kMax, 2, 64));
    CHECK(!accepts(kMax, 1, 64));
    CHECK(!accepts(kMax, 0, 64));
    // The old form would have accepted all three of these:
    CHECK(kMax + 2 == 1);            // the arithmetic the old check relied on
    CHECK(kMax + 1 == 0);
    // A huge offset with a length that cancels it is still out of range.
    CHECK(!accepts(kMax / 2, kMax / 2, 64));
}

static void test_a_signed_minus_one_cast_to_size_t_still_fails() {
    // What the CLI actually did: a negative `long long` offset cast to `size_t`.
    const long long negative = -1;
    const size_t as_size_t = static_cast<size_t>(negative);
    CHECK(as_size_t == kMax);
    CHECK(!accepts(as_size_t, 1, 64));
    CHECK(!accepts(as_size_t, 64, 64));
    // The CLI must reject it before the cast; the Core must reject it after. Both, because
    // "the caller validated it" is exactly the assumption that produced this bug.
    CHECK(!accepts(static_cast<size_t>(-4), 8, 1024));
}

static void test_a_length_near_size_max_is_rejected_without_arithmetic() {
    CHECK(!accepts(0, kMax, 64));
    CHECK(!accepts(1, kMax, 64));
    CHECK(!accepts(kMax, kMax, kMax));
    // size - off cannot underflow on the path that matters: off > size returns first.
    CHECK(range_ok(kMax, 0, kMax));
    CHECK(!range_ok(kMax, 1, kMax));
}

// ---------------------------------------------------------------- the message

static void test_the_error_names_the_range_without_computing_a_wrapped_end() {
    std::string msg;
    try {
        require_range(kMax, 2, 64, "read_act", "layer's act");
    } catch (const std::runtime_error& e) {
        msg = e.what();
    }
    CHECK(!msg.empty());
    // It must not print a wrapped end such as "[18446744073709551615, 1)" -- "1" is a byte
    // that IS inside the buffer, which is the most misleading possible description.
    CHECK(msg.find("18446744073709551615") != std::string::npos);   // the offset
    CHECK(msg.find(", 1)") == std::string::npos);
    CHECK(msg.find("+ 2") != std::string::npos);                     // the length, unsummed
    CHECK(msg.find("64-byte") != std::string::npos);                 // the buffer size
    // And an ordinary rejection still reads as a span.
    std::string plain;
    try {
        require_range(63, 2, 64, "read_state", "layer's state");
    } catch (const std::runtime_error& e) {
        plain = e.what();
    }
    CHECK(plain.find("[63, 65)") != std::string::npos);
}

// ---------------------------------------------------------------- the property

static void test_no_overflowing_pair_is_ever_accepted() {
    // Sweep the boundary shape at several sizes: every (off, n) whose sum wraps must be
    // refused, and the accepted set must be exactly the non-wrapping in-range one.
    for (size_t size : {size_t(0), size_t(1), size_t(64), size_t(4096)}) {
        for (size_t off : {size_t(0), size, size + 1, kMax - 1, kMax}) {
            for (size_t n : {size_t(0), size_t(1), size_t(2), kMax}) {
                const bool would_wrap = off > kMax - n;
                const bool accepted = accepts(off, n, size);
                if (would_wrap) {
                    CHECK(!accepted);
                } else {
                    CHECK(accepted == (off <= size && n <= size - off));
                }
            }
        }
    }
}

int main() {
    test_an_in_range_read_is_accepted();
    test_the_exact_last_byte_is_accepted();
    test_a_zero_length_range_is_legal_but_only_inside();
    test_one_byte_past_the_end_is_rejected();
    test_a_huge_offset_is_rejected_rather_than_wrapping();
    test_a_signed_minus_one_cast_to_size_t_still_fails();
    test_a_length_near_size_max_is_rejected_without_arithmetic();
    test_the_error_names_the_range_without_computing_a_wrapped_end();
    test_no_overflowing_pair_is_ever_accepted();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "buffer range: all checks passed\n";
    return 0;
}
