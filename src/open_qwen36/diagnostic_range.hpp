/// \file diagnostic_range.hpp
/// \brief The signed-range check the bring-up CLI options share.
///
/// `--dump-act`, `--dump-state`, `--dump-global` and `--state-canary` all take an offset and
/// a length as text. `sscanf`'s `%lld` is the portable way to read them, so both arrive as
/// `long long` -- and both were validated as if they were already `size_t`: only the length
/// was checked (`n <= 0`), so a negative offset passed and `static_cast<size_t>(-1)` became
/// `SIZE_MAX`. The Core's `off + n > size` test then wrapped and accepted the range.
///
/// This is the boundary where "negative is not a range" is decided. Core checks the same
/// thing again on the cast value, deliberately: the assumption that a caller validated its
/// input is what produced the bug.
///
/// Header-only and dependency-free so the predicate is testable without a process launch.
#pragma once

#include <cstddef>
#include <string_view>

namespace open_qwen36::diagnostic {

/// Whether a parsed (offset, length) pair is usable at all.
///
/// Negative is refused on BOTH, not just on the length. A zero length is refused here as
/// well: `--dump-act` writes a file, and a zero-byte dump is a typo rather than a request.
/// Core's separate contract does allow a zero-length read at an in-range offset, because a
/// library should not have opinions about what a caller wants to do with an empty slice.
inline bool range_ok(long long off, long long n) { return off >= 0 && n >= 0; }

/// The byte count, after `range_ok` has established that it is not negative.
///
/// `long long` is at most as wide as `size_t` on every target this builds for (LP64 and
/// LLP64 both), and a non-negative `long long` therefore converts without loss.
inline size_t to_size(long long v) { return static_cast<size_t>(v); }

}  // namespace open_qwen36::diagnostic
