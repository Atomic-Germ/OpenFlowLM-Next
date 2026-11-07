/// \file buffer_range.hpp
/// \brief The one range check the bring-up accessors use, written so it cannot wrap.
///
/// The old form was `if (off + n > size) throw`. Both operands are `size_t`, so `off + n`
/// wraps: with `off = SIZE_MAX` and `n = 2` it evaluates to 1, the comparison passes, and the
/// accessor then synchronises and memcpy's from a byte that does not exist. A negative CLI
/// offset reaches that state trivially -- `static_cast<size_t>(-1)` is `SIZE_MAX` -- and the
/// parsers only rejected a non-positive LENGTH, never a negative OFFSET.
///
/// Writing it as a subtraction makes the wrap impossible: `size - off` is only reached once
/// `off <= size` has been established, so the subtraction cannot underflow.
///
/// Header-only and dependency-free on purpose: the contract is testable without XRT, a
/// device or weights, which is the only way a test of it can run in CI.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace open_qwen36 {

/// Whether a diagnostic range is legal in a buffer of `size` bytes.
///
/// CONTRACT: a zero-length range is legal, but only where it starts inside the buffer
/// (`off <= size`). This matches the C++ container convention -- `[size, size)` is a valid
/// empty range, `[size + 1, size + 1)` is not -- and it is what the callers rely on: a
/// zero-length read has nothing to sync and nothing to copy, but an offset past the end is
/// still a caller bug worth reporting rather than silently accepting.
inline bool range_ok(size_t off, size_t n, size_t size) {
    // `off > size` first, so the subtraction below cannot underflow.
    return off <= size && n <= size - off;
}

/// The check as the accessors want it: throw with a message that cannot overflow either.
///
/// `what` names the accessor and `size_name` the buffer, because the message is read by a
/// person debugging a bring-up run and "outside the buffer" alone does not say which one.
/// The end of the range is printed only when it is representable: `off + n` is exactly the
/// quantity that overflowed, so the message must not compute it back.
inline void require_range(size_t off, size_t n, size_t size,
                          const std::string& what, const std::string& size_name) {
    if (range_ok(off, n, size)) return;
    // Print `[start, end)` whenever the end is ARITHMETICALLY representable -- that is a
    // narrower condition than "in range", and it has to be, or an ordinary one-byte overflow
    // would print "+ 2" instead of the span it actually asked for. Only when `off + n` would
    // wrap is the sum withheld: there it names a byte that IS inside the buffer, which is
    // the most misleading possible description.
    const bool end_known = off <= static_cast<size_t>(-1) - n;
    const std::string span =
        "[" + std::to_string(off) + ", " +
        (end_known ? std::to_string(off + n)
                   : std::to_string(off) + " + " + std::to_string(n)) + ")";
    throw std::runtime_error("open_qwen36: " + what + ": " + span + " is outside the " +
                             size_name + " " + std::to_string(size) + "-byte buffer");
}

}  // namespace open_qwen36
