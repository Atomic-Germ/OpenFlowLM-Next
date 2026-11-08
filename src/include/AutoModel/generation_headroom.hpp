/// \file generation_headroom.hpp
/// \brief How many free context slots a generation step needs, and when to refuse.
///
/// `Qwen3_5VL::generate` does not begin with a sampled token: it steps a THINK PREAMBLE
/// first, and those rows consume context positions. So the largest prompt that can still
/// emit a token is `MAX_L - (preamble_rows + 1)` and not `MAX_L - 1`. Getting this wrong was
/// a real bug: a constant "leave a few slots" guess was tried at 2, 3, 4 and 5, and each
/// guess only moved the observed boundary, because the preamble was also not being counted
/// in `total_tokens` -- the counter and the engine position had silently diverged.
///
/// Measured at context 512 after the fix:
///
///   not thinking (4 rows, need 5):  507 generates and ends at exactly MAX_L; 508 is refused
///   thinking     (2 rows, need 3):  509 generates and ends at exactly MAX_L; 510 is refused
///
/// This is arithmetic, not hardware: the numbers above justify the rule, but the rule itself
/// is testable on any machine, which is why it lives here instead of only in the model.
#pragma once

#include <cstddef>

namespace oflm::generation {

/// The rows a generation steps before its first emitted token, for this generation shape.
///
/// Not a setting and not a model name -- it is the shape's own semantics. `thinking` selects
/// the 2-row preamble (`think_start_id`, `198`); otherwise the 4-row one (`think_start_id`,
/// `271`, `think_end_id`, `271`).
inline constexpr int preamble_rows(bool thinking) { return thinking ? 2 : 4; }

/// Free slots the first generation step needs: the preamble plus one row to emit.
inline constexpr int minimum_headroom(bool thinking) { return preamble_rows(thinking) + 1; }

/// Whether a prompt of `prompt` tokens may begin generating.
///
/// Equivalent to `prompt + need <= max_l`, written so the addition cannot overflow: `max_l`
/// is compared against `need` first, and `prompt` is only subtracted once that held, so the
/// subtraction cannot wrap either.
///
/// This is NOT `prompt <= largest_generating_prompt(...)`: that form is true whenever the
/// bound clamps to 0, which would let a zero-token prompt "generate" in a context too small
/// for the preamble itself. The two are defined independently for that reason.
inline constexpr bool may_generate(std::size_t prompt, std::size_t max_l, bool thinking) {
    const std::size_t need = static_cast<std::size_t>(minimum_headroom(thinking));
    return max_l >= need && prompt <= max_l - need;
}

/// The largest prompt that can still generate at least one token.
///
/// Callers pass the TOTAL stored sequence limit (`MAX_L` is prompt + generated, not a prompt
/// limit) and the branch. Returns 0 when no prompt can generate -- check `may_generate()`
/// rather than `count == 0`, since a context that only just fits the preamble also reports 0
/// while still admitting an empty prompt.
inline constexpr std::size_t largest_generating_prompt(std::size_t max_l, bool thinking) {
    const std::size_t need = static_cast<std::size_t>(minimum_headroom(thinking));
    return max_l >= need ? max_l - need : 0;
}

}  // namespace oflm::generation
