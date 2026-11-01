// What arithmetic a block-prefill projection runs: an explicit, parsed choice.
//
// Paired prefill reaches its accuracy by splitting BOTH operands and summing
// four products per projection --
//
//     (W_hi X_hi + W_hi X_lo) + (W_lo X_hi + W_lo X_lo)
//
// with the weights carried as a q4_1 stream and the input as bf16. The `_low`
// companions are half of the prefill GEMM column (measured 2026-10-31: 38557.0
// of 74878.6 ms, 51.5%, at 513 tokens over all 64 layers), and what they carry
// is worth stating precisely, because it decides whether they can be traded
// away: `gemm_q4_dequant.h` builds `se` by dequantising the int4 weight
// EXACTLY into fp32 accumulators, and the `_low` build differs only in storing
// `se - bf16(se)` instead of `bf16(se)`. So the `_low` product removes the bf16
// ROUNDING of the already-dequantised weight. It is not a correction for the
// weight's quantisation error -- that error is the int4 lattice itself (about
// 3.6% of a weight step, uniform +/-0.5 LSB in the 4-bit table) and no product
// below touches it.
//
// That makes the rounding removable on request rather than sacred. It stays a
// request: `Mode::paired` is the default, byte for byte what Phase 20 accepted,
// and the two weaker modes have to be named to be used.
//
// This is NOT `prefill_mode`. That setting picks WHICH exported route is loaded
// (a whole second set of GEMM kernels, "lean"); this one picks the arithmetic
// of the route already loaded, and needs no second export. Naming them apart is
// deliberate -- see test_prefill_precision_switch.py.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_qwen36/prefill_precision.hpp"

namespace open_qwen36::prefill_precision_switch {

enum class Mode {
    paired,       ///< four products: both operands split. The accepted route.
    three_term,   ///< W_hi X_hi + W_hi X_lo + W_lo X_hi: omits only W_lo X_lo.
    weight_high,  ///< two products: W_hi X_hi + W_hi X_lo, no weight residual at all.
    single,       ///< one product: bf16 input, bf16 dequantised weight.
};

/// The accepted route. Every other value is opt-in.
inline constexpr Mode kDefault = Mode::paired;

inline std::string name(Mode m) {
    switch (m) {
        case Mode::paired: return "paired";
        case Mode::three_term: return "three_term";
        case Mode::weight_high: return "weight_high";
        case Mode::single: return "single";
    }
    throw std::runtime_error("prefill precision: unknown mode");
}

/// Dispatches per projection -- also the cost of the mode, since every dispatch
/// walks the same weights. paired:4 three_term:3 weight_high:2 single:1.
inline int dispatches(Mode m) {
    switch (m) {
        case Mode::paired: return 4;
        case Mode::three_term: return 3;
        case Mode::weight_high: return 2;
        case Mode::single: return 1;
    }
    throw std::runtime_error("prefill precision: unknown mode");
}

/// Does the input get split into bf16 high + residual? Only `single` skips it,
/// and `single` then reads the bf16 cast as its input, so no mode is
/// "unsplit and unrounded" -- that combination would silently be worse than
/// everything named here.
inline bool splits_input(Mode m) {
    switch (m) {
        case Mode::paired:
        case Mode::three_term:
        case Mode::weight_high: return true;
        case Mode::single: return false;
    }
    throw std::runtime_error("prefill precision: unknown mode");
}

/// Does a `_low` (bf16 weight-residual) dispatch get issued at all?
inline bool uses_weight_residual(Mode m) {
    switch (m) {
        case Mode::paired:
        case Mode::three_term: return true;
        case Mode::weight_high:
        case Mode::single: return false;
    }
    throw std::runtime_error("prefill precision: unknown mode");
}

/// Does the W_lo dispatch also run the input's residual (the W_lo X_lo product)?
///
/// This is the one term that can be dropped almost for free. With both operands
/// split, W_lo is the bf16 rounding of the dequantised weight (|W_lo| ~ 2^-9 |W|)
/// and X_lo is the input's rounding residual (|X_lo| ~ 2^-9 |X|), so W_lo X_lo is
/// ~2^-18 of the projection -- 512 times smaller than either term `weight_high`
/// gives up, and far below the bf16 sum the result is rounded to. `weight_high`
/// cannot be accepted (measured: 301 of 513 positions below the logit gate);
/// this one is 3 of 4 products in the same order the accepted route sums them.
inline bool needs_low_input(Mode m) {
    switch (m) {
        case Mode::paired: return true;
        case Mode::three_term:
        case Mode::weight_high:
        case Mode::single: return false;
    }
    throw std::runtime_error("prefill precision: unknown mode");
}

/// Parse the one string this setting carries. An unknown value is an error, not
/// a fallback: a typo'd mode that quietly ran the accepted route would hide a
/// speed-up that never happened, and one that quietly ran a weaker route would
/// change the model's numerics without anyone asking -- both are worse than a
/// stopped run. The message lists every legal value.
inline Mode parse(const std::string& s) {
    if (s == "paired") return Mode::paired;
    if (s == "three_term") return Mode::three_term;
    if (s == "weight_high") return Mode::weight_high;
    if (s == "single") return Mode::single;
    throw std::runtime_error("open_qwen36: prefill precision '" + s +
                             "' is not one of paired, three_term, weight_high, single");
}

// The arithmetic itself lives in prefill_precision.hpp, which already provides
// exactly what each mode needs: `Workspace` (four products), `Workspace::high`
// (two products: the high weight against both input halves) and a single
// `dispatch` for `single`. Core::gemm_run selects among them. There is no
// second implementation here on purpose -- a duplicate fold would be a second
// thing to keep in step with the order the accepted route was validated in, and
// the first attempt at one silently ran four products where it promised two.
}  // namespace open_qwen36::prefill_precision_switch
