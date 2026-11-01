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
#include <algorithm>
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

// ---- per-role attribution (PHASE 23) ----
//
// A mixed policy needs to say "run this arithmetic for THIS projection and the
// accepted one everywhere else". The role is derived from WHERE the step sits in
// the exported program, by pointer identity, never from a kernel name: the name
// carries a shape (gemm_n16384_k5120) that several roles can share, and a name
// match would silently apply to the wrong projection the moment an export
// renames or reshapes one.
//
// Only four roles are separable, and that is a property of the export, not a
// shortcut. The packer emits FUSED projections:
//   * DeltaNet qkv and z are one GEMM (10240 + 6144 = n16384_k5120);
//   * attention q, k, v and the gate are one GEMM (2*6144 + 2*1024 = n14336_k5120);
//   * FFN up and gate are one GEMM (2*17408 = n34816_k5120).
// Asking for "k only" or "gate only" would mean repacking the model, so the
// sweep tests what the runtime can actually select. DeltaNet alpha/beta is not a
// GEMM at all -- it runs in the glue kernel -- so no mode of this switch can
// reach it, and it is reported as not-testable rather than pretended to be.
enum class Role {
    none,       ///< not a separable projection role (tail, glue, fused-and-unused)
    qkvz,       ///< DeltaNet layer, program[0]: fused qkv + z
    attn_in,    ///< full layer, program[0]: fused q + k + v + gate
    out,        ///< both layer kinds, program[1]: the output projection
    ffn_up,     ///< ffn_program[0]: fused up + gate
    ffn_down,   ///< ffn_program[1]
};

inline std::string role_name(Role r) {
    switch (r) {
        case Role::none: return "none";
        case Role::qkvz: return "qkvz";
        case Role::attn_in: return "attn_in";
        case Role::out: return "out";
        case Role::ffn_up: return "ffn_up";
        case Role::ffn_down: return "ffn_down";
    }
    throw std::runtime_error("prefill precision: unknown role");
}

/// Every role a build can select, in the order the sweep reports them.
inline const std::vector<Role>& all_roles() {
    static const std::vector<Role> v = {Role::qkvz, Role::attn_in, Role::out,
                                        Role::ffn_up, Role::ffn_down};
    return v;
}

inline Role parse_role(const std::string& s) {
    for (Role r : all_roles())
        if (role_name(r) == s) return r;
    throw std::runtime_error("open_qwen36: prefill role '" + s +
                             "' is not one of qkvz, attn_in, out, ffn_up, ffn_down");
}

/// Parse a comma or space separated role list; an unknown name is an error, never
/// a silent skip, because a skipped role would make a variant that ran the
/// accepted arithmetic look like a successful measurement of the other one.
inline std::vector<Role> parse_roles(const std::string& s) {
    std::vector<Role> out;
    std::string cur;
    for (size_t i = 0; i <= s.size(); ++i) {
        const char c = i < s.size() ? s[i] : ',';
        if (c == ',' || c == ' ' || c == ';') {
            if (!cur.empty()) {
                const Role r = parse_role(cur);
                if (std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    return out;
}

inline bool selects(const std::vector<Role>& roles, Role r) {
    return r != Role::none && std::find(roles.begin(), roles.end(), r) != roles.end();
}

// The arithmetic itself lives in prefill_precision.hpp, which already provides
// exactly what each mode needs: `Workspace` (four products), `Workspace::high`
// (two products: the high weight against both input halves) and a single
// `dispatch` for `single`. Core::gemm_run selects among them. There is no
// second implementation here on purpose -- a duplicate fold would be a second
// thing to keep in step with the order the accepted route was validated in, and
// the first attempt at one silently ran four products where it promised two.
}  // namespace open_qwen36::prefill_precision_switch
