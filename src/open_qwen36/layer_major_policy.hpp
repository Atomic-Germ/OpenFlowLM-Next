/// \file layer_major_policy.hpp
/// \brief Which prefill schedule to use: capability, operator force, and the preference.
///
/// `OFLM_OPEN_LAYER_MAJOR` used to be read into one boolean and then re-decided by
/// `Core::layer_major_preferred()`, which applied the automatic dense-vs-MoE rule on top.
/// For a dense set whose manifest says `block_loop_preferred` that made
/// `OFLM_OPEN_LAYER_MAJOR=1` a no-op: the operator asked for layer-major, the log claimed
/// layer-major was forced, and the block loop ran.
///
/// So the switch is modelled as what it actually is -- three states, not a boolean:
///
///   unset    automatic policy (this is what every user who does not set it gets)
///   0        force OFF, even where layer-major is the natural choice
///   nonzero  force ON, wherever the kernel set can actually run it
///
/// Force-on cannot create capability. `layer_major_ok()` stays the hard test: a set with no
/// block route, or with layer types that disagree, runs the block loop no matter what the
/// operator asks, and the runtime says so rather than pretending.
///
/// Header-only and dependency-free on purpose: the policy is the thing being tested, and a
/// test of it should not need XRT, weights or a device.
#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

namespace open_qwen36::layer_major {

/// The operator's explicit request, as distinct from "they did not say".
enum class Force {
    Auto,  ///< OFLM_OPEN_LAYER_MAJOR unset: the automatic preference decides
    Off,   ///< "0": layer-major is not used
    On,    ///< any other value: layer-major is used wherever the set can run it
};

/// Parse the raw value. `nullptr` means the variable is unset.
///
/// A variable that is present but EMPTY counts as On: `OFLM_OPEN_LAYER_MAJOR= oflm ...` is a
/// shell mistake, and the older boolean read (`!= "0"`) already treated it as on, so keeping
/// that reading means this change cannot flip anyone's route by accident.
inline Force parse(const char* raw) {
    if (raw == nullptr) return Force::Auto;
    return std::string_view(raw) == "0" ? Force::Off : Force::On;
}

/// Read it from the environment, once, at construction.
inline Force from_environment() { return parse(std::getenv("OFLM_OPEN_LAYER_MAJOR")); }

/// Everything the schedule decision depends on, so the decision itself is a pure function.
struct Facts {
    bool capable = false;  ///< layer_major_ok(): a block route every layer can run
    bool has_routed_experts = false;  ///< some layer carries a moe_kernel / moe_batch program
    bool block_loop_preferred = false;  ///< the manifest says the block loop was measured better
};

/// Whether layer-major is the schedule to USE. Pure: no globals, no environment.
///
/// The order matters and is the whole point of this file:
///
///   1. capability -- a force-on cannot run a schedule the set has no kernels for;
///   2. the operator's explicit answer -- ON wins over the preference, OFF disables it;
///   3. only when nobody said anything: the automatic rule.
inline bool preferred(Force force, const Facts& f) {
    if (!f.capable) return false;
    switch (force) {
        case Force::On:  return true;   // the operator asked; the preference does not veto
        case Force::Off: return false;
        case Force::Auto: break;
    }
    // Routed experts: one expert pass a layer instead of one per 256-token block, which is
    // the win layer-major exists for.
    if (f.has_routed_experts) return true;
    // Dense: layer-major pads the tail to whole blocks (`TOT = B * T`) and amortises
    // nothing, so the block loop is better -- but only where that was measured. A set that
    // does not declare `block_loop_preferred` keeps what it shipped with.
    return !f.block_loop_preferred;
}

}  // namespace open_qwen36::layer_major
