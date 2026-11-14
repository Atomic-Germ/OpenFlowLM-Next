/// \file prefill_route_reason.hpp
/// \brief Why the sequential prefill path ran: a policy decision, or a failure?
///
/// The block route's diagnostics used to recompute the reason from static state at the end
/// of the sequential path, with `fallback` as the final `else`. That could not distinguish
/// the two facts the code elsewhere insists are different:
///
///   * `not-preferred` -- the set CAN run the block route, but the schedule policy says not
///     to use it. Nothing was attempted, and nothing failed.
///   * `fallback`     -- a block route WAS attempted, threw, its state was successfully
///     restored, and the prompt was then stepped. The answer is right either way; the
///     difference is that something went wrong and the log must say so.
///
/// A failure reported as `not-preferred` reads as "we chose this", which is exactly the
/// sentence an operator must never be given about a crash that was recovered.
///
/// So the dynamic fact is carried explicitly and given priority, and the choice is a pure
/// function so the whole matrix can be tested without a device.
#pragma once

namespace open_qwen36::prefill_route {

/// How the block route ended, as the request actually experienced it.
enum class Attempt {
    NotAttempted,     ///< the policy or a safety rule kept the prompt on the sequential path
    FailedRecovered,  ///< a block route threw; state was restored and the prompt was stepped
};

/// The static half: which rule kept the block route from running. Only consulted when no
/// attempt was made -- a failure outranks every policy reason, because it is news.
enum class Policy {
    Enabled,          ///< the block route is on and every static condition passed
    Disabled,         ///< OFLM_OPEN_GEMM_BLOCK=0
    BelowCrossover,   ///< the prompt is shorter than the measured crossover
    Mrope,            ///< the prompt carries image positions, which the block route cannot place
    NoBlockProgram,   ///< this kernel set has no gemm_block program at all
    NotPreferred,     ///< capable, but the schedule policy wants the block loop / sequenced
};

/// The value printed as `why=` on the `prefill-route sequential` line.
inline const char* reason(Attempt attempt, Policy policy) {
    // The dynamic fact first, and unconditionally: a recovered failure is not a policy.
    if (attempt == Attempt::FailedRecovered) return "fallback";
    switch (policy) {
        case Policy::Disabled:       return "disabled";
        case Policy::BelowCrossover: return "below-crossover";
        case Policy::Mrope:          return "mrope";
        case Policy::NoBlockProgram: return "no-block-program";
        case Policy::NotPreferred:   return "not-preferred";
        case Policy::Enabled:        break;
    }
    // Reached only if the sequential path ran with every static condition satisfied and no
    // attempt recorded -- the case the old code called `fallback` by default. That is a
    // caller bug rather than a state: say so instead of inventing a cause.
    return "unexpected-sequential";
}

}  // namespace open_qwen36::prefill_route
