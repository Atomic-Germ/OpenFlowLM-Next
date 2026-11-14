/// \file prefill_thresholds.hpp
/// \brief The prompt length at which each prefill schedule becomes eligible.
///
/// There are TWO numbers in this codebase and they answer different questions. They were
/// once presented as one, which is how the public header came to say 64 while the dense
/// runtime used 96:
///
///   * `kDenseBlockLoop` (96) -- the crossover for THIS schedule, the block loop, on a
///     dense kernel set (no routed experts). Below it a prompt is cheaper stepped one token
///     at a time, because a block costs its full 256-wide GEMMs however few real tokens it
///     holds. Measured on Qwen3.8-27B: the fitted crossing is 78, the first clear win 92
///     (0.85) and the applied constant 96, rounded onto the sweep's 8-token grid for margin
///     against the single unpinned pass behind each point.
///
///   * `kLayerMajor` (64) -- the crossover for the LAYER-MAJOR schedule, which exists to
///     amortise a routed expert's weight stream across a whole prompt. It belongs to the
///     MoE families (Qwen3.6-35B was measured there) and is NOT the dense default. A dense
///     set can run layer-major -- `Core::layer_major_ok()` says so -- but it has nothing to
///     amortise, so `Core::layer_major_preferred()` turns it down unless an operator forces
///     it with `OFLM_OPEN_LAYER_MAJOR=1`.
///
/// A reader who needs to predict the DENSE route wants `kDenseBlockLoop`. A prompt of 64
/// to 95 engine tokens is below it and is stepped, which is the range the old header
/// described wrongly.
///
/// Header-only and dependency-free so the numbers are testable without a device.
#pragma once

#include <cstddef>

namespace open_qwen36::prefill {

/// The dense sequential-to-block-loop crossover, in tokens as `Engine::prefill` receives
/// them (not text tokens: the chat template's offset varies and has been seen at 8, 10, 11).
///
/// Overridable at runtime by `OFLM_OPEN_GEMM_BLOCK_MIN`, which is a diagnostic knob and not
/// a setting; the override changes the boundary, never this constant.
inline constexpr size_t kDenseBlockLoop = 96;

/// The layer-major crossover. Separate schedule, separate measurement, separate audience:
/// the MoE families. Never the dense default -- see the file comment.
inline constexpr size_t kLayerMajor = 64;

/// Whether a prompt of `tokens` is eligible for the block route, absent an override and
/// before the safety conditions (capability, M-RoPE) are considered.
///
/// `>=` and not `>`: a prompt of exactly the crossover is eligible. That is the boundary the
/// tests pin at 95 and 96.
inline bool block_route_eligible(size_t tokens, size_t crossover = kDenseBlockLoop) {
    return tokens >= crossover;
}

}  // namespace open_qwen36::prefill
