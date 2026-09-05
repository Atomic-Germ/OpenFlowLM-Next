//===- npue_encoder.cpp --------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- definitions for the encoder's process-wide model geometry.
// SPDX-License-Identifier: MIT
//
// This file is small on purpose. Everything else the encoder needs is defined
// in npue_encoder.hpp, for the inlining reason recorded there; what has to
// live in exactly one translation unit is the mutable state.
//
// Everything here is a scalar with a CONSTANT initialiser, so it is
// initialised before any code runs and there is no static-initialisation
// order left in this file to get wrong. That is why the container-typed
// state (prompts, rope_inv_freq, the two names) is NOT here -- those are
// function-local statics behind accessors in the header, constructed on
// first use. tasks/0156 step 2.
//
// detail::apply_model_shape() in the header is the only writer, it runs
// once under a ShapeLease, and every value starts at 0 so that a missed
// initialisation divides by zero or allocates nothing rather than quietly
// using a stale number from a previously loaded model.

#include "npue_encoder.hpp"

namespace npue {
namespace enc {

int64_t g_seq = 0, g_hidden = 0, g_heads = 0, g_head_dim = 0;
int64_t g_ffn = 0, g_layers = 0, g_max_positions = 0;
bool g_cls_pool = false, g_l2_normalize = true;
bool g_allow_truncation = false;
std::atomic<bool> g_truncation_warned{false};
bool g_wide_lock = false;
bool g_rope = false, g_gated_ffn = false;
double g_rope_theta = 0.0;
GatedAct g_gated_act = GatedAct::Silu;

}  // namespace enc
}  // namespace npue
