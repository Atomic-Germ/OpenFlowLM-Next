/// \file flm_add.hpp
/// \brief Native reimplementation of the `flm add` installer (formerly the
///        standalone `flm-add` Python tool) for the open-kernel system.
#pragma once

#include <string>

struct program_args_t;

namespace flm_add {

/// \brief Install a pre-converted FLM (Q4NX) model and register it with FastFlowLM.
/// \param args Parsed CLI arguments; the repo is taken from args.model_tag.
/// \return process exit code (0 on success).
int run(const program_args_t& args);

} // namespace flm_add
