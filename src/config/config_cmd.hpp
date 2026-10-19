#pragma once
#include "program_args.hpp"

namespace oflm::config {

/// Main entry point for `oflm config <action>`.
/// The action is parsed_args.model_tag (e.g. "show", "set", "unset").
/// Returns 0 on success, non-zero on error.
int config_command_main(const program_args_t& args);

} // namespace oflm::config
