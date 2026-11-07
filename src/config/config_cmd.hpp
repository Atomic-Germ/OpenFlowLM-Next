#pragma once
#include "program_args.hpp"
#include "settings_resolver.hpp"

namespace oflm::config {

// Forward declarations for UI functions
int config_show(const program_args_t& args);
const char* source_tag(SettingValue::Source s);

/// Main entry point for `oflm config <action>`.
/// The action is parsed_args.model_tag (e.g. "show", "set", "unset").
/// Returns 0 on success, non-zero on error.
int config_command_main(const program_args_t& args);

/// Same, but turns a missing or unreadable registry into a message and a
/// non-zero exit instead of an uncaught exception.
int config_command_main_guarded(const program_args_t& args);

} // namespace oflm::config
