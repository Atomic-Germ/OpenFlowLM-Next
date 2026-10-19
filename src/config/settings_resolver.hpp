#pragma once
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include "program_args.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace oflm::config {

/// Resolve effective settings for a model tag.
/// Precedence: shipped defaults < user settings < env vars < CLI flags
/// Returns JSON object: { "prefill_mode": "lean", "ctx_len": 16384, ... }
nlohmann::json resolve_settings(
    const EffectiveConfig& cfg,
    const std::string& model_tag,
    const program_args_t& args);

/// Apply resolved settings using each setting's declared delivery mechanism.
/// ENV_VAR: setenv() each setting that has env_var defined.
/// DIRECT/API_FIELD/REGISTRY: no-op (value is in the JSON for consumers).
void apply_settings(const nlohmann::json& resolved);

/// Get a single setting's effective value with source tracking.
struct SettingValue {
    std::string value;
    enum Source { SHIPPED, USER, ENV, CLI, API } source;
};
std::optional<SettingValue> get_setting_value(
    const EffectiveConfig& cfg,
    const std::string& model_tag,
    const std::string& setting_key,
    const program_args_t& args);

} // namespace oflm::config
