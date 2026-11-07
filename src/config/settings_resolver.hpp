#pragma once
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include "program_args.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>

namespace oflm::config {

/// The OFLM_* environment variables that were set when the process started.
///
/// The env-var delivery mechanism writes the settings into the environment,
/// and the environment is process-wide: a server that serves several models
/// would otherwise have the first model's values shadow every later one. So
/// the resolver treats an env var as an override only when it was the user's
/// to begin with -- one we wrote ourselves is not an override, it is our own
/// last write.
///
/// Call capture_external_env_vars() before the first apply_settings().
extern std::set<std::string> g_external_env_vars;

/// Record which OFLM_* variables the user set. Must run before the first
/// apply_settings(), or our own writes would be taken for the user's.
void capture_external_env_vars();

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

/// Global effective config — set by main.cpp, read by Runner/RestHandler.
/// Valid only after load_effective_config() has been called.
extern EffectiveConfig* g_effective_config;

/// Global resolved settings — set by main.cpp after resolve_settings().
extern nlohmann::json* g_resolved_settings;

} // namespace oflm::config
