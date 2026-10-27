#include "config/settings_resolver.hpp"
// Forward-declare instead of including utils/utils.hpp (which pulls in XRT via buffer.hpp)
namespace utils { std::string getenv_oflm(const char* oflm_name); }
#include <cstdlib>
#include <iostream>

// The environment block. POSIX: declared here at global scope, so it is not
// resolved inside the oflm::config namespace. MSVC: <stdlib.h> declares it,
// and environ is a macro there, so ::environ does not compile.
#ifdef _WIN32
static char** environment_block() { return _environ; }
#else
extern char** environ;
static char** environment_block() { return ::environ; }
#endif

namespace oflm::config {

std::set<std::string> g_external_env_vars;

void capture_external_env_vars() {
    // The environment as it was before we wrote anything into it. Anything
    // named OFLM_* that is here now is the user's.
    for (char** e = environment_block(); e && *e; ++e) {
        std::string entry(*e);
        size_t eq = entry.find('=');
        if (eq == std::string::npos) continue;
        std::string name = entry.substr(0, eq);
        if (name.rfind("OFLM_", 0) == 0) {
            g_external_env_vars.insert(name);
        }
    }
}

/// The value of an env var, but only if the USER set it. Ours is not an
/// override: it is the value we last delivered, and it belongs to the model
/// we last delivered it for.
static std::string external_env_value(const std::string& var) {
    if (!g_external_env_vars.count(var)) return std::string();
    return utils::getenv_oflm(var.c_str());
}


// Helper: get CLI flag value from program_args_t
// Maps cli_flag (e.g. "--prefill-mode") to the corresponding field.
// Returns nullopt if the flag was not explicitly set (i.e. value is the default).
// Pattern matches main.cpp: `bool got_power_mode = (parsed_args.power_mode != "performance");`
static std::optional<std::string> get_cli_flag_value(const program_args_t& args, const std::string& flag) {
    if (flag == "--prefill-mode") {
        // The parser's sentinel for "not supplied" is the empty string
        // (vm_args.hpp: default_value("")), and it validates the value to
        // fast|lean. An explicit `--prefill-mode fast` is therefore a real
        // override and must win over a saved or env `lean` -- "fast" cannot
        // be treated as "not supplied".
        if (!args.prefill_mode.empty()) return args.prefill_mode;
        return std::nullopt;
    }
    if (flag == "--pmode") {
        // Default is "performance" — only return if user changed it
        if (!args.power_mode.empty() && args.power_mode != "performance") return args.power_mode;
        return std::nullopt;
    }
    if (flag == "--ctx-len") {
        if (args.ctx_length > 0) return std::to_string(args.ctx_length);
        return std::nullopt;
    }
    if (flag == "--prefill-chunk-len") {
        if (args.prefill_chunk_len > 0) return std::to_string(args.prefill_chunk_len);
        return std::nullopt;
    }
    if (flag == "--speculative") {
        // A switch: absent means "not supplied", so the saved or env value
        // stands. Only an explicit --speculative overrides it.
        if (args.speculative) return "true";
        return std::nullopt;
    }
    return std::nullopt;
}

nlohmann::json resolve_settings(
    const EffectiveConfig& cfg,
    const std::string& model_tag,
    const program_args_t& args)
{
    nlohmann::json resolved;
    for (const auto& def : settings_registry()) {
        // Skip per-model settings if no model tag
        if (def.scope == SettingScope::PER_MODEL && model_tag.empty()) continue;

        // 4. CLI flags (highest precedence)
        if (!def.cli_flag.empty()) {
            if (auto v = get_cli_flag_value(args, def.cli_flag)) {
                resolved[def.key] = convert_type(*v, def.type);
                continue;
            }
        }

        // 3. Environment variables -- the ones the user set. The ones we
        // wrote ourselves are not an override: they belong to the model we
        // last delivered them for.
        if (!def.env_var.empty()) {
            std::string env_val = external_env_value(def.env_var);
            if (!env_val.empty()) {
                resolved[def.key] = convert_type(env_val, def.type);
                continue;
            }
        }

        // 2. User settings
        if (model_tag.empty()) {
            // Global setting — check user defaults
            if (cfg.user_settings.contains("defaults") &&
                cfg.user_settings["defaults"].contains(def.key)) {
                resolved[def.key] = cfg.user_settings["defaults"][def.key];
                continue;
            }
        } else {
            // Per-model setting — check user settings overlay
            auto [type, size] = split_tag(model_tag);
            if (cfg.user_settings.contains("settings") &&
                cfg.user_settings["settings"].contains(type) &&
                cfg.user_settings["settings"][type].contains(size) &&
                cfg.user_settings["settings"][type][size].contains(def.key)) {
                resolved[def.key] = cfg.user_settings["settings"][type][size][def.key];
                continue;
            }
            // The user's app-wide default applies to every model that does
            // not override the setting for itself. Without this a setting
            // changed app-wide would show up in the global view and in no
            // model at all.
            if (cfg.user_settings.contains("defaults") &&
                cfg.user_settings["defaults"].contains(def.key)) {
                resolved[def.key] = cfg.user_settings["defaults"][def.key];
                continue;
            }
        }

        // 1. Shipped defaults
        if (cfg.shipped_registry.contains("defaults") &&
            cfg.shipped_registry["defaults"].contains(def.key)) {
            resolved[def.key] = cfg.shipped_registry["defaults"][def.key];
            continue;
        }
        // Per-model shipped default from registry entry
        if (!def.registry_key.empty() && !model_tag.empty()) {
            auto entry = get_merged_model_entry(cfg, model_tag);
            if (!entry.empty()) {
                // Handle nested keys like "details.think"
                if (def.registry_key.find('.') != std::string::npos) {
                    auto dot_pos = def.registry_key.find('.');
                    std::string parent = def.registry_key.substr(0, dot_pos);
                    std::string child = def.registry_key.substr(dot_pos + 1);
                    if (entry.contains(parent) && entry[parent].contains(child)) {
                        resolved[def.key] = entry[parent][child];
                        continue;
                    }
                } else if (entry.contains(def.registry_key)) {
                    resolved[def.key] = entry[def.registry_key];
                    continue;
                }
            }
        }
        // Built-in default from registry definition
        if (!def.default_value.empty()) {
            resolved[def.key] = convert_type(def.default_value, def.type);
        }
    }
    return resolved;
}

void apply_settings(const nlohmann::json& resolved) {
    for (const auto& def : settings_registry()) {
        if (!resolved.contains(def.key)) continue;
        if (def.delivery != DeliveryMechanism::ENV_VAR) continue;

        std::string value = resolved[def.key].is_string()
            ? resolved[def.key].get<std::string>()
            : resolved[def.key].dump();

#ifdef _WIN32
        _putenv_s(def.env_var.c_str(), value.c_str());
#else
        setenv(def.env_var.c_str(), value.c_str(), 1);
#endif
    }
}

std::optional<SettingValue> get_setting_value(
    const EffectiveConfig& cfg,
    const std::string& model_tag,
    const std::string& setting_key,
    const program_args_t& args)
{
    auto def = find_setting(setting_key);
    if (!def) return std::nullopt;

    // Helper: extract string value from JSON (no quotes for strings)
    auto json_to_string = [](const nlohmann::json& j) -> std::string {
        if (j.is_string()) return j.get<std::string>();
        if (j.is_boolean()) return j.get<bool>() ? "true" : "false";
        if (j.is_number()) return j.dump();
        return j.dump();
    };

    // Check CLI first
    if (!def->cli_flag.empty()) {
        if (auto v = get_cli_flag_value(args, def->cli_flag)) {
            return SettingValue{*v, SettingValue::CLI};
        }
    }
    // Check env var -- the ones the user set, not the ones we wrote.
    if (!def->env_var.empty()) {
        std::string env_val = external_env_value(def->env_var);
        if (!env_val.empty()) {
            return SettingValue{env_val, SettingValue::ENV};
        }
    }
    // Check user settings
    if (model_tag.empty()) {
        if (cfg.user_settings.contains("defaults") && cfg.user_settings["defaults"].contains(setting_key)) {
            return SettingValue{json_to_string(cfg.user_settings["defaults"][setting_key]), SettingValue::USER};
        }
    } else {
        auto [type, size] = split_tag(model_tag);
        if (cfg.user_settings.contains("settings") &&
            cfg.user_settings["settings"].contains(type) &&
            cfg.user_settings["settings"][type].contains(size) &&
            cfg.user_settings["settings"][type][size].contains(setting_key)) {
            return SettingValue{json_to_string(cfg.user_settings["settings"][type][size][setting_key]), SettingValue::USER};
        }
        // The user's app-wide default applies to every model that does not
        // override the setting for itself.
        if (cfg.user_settings.contains("defaults") && cfg.user_settings["defaults"].contains(setting_key)) {
            return SettingValue{json_to_string(cfg.user_settings["defaults"][setting_key]), SettingValue::USER};
        }
    }
    // Check shipped defaults
    if (cfg.shipped_registry.contains("defaults") && cfg.shipped_registry["defaults"].contains(setting_key)) {
        return SettingValue{json_to_string(cfg.shipped_registry["defaults"][setting_key]), SettingValue::SHIPPED};
    }
    if (!def->registry_key.empty() && !model_tag.empty()) {
        auto entry = get_merged_model_entry(cfg, model_tag);
        if (!entry.empty()) {
            if (def->registry_key.find('.') != std::string::npos) {
                auto dot_pos = def->registry_key.find('.');
                std::string parent = def->registry_key.substr(0, dot_pos);
                std::string child = def->registry_key.substr(dot_pos + 1);
                if (entry.contains(parent) && entry[parent].contains(child)) {
                    return SettingValue{json_to_string(entry[parent][child]), SettingValue::SHIPPED};
                }
            } else if (entry.contains(def->registry_key)) {
                return SettingValue{json_to_string(entry[def->registry_key]), SettingValue::SHIPPED};
            }
        }
    }
    if (!def->default_value.empty()) {
        return SettingValue{def->default_value, SettingValue::SHIPPED};
    }
    return std::nullopt;
}

EffectiveConfig* g_effective_config = nullptr;
nlohmann::json* g_resolved_settings = nullptr;

} // namespace oflm::config
