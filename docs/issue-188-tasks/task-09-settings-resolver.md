# Task 09: Settings Resolver Implementation

**Phase:** 3 (Settings Resolution and Application)
**Depends on:** Task 02, Task 05
**Files to create:** `src/config/settings_resolver.hpp`, `src/config/settings_resolver.cpp`
**Estimated complexity:** Medium (~150 lines)

## Context

The resolver computes effective settings by applying the precedence chain: shipped defaults < user settings < env vars < CLI flags. The `apply_settings()` function then delivers values via each setting's declared `DeliveryMechanism` (the "reachability" fix).

Read parent plan **§2.1 (Configuration Reachability)** and **§Phase 3** in `docs/issue-188-implementation-plan.md`.

## What to do

Create `src/config/settings_resolver.hpp` and `src/config/settings_resolver.cpp`.

## Header (`settings_resolver.hpp`)

```cpp
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
```

## Implementation (`settings_resolver.cpp`)

```cpp
#include "config/settings_resolver.hpp"
#include "utils/utils.hpp"
#include <cstdlib>
#include <iostream>

namespace oflm::config {

// Helper: get CLI flag value from program_args_t
// Maps cli_flag (e.g. "--prefill-mode") to the corresponding field
static std::optional<std::string> get_cli_flag_value(const program_args_t& args, const std::string& flag) {
    if (flag == "--prefill-mode") {
        if (args.prefill_mode != "fast") return args.prefill_mode; // "fast" is default, only return if changed
        // Actually, check if the flag was explicitly set — for now, return if not default
        return std::nullopt; // TODO: check if explicitly set via Boost
    }
    if (flag == "--pmode") {
        return args.power_mode; // always has a value (default "performance")
    }
    if (flag == "--ctx-len") {
        if (args.ctx_length > 0) return std::to_string(args.ctx_length);
        return std::nullopt;
    }
    if (flag == "--prefill-chunk-len") {
        if (args.prefill_chunk_len > 0) return std::to_string(args.prefill_chunk_len);
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

        // 3. Environment variables
        if (!def.env_var.empty()) {
            std::string env_val = utils::getenv_oflm(def.env_var.c_str());
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
            if (!entry.empty() && entry.contains(def.registry_key)) {
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

    // Check CLI first
    if (!def->cli_flag.empty()) {
        if (auto v = get_cli_flag_value(args, def->cli_flag)) {
            return SettingValue{*v, SettingValue::CLI};
        }
    }
    // Check env var
    if (!def->env_var.empty()) {
        std::string env_val = utils::getenv_oflm(def->env_var.c_str());
        if (!env_val.empty()) {
            return SettingValue{env_val, SettingValue::ENV};
        }
    }
    // Check user settings
    if (model_tag.empty()) {
        if (cfg.user_settings.contains("defaults") && cfg.user_settings["defaults"].contains(setting_key)) {
            return SettingValue{cfg.user_settings["defaults"][setting_key].dump(), SettingValue::USER};
        }
    } else {
        auto [type, size] = split_tag(model_tag);
        if (cfg.user_settings.contains("settings") &&
            cfg.user_settings["settings"].contains(type) &&
            cfg.user_settings["settings"][type].contains(size) &&
            cfg.user_settings["settings"][type][size].contains(setting_key)) {
            return SettingValue{cfg.user_settings["settings"][type][size][setting_key].dump(), SettingValue::USER};
        }
    }
    // Check shipped defaults
    if (cfg.shipped_registry.contains("defaults") && cfg.shipped_registry["defaults"].contains(setting_key)) {
        return SettingValue{cfg.shipped_registry["defaults"][setting_key].dump(), SettingValue::SHIPPED};
    }
    if (!def->registry_key.empty() && !model_tag.empty()) {
        auto entry = get_merged_model_entry(cfg, model_tag);
        if (!entry.empty()) {
            if (def->registry_key.find('.') != std::string::npos) {
                auto dot_pos = def->registry_key.find('.');
                std::string parent = def->registry_key.substr(0, dot_pos);
                std::string child = def->registry_key.substr(dot_pos + 1);
                if (entry.contains(parent) && entry[parent].contains(child)) {
                    return SettingValue{entry[parent][child].dump(), SettingValue::SHIPPED};
                }
            } else if (entry.contains(def->registry_key)) {
                return SettingValue{entry[def->registry_key].dump(), SettingValue::SHIPPED};
            }
        }
    }
    if (!def->default_value.empty()) {
        return SettingValue{def->default_value, SettingValue::SHIPPED};
    }
    return std::nullopt;
}

} // namespace oflm::config
```

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -10
```

## Success criteria

- [ ] `settings_resolver.hpp` and `.cpp` created
- [ ] `resolve_settings()` implements full precedence chain
- [ ] `apply_settings()` setenvs ENV_VAR settings
- [ ] `get_setting_value()` tracks source
- [ ] Project compiles
