# Task 02: Settings Registry Implementation

**Phase:** 1 (Settings Registry)
**Depends on:** Task 01
**Files to create:** `src/config/settings_registry.cpp`
**Estimated complexity:** Medium (~200 lines)

## Context

This file populates the registry with the 5 initial user-facing settings and implements the schema/help generation functions. The registry data is the single source of truth — everything else derives from it.

Read the parent plan sections **§2.1**, **§2.2**, **§2.3** in `docs/issue-188-implementation-plan.md`.

## What to do

Create `src/config/settings_registry.cpp` with:
1. The registry data (5 settings as a static vector)
2. `settings_registry()` — returns the static vector
3. `find_setting()` — lookup by key
4. `settings_for_scope()` — filter by scope
5. `generate_schema_json()` — serialize to settings_schema block
6. `generate_full_schema_json()` — schema + defaults_version
7. `compute_defaults_version()` — hash of defaults
8. `generate_help_lines()` — format for --help
9. `get_all_setting_keys()` — list of keys
10. `convert_type()` — string → typed JSON

## The 5 settings to register

| Key | Type | Options | Default | Scope | CLI Flag | Env Var | Delivery | Registry Key | Reload | Category |
|-----|------|---------|---------|-------|----------|---------|----------|--------------|--------|----------|
| `prefill_mode` | ENUM | fast, lean | `fast` | PER_MODEL | `--prefill-mode` | `OFLM_OPEN_PREFILL_MODE` | ENV_VAR | — | Yes | performance |
| `pmode` | ENUM | default, powersaver, balanced, performance, turbo | `performance` | GLOBAL | `--pmode` | — | DIRECT | — | No | performance |
| `ctx_len` | INT | min 512 | (from entry) | PER_MODEL | `--ctx-len` | — | DIRECT | `default_context_length` | Yes | memory |
| `prefill_chunk_len` | INT | min 512 | (from entry) | PER_MODEL | `--prefill-chunk-len` | — | DIRECT | `max_prefill_len` | Yes | memory |
| `think` | BOOL | true/false | (from entry) | PER_MODEL | — | — | API_FIELD | `details.think` | No | generation |

Descriptions:
- `prefill_mode`: "Memory/speed trade-off for prefill"
  - fast: "Default prefill, higher memory"
  - lean: "Lower memory, ~15% slower"
- `pmode`: "NPU power mode"
  - default: "Let the driver decide"
  - powersaver: "Lower power, reduced performance"
  - balanced: "Balance of power and performance"
  - performance: "Maximum performance (oflm default)"
  - turbo: "Maximum clocks, highest power"
- `ctx_len`: "Context window length (overrides model default)"
- `prefill_chunk_len`: "Maximum prefill chunk length (overrides model default)"
- `think`: "Enable reasoning/thinking mode by default"

## Code to write

Create `src/config/settings_registry.cpp`:

```cpp
/// \file settings_registry.cpp
/// \brief Settings registry implementation — populates the registry and generates schema/help
#include "config/settings_registry.hpp"
#include <algorithm>
#include <sstream>
#include <functional>

namespace oflm::config {

namespace {

// Schema version — bump when the schema format changes
const int SCHEMA_VERSION = 1;

// The registry — populated once, returned by reference
const std::vector<SettingDef>& build_registry() {
    static const std::vector<SettingDef> registry = {
        {
            .key = "prefill_mode",
            .cli_flag = "--prefill-mode",
            .env_var = "OFLM_OPEN_PREFILL_MODE",
            .description = "Memory/speed trade-off for prefill",
            .type = SettingType::ENUM,
            .options = {
                {"fast", "Default prefill, higher memory"},
                {"lean", "Lower memory, ~15% slower"},
            },
            .min_value = std::nullopt,
            .max_value = std::nullopt,
            .presets = {},
            .default_value = "fast",
            .scope = SettingScope::PER_MODEL,
            .reload_required = true,
            .category = "performance",
            .registry_key = "",
            .delivery = DeliveryMechanism::ENV_VAR,
        },
        {
            .key = "pmode",
            .cli_flag = "--pmode",
            .env_var = "",
            .description = "NPU power mode",
            .type = SettingType::ENUM,
            .options = {
                {"default", "Let the driver decide"},
                {"powersaver", "Lower power, reduced performance"},
                {"balanced", "Balance of power and performance"},
                {"performance", "Maximum performance (oflm default)"},
                {"turbo", "Maximum clocks, highest power"},
            },
            .min_value = std::nullopt,
            .max_value = std::nullopt,
            .presets = {},
            .default_value = "performance",
            .scope = SettingScope::GLOBAL,
            .reload_required = false,
            .category = "performance",
            .registry_key = "",
            .delivery = DeliveryMechanism::DIRECT,
        },
        {
            .key = "ctx_len",
            .cli_flag = "--ctx-len",
            .env_var = "",
            .description = "Context window length (overrides model default)",
            .type = SettingType::INT,
            .options = {},
            .min_value = 512,
            .max_value = std::nullopt,
            .presets = {2048, 4096, 8192, 16384, 32768},
            .default_value = "",
            .scope = SettingScope::PER_MODEL,
            .reload_required = true,
            .category = "memory",
            .registry_key = "default_context_length",
            .delivery = DeliveryMechanism::DIRECT,
        },
        {
            .key = "prefill_chunk_len",
            .cli_flag = "--prefill-chunk-len",
            .env_var = "",
            .description = "Maximum prefill chunk length (overrides model default)",
            .type = SettingType::INT,
            .options = {},
            .min_value = 512,
            .max_value = std::nullopt,
            .presets = {1024, 2048, 4096, 8192},
            .default_value = "",
            .scope = SettingScope::PER_MODEL,
            .reload_required = true,
            .category = "memory",
            .registry_key = "max_prefill_len",
            .delivery = DeliveryMechanism::DIRECT,
        },
        {
            .key = "think",
            .cli_flag = "",
            .env_var = "",
            .description = "Enable reasoning/thinking mode by default",
            .type = SettingType::BOOL,
            .options = {},
            .min_value = std::nullopt,
            .max_value = std::nullopt,
            .presets = {},
            .default_value = "",
            .scope = SettingScope::PER_MODEL,
            .reload_required = false,
            .category = "generation",
            .registry_key = "details.think",
            .delivery = DeliveryMechanism::API_FIELD,
        },
    };
    return registry;
}

std::string scope_to_string(SettingScope scope) {
    switch (scope) {
        case SettingScope::GLOBAL: return "global";
        case SettingScope::PER_MODEL: return "per_model";
        case SettingScope::BOTH: return "both";
    }
    return "unknown";
}

std::string type_to_string(SettingType type) {
    switch (type) {
        case SettingType::ENUM: return "enum";
        case SettingType::INT: return "int";
        case SettingType::BOOL: return "bool";
        case SettingType::STRING: return "string";
    }
    return "unknown";
}

std::string delivery_to_string(DeliveryMechanism d) {
    switch (d) {
        case DeliveryMechanism::DIRECT: return "direct";
        case DeliveryMechanism::ENV_VAR: return "env_var";
        case DeliveryMechanism::API_FIELD: return "api_field";
        case DeliveryMechanism::REGISTRY: return "registry";
    }
    return "unknown";
}

nlohmann::json setting_to_schema_json(const SettingDef& def) {
    nlohmann::json j;
    j["key"] = def.key;
    j["type"] = type_to_string(def.type);
    j["description"] = def.description;
    j["default"] = def.default_value;
    j["scope"] = scope_to_string(def.scope);
    j["reload_required"] = def.reload_required;
    j["category"] = def.category;
    j["delivery"] = delivery_to_string(def.delivery);
    if (!def.env_var.empty()) j["env_var"] = def.env_var;
    if (!def.cli_flag.empty()) j["cli_flag"] = def.cli_flag;
    if (!def.registry_key.empty()) j["registry_key"] = def.registry_key;
    if (def.type == SettingType::ENUM) {
        nlohmann::json opts = nlohmann::json::array();
        for (const auto& opt : def.options) {
            opts.push_back({{"value", opt.value}, {"description", opt.description}});
        }
        j["options"] = opts;
    }
    if (def.type == SettingType::INT) {
        if (def.min_value) j["min"] = *def.min_value;
        if (def.max_value) j["max"] = *def.max_value;
        if (!def.presets.empty()) {
            nlohmann::json p = nlohmann::json::array();
            for (auto v : def.presets) p.push_back(v);
            j["presets"] = p;
        }
    }
    return j;
}

} // anonymous namespace

const std::vector<SettingDef>& settings_registry() {
    return build_registry();
}

std::optional<SettingDef> find_setting(const std::string& key) {
    for (const auto& def : settings_registry()) {
        if (def.key == key) return def;
    }
    return std::nullopt;
}

std::vector<SettingDef> settings_for_scope(SettingScope scope) {
    std::vector<SettingDef> result;
    for (const auto& def : settings_registry()) {
        if (def.scope == scope || def.scope == SettingScope::BOTH) {
            result.push_back(def);
        }
    }
    return result;
}

nlohmann::json generate_schema_json() {
    nlohmann::json schema;
    schema["version"] = SCHEMA_VERSION;
    nlohmann::json settings = nlohmann::json::array();
    for (const auto& def : settings_registry()) {
        settings.push_back(setting_to_schema_json(def));
    }
    schema["settings"] = settings;
    return schema;
}

std::string compute_defaults_version(const nlohmann::json& shipped_registry) {
    // Hash: top-level "defaults" block + per-model default values from settings registry entries
    nlohmann::json hash_input = nlohmann::json::object();
    if (shipped_registry.contains("defaults")) {
        hash_input["defaults"] = shipped_registry["defaults"];
    }
    // Collect per-model defaults for settings that have a registry_key
    nlohmann::json per_model = nlohmann::json::object();
    if (shipped_registry.contains("models")) {
        for (const auto& def : settings_registry()) {
            if (def.registry_key.empty()) continue;
            for (const auto& [type, sizes] : shipped_registry["models"].items()) {
                for (const auto& [size, entry] : sizes.items()) {
                    if (entry.contains(def.registry_key)) {
                        per_model[type + ":" + size + "." + def.key] = entry[def.registry_key];
                    }
                }
            }
        }
    }
    hash_input["per_model_defaults"] = per_model;
    // Simple hash: use std::hash on the serialized JSON
    std::string serialized = hash_input.dump();
    std::hash<std::string> hasher;
    size_t h = hasher(serialized);
    std::ostringstream oss;
    oss << std::hex << h;
    return oss.str();
}

nlohmann::json generate_full_schema_json(const nlohmann::json& shipped_registry) {
    nlohmann::json result;
    result["settings_schema"] = generate_schema_json();
    result["defaults_version"] = compute_defaults_version(shipped_registry);
    return result;
}

std::vector<std::string> generate_help_lines() {
    std::vector<std::string> lines;
    lines.push_back("Config settings (managed via 'oflm config' or ~/.config/oflm/settings.json):");
    lines.push_back("");
    for (const auto& def : settings_registry()) {
        std::ostringstream oss;
        oss << "  " << def.key;
        if (!def.cli_flag.empty()) oss << " (" << def.cli_flag << ")";
        oss << " — " << def.description;
        lines.push_back(oss.str());
        if (def.type == SettingType::ENUM) {
            std::ostringstream opts;
            opts << "    Options:";
            for (const auto& opt : def.options) {
                opts << " " << opt.value;
            }
            lines.push_back(opts.str());
        }
        if (!def.default_value.empty()) {
            lines.push_back("    Default: " + def.default_value);
        }
        lines.push_back("");
    }
    return lines;
}

std::vector<std::string> get_all_setting_keys() {
    std::vector<std::string> keys;
    for (const auto& def : settings_registry()) {
        keys.push_back(def.key);
    }
    return keys;
}

nlohmann::json convert_type(const std::string& value, SettingType type) {
    switch (type) {
        case SettingType::INT:
            try { return nlohmann::json(std::stol(value)); }
            catch (...) { return nlohmann::json(value); }
        case SettingType::BOOL:
            if (value == "true" || value == "1" || value == "yes" || value == "on") return nlohmann::json(true);
            if (value == "false" || value == "0" || value == "no" || value == "off") return nlohmann::json(false);
            return nlohmann::json(value);
        case SettingType::ENUM:
        case SettingType::STRING:
        default:
            return nlohmann::json(value);
    }
}

} // namespace oflm::config
```

## CMake

Verify that `src/config/settings_registry.cpp` is picked up by the build. Check `src/CMakeLists.txt` around line 293 — the `file(GLOB ...)` should include `config/*.cpp`. If not, add:

```cmake
file(GLOB OFLM_CONFIG_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/config/*.cpp")
list(APPEND SOURCES ${OFLM_CONFIG_SOURCES})
```

## Test

```bash
# Build
cmake --preset linux-default 2>&1 | tail -5
cmake --build --preset linux-default --target oflm 2>&1 | tail -20

# Verify the registry compiles and links
# (Full test comes in Task 03)
```

## Success criteria

- [ ] File `src/config/settings_registry.cpp` exists
- [ ] Project compiles without errors
- [ ] `settings_registry()` returns 5 settings
- [ ] All 5 settings have correct key, type, scope, delivery, category
- [ ] `find_setting("prefill_mode")` returns a def with `delivery == ENV_VAR`
- [ ] `find_setting("nonexistent")` returns nullopt
- [ ] `generate_schema_json()` produces valid JSON with `version: 1` and 5 settings
- [ ] `compute_defaults_version()` returns a hex string
- [ ] `convert_type("8192", INT)` returns JSON number 8192
- [ ] `convert_type("true", BOOL)` returns JSON boolean true
- [ ] `convert_type("lean", ENUM)` returns JSON string "lean"
- [ ] CMake picks up the new source file (check build output)
