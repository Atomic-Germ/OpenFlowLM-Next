# Task 01: Settings Registry Header

**Phase:** 1 (Settings Registry)
**Depends on:** None
**Files to create:** `src/config/settings_registry.hpp`
**Estimated complexity:** Small (~80 lines)

## Context

This is the foundation of the entire settings system. The settings registry is the **single source of truth** for all user-facing configuration. Every other component (CLI, UI, schema generation, engine integration) reads from this registry.

Read the parent plan section **§2.1 (Configuration Reachability)** and **§2.2 (Settings Registry)** in `docs/issue-188-implementation-plan.md` for full context.

## What to do

Create `src/config/settings_registry.hpp` with:
1. The `oflm::config` namespace
2. Enums: `SettingType`, `SettingScope`, `DeliveryMechanism`
3. Structs: `SettingOption`, `SettingDef`
4. Function declarations

## Code to write

Create the file `src/config/settings_registry.hpp` with exactly this content:

```cpp
/// \file settings_registry.hpp
/// \brief Settings registry — single source of truth for all user-facing settings
/// \note Every setting is defined here exactly once. Schema, help text, picker
///       options, validation, and delivery mechanism all derive from this registry.
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace oflm::config {

/// The type of a setting's value
enum class SettingType { ENUM, INT, BOOL, STRING };

/// Where a setting can be applied
enum class SettingScope { GLOBAL, PER_MODEL, BOTH };

/// How a setting's value reaches the code that consumes it.
/// This is the "reachability" mechanism — it replaces ad-hoc setenv() bridges.
enum class DeliveryMechanism {
    DIRECT,     ///< Passed via function parameter (same process, no DLL boundary)
    ENV_VAR,    ///< setenv() bridge — for values crossing the OFLM_DLL boundary
    API_FIELD,  ///< Per-request override via REST API (serve mode)
    REGISTRY,   ///< Read from model registry entry at load time
};

/// One valid option for an ENUM setting
struct SettingOption {
    std::string value;        ///< The value as stored in JSON (e.g. "lean")
    std::string description;  ///< Human-readable description for picker/help
};

/// Definition of one user-facing setting
struct SettingDef {
    std::string key;             ///< Canonical name in settings.json (e.g. "prefill_mode")
    std::string cli_flag;        ///< CLI flag (e.g. "--prefill-mode"), empty if none
    std::string env_var;         ///< Env var name (e.g. "OFLM_OPEN_PREFILL_MODE"), empty if none
    std::string description;     ///< Human-readable description
    SettingType type;            ///< Value type
    std::vector<SettingOption> options;  ///< Valid options (ENUM only)
    std::optional<long> min_value;       ///< Minimum value (INT only)
    std::optional<long> max_value;       ///< Maximum value (INT only)
    std::vector<long> presets;           ///< Quick-pick presets (INT only)
    std::string default_value;           ///< Fallback default, empty if from registry entry
    SettingScope scope;                  ///< GLOBAL, PER_MODEL, or BOTH
    bool reload_required;                ///< Changing requires model reload
    std::string category;                ///< Grouping: "performance", "memory", "generation"
    std::string registry_key;            ///< Maps to model_list.json entry key (e.g. "default_context_length")
    DeliveryMechanism delivery;          ///< How the value reaches its consumer
};

/// The registry — the ONLY place settings are defined.
/// Returns a static vector of all SettingDef entries.
const std::vector<SettingDef>& settings_registry();

/// Find a setting by key. Returns nullopt if not found.
std::optional<SettingDef> find_setting(const std::string& key);

/// Get all settings for a given scope.
std::vector<SettingDef> settings_for_scope(SettingScope scope);

/// Generate the settings_schema JSON block for model_list.json.
/// Output format: { "version": 1, "settings": [ ... ] }
nlohmann::json generate_schema_json();

/// Generate the full gen-schema output (settings_schema + defaults_version).
/// Output format: { "settings_schema": { ... }, "defaults_version": "..." }
nlohmann::json generate_full_schema_json(const nlohmann::json& shipped_registry);

/// Compute the defaults_version hash from a shipped registry.
/// Hashes: top-level "defaults" block + per-model default values from settings registry entries.
std::string compute_defaults_version(const nlohmann::json& shipped_registry);

/// Generate help text lines for `oflm config --help`.
std::vector<std::string> generate_help_lines();

/// Get all setting keys as a vector of strings.
std::vector<std::string> get_all_setting_keys();

/// Convert a string value to the setting's declared type for JSON storage.
/// INT → int64, BOOL → bool, ENUM/STRING → string.
nlohmann::json convert_type(const std::string& value, SettingType type);

} // namespace oflm::config
```

## Test

```bash
# Create the directory and file, then verify it compiles as part of the project
mkdir -p src/config
# (file is created above)

# Verify the header is syntactically valid by compiling a test TU
cat > /tmp/test_registry_header.cpp << 'EOF'
#include "config/settings_registry.hpp"
int main() { return 0; }
EOF

# Compile just the header check (from src/ directory)
g++ -std=c++20 -I src/include -I src -c /tmp/test_registry_header.cpp -o /tmp/test_registry_header.o && echo "HEADER OK"
```

## Success criteria

- [ ] File `src/config/settings_registry.hpp` exists
- [ ] File compiles without errors (header check above passes)
- [ ] All enums, structs, and function declarations are present
- [ ] Namespace is `oflm::config`
- [ ] `DeliveryMechanism` enum has all 4 values: DIRECT, ENV_VAR, API_FIELD, REGISTRY
- [ ] `SettingDef` struct has all fields from the plan §2.2
- [ ] No TODO or FIXME comments left
