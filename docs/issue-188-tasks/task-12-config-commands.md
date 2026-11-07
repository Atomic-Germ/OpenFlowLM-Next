# Task 12: Config CLI Commands (show/set/unset/edit/gen-schema/migrate)

**Phase:** 4 (CLI Commands)
**Depends on:** Task 09, Task 11
**Files to create:** `src/config/config_cmd.hpp`, `src/config/config_cmd.cpp`
**Estimated complexity:** Large (~300 lines)

## Context

Implement the core `oflm config` subcommands: `show`, `set`, `unset`, `edit`, `gen-schema`, `migrate`. The `review` command comes in Task 13. Interactive picker (when `set` has no value) comes in Task 17.

## What to do

Create `src/config/config_cmd.hpp` and `src/config/config_cmd.cpp`.

## Header

```cpp
#pragma once
#include "program_args.hpp"

namespace oflm::config {

/// Main entry point for `oflm config <action>`.
/// The action is parsed_args.model_tag (e.g. "show", "set", "unset").
/// Returns 0 on success, non-zero on error.
int config_command_main(const program_args_t& args);

} // namespace oflm::config
```

## Implementation

Create `src/config/config_cmd.cpp` with these commands:

### `config show`
- Load EffectiveConfig
- If `--config-model` given: show settings for that model
- Else: show global settings
- If `--json` (global flag): output JSON
- Format: `key    value    (source)` where source is shipped/user/env/cli

### `config set`
- Validate key exists (warn if unknown, ask for confirmation — H5 fix)
- If no `--config-value`: print "Interactive picker not yet available, use --config-value" (Task 17 adds picker)
- Validate value against type/options
- Load settings.json (or create empty)
- Set value at correct path:
  - Global: `defaults.<key>`
  - Per-model: `settings.<type>.<size>.<key>`
- Update `defaults_version` to current shipped version
- Save with SettingsLock

### `config unset`
- Remove key from settings.json
- If model entry becomes empty, remove it
- Save with SettingsLock

### `config edit`
- Open settings.json in $EDITOR (or %EDITOR% on Windows)
- If file doesn't exist, create empty template first

### `config gen-schema`
- Output `generate_full_schema_json(shipped_registry)` to stdout
- Used by inject-schema.py at package time

### `config migrate`
- Force re-check for legacy migration
- Print result

## Key implementation details

```cpp
// Source tag strings (consistent with plan §Phase 4)
static const char* source_tag(SettingValue::Source s) {
    switch (s) {
        case SettingValue::SHIPPED: return "(shipped)";
        case SettingValue::USER: return "(user)";
        case SettingValue::ENV: return "(env)";
        case SettingValue::CLI: return "(cli)";
        case SettingValue::API: return "(api)";
    }
    return "(?)";
}

// Validate a value against a setting definition
static bool validate_value(const SettingDef& def, const std::string& value, std::string& error) {
    switch (def.type) {
        case SettingType::ENUM:
            for (const auto& opt : def.options) {
                if (opt.value == value) return true;
            }
            error = "Valid options:";
            for (const auto& opt : def.options) error += " " + opt.value;
            return false;
        case SettingType::INT: {
            try {
                long v = std::stol(value);
                if (def.min_value && v < *def.min_value) {
                    error = "Value must be >= " + std::to_string(*def.min_value);
                    return false;
                }
                if (def.max_value && v > *def.max_value) {
                    error = "Value must be <= " + std::to_string(*def.max_value);
                    return false;
                }
                return true;
            } catch (...) {
                error = "Value must be an integer";
                return false;
            }
        }
        case SettingType::BOOL:
            return value == "true" || value == "false" || value == "0" || value == "1" ||
                   value == "yes" || value == "no" || value == "on" || value == "off";
        case SettingType::STRING:
            return true;
    }
    return false;
}

// H5 fix: warn on unknown keys, allow with confirmation
static bool check_unknown_key(const std::string& key) {
    if (find_setting(key)) return true; // Known key
    std::cerr << "[OFLM] WARNING: '" << key << "' is not a known setting in this version of oflm." << std::endl;
    std::cerr << "[OFLM]   It will be stored but may not have any effect." << std::endl;
    std::cerr << "[OFLM]   Known settings:";
    for (const auto& k : get_all_setting_keys()) std::cerr << " " << k;
    std::cerr << std::endl;
    std::cerr << "Set anyway? [y/N] ";
    std::string response;
    std::getline(std::cin, response);
    return response == "y" || response == "Y" || response == "yes";
}
```

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -10

# Test show
export OFLM_SETTINGS_PATH=/tmp/test_settings.json
./build/src/oflm config show

# Test set + show + unset
./build/src/oflm config set --config-key pmode --config-value balanced
./build/src/oflm config show
./build/src/oflm config unset --config-key pmode

# Test gen-schema
./build/src/oflm config gen-schema | python3 -m json.tool | head -20

# Test per-model set
./build/src/oflm config set --config-model qwen3.8:27b --config-key prefill_mode --config-value lean
cat /tmp/test_settings.json
```

## Success criteria

- [ ] `config_cmd.hpp` and `config_cmd.cpp` created
- [ ] `config show` displays settings with sources
- [ ] `config show --json` outputs valid JSON
- [ ] `config set` writes to settings.json (global and per-model)
- [ ] `config set` validates values (enum options, int range, bool)
- [ ] `config set` warns on unknown keys (H5)
- [ ] `config unset` removes settings
- [ ] `config edit` opens $EDITOR
- [ ] `config gen-schema` outputs valid JSON with settings_schema + defaults_version
- [ ] `config migrate` triggers migration check
- [ ] Project compiles and all commands work
