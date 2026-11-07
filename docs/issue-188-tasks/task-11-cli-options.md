# Task 11: CLI Options in vm_args.hpp

**Phase:** 4 (CLI Commands)
**Depends on:** None
**Files to modify:** `src/include/utils/vm_args.hpp`, `src/include/program_args.hpp`
**Estimated complexity:** Small (~30 lines)

## Context

Add config-specific options to the Boost program_options parser. The `config` action is **positional** — parsed from the existing `model_tag` positional when `command == "config"`.

## What to do

1. Add config fields to `program_args_t` in `src/include/program_args.hpp`
2. Add config options to the Boost options_description in `src/include/utils/vm_args.hpp`
3. Allow `config` as a valid command
4. Add per-command validation (config options only valid for config command)

## Code changes

### In `src/include/program_args.hpp`

Add these fields to `program_args_t`:

```cpp
// Config command fields
std::string config_key;
std::string config_value;
std::string config_model;
bool config_accept_shipped = false;
bool config_keep_all = false;
```

### In `src/include/utils/vm_args.hpp`

1. Add to the options_description (around line 154, after existing options):

```cpp
("config-key", po::value<std::string>(&parsed_args.config_key), "Setting key for config set/unset")
("config-value", po::value<std::string>(&parsed_args.config_value), "Setting value for config set")
("config-model", po::value<std::string>(&parsed_args.config_model), "Model tag for per-model config operations")
("config-accept-shipped", po::bool_switch(&parsed_args.config_accept_shipped), "config review: remove all redundant overrides")
("config-keep-all", po::bool_switch(&parsed_args.config_keep_all), "config review: keep all overrides, just update version")
```

Note: The options need to be bound to `parsed_args` fields. Check how existing options are bound (e.g. `po::value<std::string>(&parsed_args.power_mode)`).

2. Add `"config"` to the list of valid commands. Find where commands are validated (around line 280-297) and add `config` to the allowed list.

3. Add per-command validation: config-specific options (`--config-key`, `--config-value`, etc.) should only be valid when `command == "config"`. Add a check similar to the existing per-command checks (around line 259-278).

4. The `model_tag` positional will hold the config action (show/set/unset/etc.) when command is "config". No changes needed to positional parsing — just document this.

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -10
./build/src/oflm config show --help 2>&1 | head -20
./build/src/oflm --config-key test config show 2>&1 | head -5  # Should work
./build/src/oflm run --config-key test 2>&1 | head -5  # Should error: config option not valid for run
```

## Success criteria

- [ ] `program_args_t` has config_key, config_value, config_model, config_accept_shipped, config_keep_all
- [ ] Boost options defined for all 5 config flags
- [ ] `config` is a valid command
- [ ] Config options rejected for non-config commands
- [ ] Project compiles
- [ ] `oflm config show` parses without error (even if command not fully implemented yet)
