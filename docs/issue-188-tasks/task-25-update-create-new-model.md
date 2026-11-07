# Task 25: Update create_new_model.md

**Phase:** 10 (Documentation)
**Depends on:** Task 01 (needs to know the registry structure)
**Files to modify:** `src/create_new_model.md`
**Estimated complexity:** Small (~40 lines added)

## Context

`src/create_new_model.md` documents how to add a new model to the registry. It needs a new section documenting how to add a new **setting** to the settings registry.

## What to do

Add a new section to `src/create_new_model.md` explaining how to add a new setting to the settings registry.

## Where to add

Find a good spot — probably after the existing documentation of model entry fields (around line 280-339). Add a new section:

```markdown
## Adding a new setting

User-facing settings are defined in **one place**: the settings registry at
`src/config/settings_registry.cpp`. Everything else (CLI flags, help text,
interactive picker, schema, validation) is generated from the registry.

### Step 1: Add to the registry

In `src/config/settings_registry.cpp`, add a new `SettingDef` entry to the
`build_registry()` vector:

```cpp
{
    .key = "decode_mode",                    // Canonical name in settings.json
    .cli_flag = "--decode-mode",             // CLI flag (empty if none)
    .env_var = "OFLM_OPEN_DECODE_MODE",      // Env var (for ENV_VAR delivery)
    .description = "Decode mode: exact or fast",
    .type = SettingType::ENUM,
    .options = {
        {"exact", "Bit-exact, slower"},
        {"fast", "Faster, approximate"},
    },
    .min_value = std::nullopt,
    .max_value = std::nullopt,
    .presets = {},
    .default_value = "exact",
    .scope = SettingScope::PER_MODEL,        // GLOBAL, PER_MODEL, or BOTH
    .reload_required = true,                 // Needs model reload if changed
    .category = "performance",               // Grouping for help/picker
    .registry_key = "",                      // Maps to model_list.json entry key
    .delivery = DeliveryMechanism::ENV_VAR,  // How the value reaches the engine
},
```

### Step 2: Choose the delivery mechanism

The `delivery` field controls how the setting's value reaches the code that
consumes it:

| Delivery | When to use | What happens |
|----------|-------------|--------------|
| `ENV_VAR` | Value must cross the `OFLM_DLL` boundary | `apply_settings()` calls `setenv()` automatically |
| `DIRECT` | Same process, passed via function parameter | Value is in the resolved JSON; consumer reads it |
| `API_FIELD` | Per-request override via REST API | Value is used as default; API field overrides |
| `REGISTRY` | Value lives in the model registry entry | Read from merged registry at load time |

**For engine options that cross the DLL boundary, use `ENV_VAR`.** The engine
reads the env var (one line: `getenv_oflm("OFLM_OPEN_DECODE_MODE")`), and
`apply_settings()` setenvs it automatically. No glue code needed.

### Step 3: Engine reads the value

In the engine code, read the env var:

```cpp
std::string decode_mode = utils::getenv_oflm("OFLM_OPEN_DECODE_MODE");
if (decode_mode.empty()) decode_mode = "exact"; // default
```

### Step 4: Build and test

```bash
cmake --build --preset linux-default --target oflm
./build/src/oflm config gen-schema | python3 -m json.tool | grep decode_mode
./build/src/oflm config show
```

The new setting automatically appears in:
- `oflm config --help`
- `oflm config show`
- `oflm config set` (with validation)
- Interactive picker
- Shipped `model_list.json` schema (at package time)

### Field reference

| Field | Required | Description |
|-------|----------|-------------|
| `key` | Yes | Canonical name in settings.json and API |
| `cli_flag` | No | CLI flag (e.g. `--decode-mode`), empty if none |
| `env_var` | If ENV_VAR | Environment variable name |
| `description` | Yes | Human-readable description |
| `type` | Yes | `ENUM`, `INT`, `BOOL`, or `STRING` |
| `options` | If ENUM | Valid options with descriptions |
| `min_value`/`max_value` | If INT | Value range |
| `presets` | No | Quick-pick values for INT in picker |
| `default_value` | No | Fallback default (empty = from registry entry) |
| `scope` | Yes | `GLOBAL`, `PER_MODEL`, or `BOTH` |
| `reload_required` | Yes | `true` if changing requires model reload |
| `category` | Yes | Grouping: `performance`, `memory`, `generation` |
| `registry_key` | No | Maps to model_list.json entry key (e.g. `default_context_length`) |
| `delivery` | Yes | `DIRECT`, `ENV_VAR`, `API_FIELD`, or `REGISTRY` |
```

## Test

```bash
# Verify the file was updated
grep -n "Adding a new setting" src/create_new_model.md
wc -l src/create_new_model.md
```

## Success criteria

- [ ] New section "Adding a new setting" added to `src/create_new_model.md`
- [ ] Section explains: registry location, SettingDef fields, delivery mechanisms, engine integration
- [ ] Includes a complete code example
- [ ] Includes a field reference table
- [ ] Delivery mechanism table explains when to use each
- [ ] Mentions that everything (help, picker, schema) is auto-generated
