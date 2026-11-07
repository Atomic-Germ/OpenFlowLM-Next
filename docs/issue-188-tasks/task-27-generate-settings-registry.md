# Task 27: Auto-Generate Settings Registry from Source Code

**Phase:** Enforcement (systematic generation)
**Depends on:** Tasks 01-02 (registry), 11 (CLI options), 12 (gen-schema)
**Replaces/augments:** Task 26 (lint script becomes "verify generated files are up to date")
**Files to create:**
- `utilities/scan-settings.py` — scanner (extracts env vars, CLI flags, descriptions from source)
- `utilities/generate-registry.py` — generator (merges scanner output + manual overrides → C++ header)
- `src/config/settings.json` — source of truth (human-editable, manual + auto fields)
- `src/config/settings_registry_generated.hpp` — generated C++ header (committed to repo)
**Files to modify:**
- `src/config/settings_registry.cpp` — include generated header instead of hardcoded data
- `src/CMakeLists.txt` — add `generate-settings` target
- `.github/workflows/cmake-build-test.yml` — verify generated files are up to date
**Estimated complexity:** Large (~400 lines Python + ~50 lines CMake/YAML)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Source code                                                 │
│  - vm_args.hpp: ("--prefill-mode", po::value<...>, "desc")  │
│  - engine.cpp: getenv_oflm("OFLM_OPEN_PREFILL_MODE")        │
│  - main.cpp: setenv("OFLM_OPEN_PREFILL_MODE", ...)           │
└────────────────────┬────────────────────────────────────────┘
                     │ scan-settings.py
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Auto-extracted data (stdout JSON)                           │
│  {                                                           │
│    "cli_flags": {                                            │
│      "--prefill-mode": {                                     │
│        "description": "Prefill mode: fast or lean",          │
│        "type_hint": "string",                                │
│        "field": "prefill_mode",                              │
│        "default": "fast"                                     │
│      }                                                       │
│    },                                                        │
│    "env_vars": {                                             │
│      "OFLM_OPEN_PREFILL_MODE": {                             │
│        "read_locations": ["src/open_qwen36/engine.cpp:82"],  │
│        "set_locations": ["src/src/main.cpp:619"]            │
│      }                                                       │
│    }                                                         │
│  }                                                           │
└────────────────────┬────────────────────────────────────────┘
                     │ merge with manual overrides
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  src/config/settings.json (SOURCE OF TRUTH, human-editable)  │
│  {                                                           │
│    "settings": [                                             │
│      {                                                       │
│        "key": "prefill_mode",          ← derived from flag   │
│        "cli_flag": "--prefill-mode",   ← AUTO from scanner   │
│        "env_var": "OFLM_OPEN_PREFILL_MODE", ← AUTO           │
│        "description": "...",           ← AUTO from help text │
│        "type": "enum",                 ← MANUAL              │
│        "options": ["fast", "lean"],    ← MANUAL              │
│        "scope": "per_model",           ← MANUAL              │
│        "delivery": "env_var",          ← MANUAL (critical!)  │
│        "category": "performance",      ← MANUAL              │
│        "reload_required": true,        ← MANUAL              │
│        "default_value": "fast"         ← AUTO if available   │
│      }                                                       │
│    ]                                                         │
│  }                                                           │
└────────────────────┬────────────────────────────────────────┘
                     │ generate-registry.py
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  src/config/settings_registry_generated.hpp (COMMITTED)      │
│  #pragma once                                                │
│  #include "settings_registry.hpp"                            │
│  namespace oflm::config {                                    │
│  inline const std::vector<SettingDef>& build_registry() {    │
│    static const std::vector<SettingDef> registry = {         │
│      SettingDef{ .key = "prefill_mode", ... },               │
│      ...                                                     │
│    };                                                        │
│    return registry;                                          │
│  }                                                           │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow: Adding a New Setting

1. **Add CLI flag** to `vm_args.hpp`:
   ```cpp
   ("--decode-mode", po::value<std::string>(&parsed_args.decode_mode)->default_value("exact"),
    "Decode mode: exact (bit-exact) or fast (approximate)")
   ```

2. **Use env var** in engine code:
   ```cpp
   std::string mode = utils::getenv_oflm("OFLM_OPEN_DECODE_MODE");
   ```

3. **Run scanner** to update `settings.json`:
   ```bash
   cmake --build --preset linux-default --target generate-settings
   ```
   Scanner finds `--decode-mode` and `OFLM_OPEN_DECODE_MODE`, adds a stub entry:
   ```json
   {
     "key": "decode_mode",
     "cli_flag": "--decode-mode",
     "env_var": "OFLM_OPEN_DECODE_MODE",
     "description": "Decode mode: exact (bit-exact) or fast (approximate)",
     "_auto_type_hint": "string",
     "_auto_default": "exact"
   }
   ```

4. **Fill in manual fields** in `settings.json`:
   ```json
   {
     "key": "decode_mode",
     "cli_flag": "--decode-mode",
     "env_var": "OFLM_OPEN_DECODE_MODE",
     "description": "Decode mode: exact (bit-exact) or fast (approximate)",
     "type": "enum",
     "options": [
       {"value": "exact", "description": "Bit-exact, slower"},
       {"value": "fast", "description": "Faster, approximate"}
     ],
     "scope": "per_model",
     "delivery": "env_var",
     "category": "performance",
     "reload_required": true,
     "default_value": "exact"
   }
   ```

5. **Regenerate** the C++ header:
   ```bash
   cmake --build --preset linux-default --target generate-settings
   ```

6. **Build and test**:
   ```bash
   cmake --build --preset linux-default --target oflm
   ctest --preset linux-default -R settings
   ```

7. **Commit** `settings.json` + `settings_registry_generated.hpp`

---

## CI Enforcement

```yaml
# .github/workflows/cmake-build-test.yml
- name: Verify settings registry is up to date
  run: |
    cmake --build --preset linux-default --target generate-settings
    git diff --exit-code src/config/settings.json src/config/settings_registry_generated.hpp
```

If someone adds a CLI flag or env var but doesn't run the generator, CI fails with a diff showing what's missing.

---

## What to Implement

### 1. `utilities/scan-settings.py`

Scans source code and outputs JSON with auto-extracted data.

```python
#!/usr/bin/env python3
"""Scan source code for CLI flags and env vars. Outputs JSON to stdout.

Usage:
    scan-settings.py --source-dir src
"""
import argparse, json, re, sys
from pathlib import Path

def scan_cli_flags(source_dir):
    """Extract CLI flags from vm_args.hpp."""
    vm_args = Path(source_dir) / "include" / "utils" / "vm_args.hpp"
    if not vm_args.exists():
        return {}
    content = vm_args.read_text()
    flags = {}
    # Pattern: ("--flag-name", po::value<T>(&parsed_args.field)->default_value(X), "description")
    # or: ("--flag-name", po::bool_switch(&parsed_args.field), "description")
    # or: ("--flag-name", po::value<T>(&parsed_args.field), "description")
    pattern = re.compile(
        r'\("(--[a-z0-9-]+)"\s*,\s*'
        r'po::(value<([^>]+)>|bool_switch)\(&parsed_args\.(\w+)\)'
        r'(?:->default_value\(([^)]+)\))?'
        r'\s*,\s*"([^"]*)"\s*\)'
    )
    for m in pattern.finditer(content):
        flag, boost_type, value_type, field, default, desc = m.groups()
        type_hint = "bool" if boost_type == "bool_switch" else (
            "int" if value_type and "int" in value_type else "string"
        )
        key = flag.lstrip("-").replace("-", "_")
        flags[flag] = {
            "key": key,
            "description": desc,
            "type_hint": type_hint,
            "field": field,
            "default": default.strip('"') if default else None,
        }
    return flags

def scan_env_vars(source_dir):
    """Extract OFLM_* env var usage from source files."""
    src = Path(source_dir)
    env_vars = {}
    read_pattern = re.compile(r'getenv_oflm\(\s*"(OFLM_[A-Z0-9_]+)"\s*\)')
    set_pattern = re.compile(r'setenv\(\s*"(OFLM_[A-Z0-9_]+)"')
    getenv_pattern = re.compile(r'std::getenv\(\s*"(OFLM_[A-Z0-9_]+)"\s*\)')

    for pattern, kind in [(read_pattern, "read"), (set_pattern, "set"), (getenv_pattern, "read")]:
        for f in list(src.rglob("*.cpp")) + list(src.rglob("*.hpp")):
            if "test" in str(f) or "build" in str(f) or "config" in str(f):
                continue  # Skip tests, build dir, and config dir (avoid self-reference)
            try:
                content = f.read_text(errors="ignore")
            except:
                continue
            for m in pattern.finditer(content):
                var = m.group(1)
                line_num = content[:m.start()].count("\n") + 1
                if var not in env_vars:
                    env_vars[var] = {"read_locations": [], "set_locations": []}
                if kind == "read":
                    env_vars[var]["read_locations"].append(f"{f.relative_to(src)}:{line_num}")
                else:
                    env_vars[var]["set_locations"].append(f"{f.relative_to(src)}:{line_num}")
    return env_vars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="src")
    args = parser.parse_args()

    result = {
        "cli_flags": scan_cli_flags(args.source_dir),
        "env_vars": scan_env_vars(args.source_dir),
    }
    json.dump(result, sys.stdout, indent=2)

if __name__ == "__main__":
    main()
```

### 2. `src/config/settings.json`

The source of truth. Human-editable. Auto fields are updated by the scanner, manual fields by the developer.

```json
{
  "version": 1,
  "settings": [
    {
      "key": "prefill_mode",
      "cli_flag": "--prefill-mode",
      "env_var": "OFLM_OPEN_PREFILL_MODE",
      "description": "Memory/speed trade-off for prefill",
      "type": "enum",
      "options": [
        {"value": "fast", "description": "Default prefill, higher memory"},
        {"value": "lean", "description": "Lower memory, ~15% slower"}
      ],
      "scope": "per_model",
      "delivery": "env_var",
      "category": "performance",
      "reload_required": true,
      "default_value": "fast"
    },
    {
      "key": "pmode",
      "cli_flag": "--pmode",
      "description": "NPU power mode",
      "type": "enum",
      "options": [
        {"value": "default", "description": "Let the driver decide"},
        {"value": "powersaver", "description": "Lower power, reduced performance"},
        {"value": "balanced", "description": "Balance of power and performance"},
        {"value": "performance", "description": "Maximum performance (oflm default)"},
        {"value": "turbo", "description": "Maximum clocks, highest power"}
      ],
      "scope": "global",
      "delivery": "direct",
      "category": "performance",
      "reload_required": false,
      "default_value": "performance"
    },
    {
      "key": "ctx_len",
      "cli_flag": "--ctx-len",
      "description": "Context window length (overrides model default)",
      "type": "int",
      "min_value": 512,
      "presets": [2048, 4096, 8192, 16384, 32768],
      "scope": "per_model",
      "delivery": "direct",
      "category": "memory",
      "reload_required": true,
      "registry_key": "default_context_length"
    },
    {
      "key": "prefill_chunk_len",
      "cli_flag": "--prefill-chunk-len",
      "description": "Maximum prefill chunk length (overrides model default)",
      "type": "int",
      "min_value": 512,
      "presets": [1024, 2048, 4096, 8192],
      "scope": "per_model",
      "delivery": "direct",
      "category": "memory",
      "reload_required": true,
      "registry_key": "max_prefill_len"
    },
    {
      "key": "think",
      "description": "Enable reasoning/thinking mode by default",
      "type": "bool",
      "scope": "per_model",
      "delivery": "api_field",
      "category": "generation",
      "reload_required": false,
      "registry_key": "details.think"
    }
  ]
}
```

### 3. `utilities/generate-registry.py`

Merges scanner output + settings.json → generates `settings_registry_generated.hpp`.

```python
#!/usr/bin/env python3
"""Generate settings_registry_generated.hpp from settings.json + code scan.

Usage:
    generate-registry.py --settings src/config/settings.json
                         --scan-output /tmp/scan.json
                         --output src/config/settings_registry_generated.hpp
    
    If --scan-output is not provided, runs scan-settings.py automatically.
"""
import argparse, json, subprocess, sys
from pathlib import Path

# Required manual fields (fail if missing)
REQUIRED_MANUAL = ["type", "scope", "delivery", "category"]

def generate_cpp_header(settings):
    """Generate C++ header content from settings list."""
    lines = []
    lines.append("/// \\file settings_registry_generated.hpp")
    lines.append("/// \\brief AUTO-GENERATED settings registry — do not edit manually.")
    lines.append("/// \\note Generated by utilities/generate-registry.py from settings.json + code scan.")
    lines.append("/// \\note To add a setting: add CLI flag/env var to code, run generate-settings target,")
    lines.append("///       fill in manual fields in settings.json, regenerate.")
    lines.append("#pragma once")
    lines.append("")
    lines.append('#include "settings_registry.hpp"')
    lines.append("")
    lines.append("namespace oflm::config {")
    lines.append("")
    lines.append("inline const std::vector<SettingDef>& build_registry() {")
    lines.append("    static const std::vector<SettingDef> registry = {")

    for s in settings:
        lines.append("        SettingDef{")
        lines.append(f'            .key = "{s["key"]}",')
        lines.append(f'            .cli_flag = "{s.get("cli_flag", "")}",')
        lines.append(f'            .env_var = "{s.get("env_var", "")}",')
        # Escape description
        desc = s.get("description", "").replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'            .description = "{desc}",')
        # Type
        type_map = {"enum": "SettingType::ENUM", "int": "SettingType::INT",
                    "bool": "SettingType::BOOL", "string": "SettingType::STRING"}
        lines.append(f'            .type = {type_map.get(s["type"], "SettingType::STRING")},')
        # Options
        if s.get("options"):
            lines.append("            .options = {")
            for opt in s["options"]:
                opt_desc = opt.get("description", "").replace('\\', '\\\\').replace('"', '\\"')
                lines.append(f'                {{"{opt["value"]}", "{opt_desc}"}},')
            lines.append("            },")
        else:
            lines.append("            .options = {},")
        # Min/max
        if s.get("min_value") is not None:
            lines.append(f'            .min_value = {s["min_value"]},')
        else:
            lines.append("            .min_value = std::nullopt,")
        if s.get("max_value") is not None:
            lines.append(f'            .max_value = {s["max_value"]},')
        else:
            lines.append("            .max_value = std::nullopt,")
        # Presets
        if s.get("presets"):
            presets = ", ".join(str(p) for p in s["presets"])
            lines.append(f"            .presets = {{{presets}}},")
        else:
            lines.append("            .presets = {},")
        # Default
        lines.append(f'            .default_value = "{s.get("default_value", "")}",')
        # Scope
        scope_map = {"global": "SettingScope::GLOBAL", "per_model": "SettingScope::PER_MODEL",
                     "both": "SettingScope::BOTH"}
        lines.append(f'            .scope = {scope_map.get(s["scope"], "SettingScope::GLOBAL")},')
        # Reload required
        lines.append(f'            .reload_required = {"true" if s.get("reload_required") else "false"},')
        # Category
        lines.append(f'            .category = "{s.get("category", "general")}",')
        # Registry key
        lines.append(f'            .registry_key = "{s.get("registry_key", "")}",')
        # Delivery
        delivery_map = {"direct": "DeliveryMechanism::DIRECT", "env_var": "DeliveryMechanism::ENV_VAR",
                        "api_field": "DeliveryMechanism::API_FIELD", "registry": "DeliveryMechanism::REGISTRY"}
        lines.append(f'            .delivery = {delivery_map.get(s["delivery"], "DeliveryMechanism::DIRECT")},')
        lines.append("        },")

    lines.append("    };")
    lines.append("    return registry;")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace oflm::config")
    lines.append("")
    return "\n".join(lines)

def validate_settings(settings):
    """Check that all settings have required manual fields."""
    errors = []
    for s in settings:
        key = s.get("key", "?")
        for field in REQUIRED_MANUAL:
            if field not in s:
                errors.append(f"Setting '{key}' missing required manual field: {field}")
        if s.get("type") == "enum" and not s.get("options"):
            errors.append(f"Setting '{key}' is enum but has no options")
        if s.get("type") == "int" and s.get("min_value") is None:
            errors.append(f"Setting '{key}' is int but has no min_value")
    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="src/config/settings.json")
    parser.add_argument("--scan-output", default=None, help="Pre-existing scan output (runs scanner if not provided)")
    parser.add_argument("--output", default="src/config/settings_registry_generated.hpp")
    parser.add_argument("--source-dir", default="src")
    args = parser.parse_args()

    # Load settings.json
    with open(args.settings) as f:
        data = json.load(f)
    settings = data.get("settings", [])

    # Validate
    errors = validate_settings(settings)
    if errors:
        print("ERROR: settings.json validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    # Run scanner if no pre-existing output
    if args.scan_output:
        with open(args.scan_output) as f:
            scan = json.load(f)
    else:
        result = subprocess.run(
            [sys.executable, "utilities/scan-settings.py", "--source-dir", args.source_dir],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"ERROR: scanner failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        scan = json.loads(result.stdout)

    # Cross-check: warn if settings.json has entries not found in code
    scan_flags = set(scan.get("cli_flags", {}).keys())
    scan_envs = set(scan.get("env_vars", {}).keys())
    for s in settings:
        if s.get("cli_flag") and s["cli_flag"] not in scan_flags:
            print(f"WARNING: {s['cli_flag']} in settings.json but not found in vm_args.hpp")
        if s.get("env_var") and s["env_var"] not in scan_envs:
            print(f"WARNING: {s['env_var']} in settings.json but not found in source code")

    # Generate C++ header
    cpp = generate_cpp_header(settings)
    with open(args.output, "w") as f:
        f.write(cpp)
    print(f"Generated {args.output} ({len(settings)} settings)")

if __name__ == "__main__":
    main()
```

### 4. Update `src/config/settings_registry.cpp`

Replace the hardcoded `build_registry()` with an include of the generated header:

```cpp
// At the top, replace the build_registry() function definition with:
#include "settings_registry_generated.hpp"
// The generated header provides build_registry() — delete the old one.
```

Specifically: delete the `build_registry()` function and the `namespace { ... }` block that contains it, and add `#include "settings_registry_generated.hpp"` at the top of the file (inside or outside the namespace — the generated header handles its own namespace).

### 5. Add CMake target in `src/CMakeLists.txt`

```cmake
# Generate settings registry from source code scan + settings.json
add_custom_target(generate-settings
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../utilities/scan-settings.py
            --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
            > ${CMAKE_BINARY_DIR}/scan-settings.json
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../utilities/generate-registry.py
            --settings ${CMAKE_CURRENT_SOURCE_DIR}/config/settings.json
            --scan-output ${CMAKE_BINARY_DIR}/scan-settings.json
            --output ${CMAKE_CURRENT_SOURCE_DIR}/config/settings_registry_generated.hpp
            --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating settings registry from source code"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
)
```

### 6. Update CI in `.github/workflows/cmake-build-test.yml`

```yaml
      - name: Verify settings registry is up to date
        run: |
          cmake --build --preset linux-default --target generate-settings
          git diff --exit-code src/config/settings.json src/config/settings_registry_generated.hpp
```

## Test

```bash
# Build oflm
cmake --build --preset linux-default --target oflm

# Generate registry
cmake --build --preset linux-default --target generate-settings

# Verify generated header compiles
cmake --build --preset linux-default --target oflm

# Run tests
ctest --preset linux-default -R settings --output-on-failure

# Verify CI check works: add a fake flag to vm_args.hpp, run generate-settings,
# check that git diff shows the new stub entry, then revert

# Verify validation: remove a required field from settings.json, run generate-settings,
# check that it fails with a helpful error
```

## Success criteria

- [ ] `utilities/scan-settings.py` created — extracts CLI flags and env vars from source
- [ ] `src/config/settings.json` created — source of truth with 5 settings
- [ ] `utilities/generate-registry.py` created — merges scan + settings.json → C++ header
- [ ] `src/config/settings_registry_generated.hpp` created (committed to repo)
- [ ] `src/config/settings_registry.cpp` updated to include generated header (old build_registry() removed)
- [ ] CMake target `generate-settings` added
- [ ] `cmake --build --preset linux-default --target generate-settings` works
- [ ] `cmake --build --preset linux-default --target oflm` compiles with generated header
- [ ] All existing tests pass (settings_registry_test, settings_resolver_test, config_store_test)
- [ ] CI step added to verify generated files are up to date
- [ ] Validation catches missing manual fields (type, scope, delivery, category)
- [ ] Scanner warns if settings.json has entries not found in code
- [ ] Generated header has auto-generation warning comment ("do not edit manually")
