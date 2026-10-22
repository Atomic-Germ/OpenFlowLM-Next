# Task 21: Engine Integration — main.cpp + automodel.cpp

**Phase:** 8 (Integration with Engine Settings)
**Depends on:** Task 09, Task 14
**Files to modify:** `src/src/main.cpp`, `src/common/AutoModel/automodel.cpp`, `src/include/AutoModel/automodel.hpp`
**Estimated complexity:** Medium (~80 lines)

## Context

Wire the settings system into the engine startup flow. Replace the hardcoded `setenv("OFLM_OPEN_PREFILL_MODE")` at main.cpp:619-625 with the systematic `apply_settings()`. Pass resolved settings to `_shared_load_model`.

Read parent plan **§Phase 8** in `docs/issue-188-implementation-plan.md`.

## What to do

1. In `main.cpp`: load EffectiveConfig, resolve settings, apply settings (replaces hardcoded setenv)
2. In `automodel.hpp/cpp`: add `resolved_settings` parameter to `_shared_load_model`
3. In `automodel.cpp`: use resolved settings for ctx_len

## Code changes

### In `src/src/main.cpp`

Add includes at top:
```cpp
#include "config/config_store.hpp"
#include "config/settings_resolver.hpp"
```

Replace the hardcoded setenv (lines 619-625):
```cpp
// OLD (delete this):
// if (parsed_args.command == "serve" || parsed_args.command == "run" || ...) {
//     setenv("OFLM_OPEN_PREFILL_MODE", parsed_args.prefill_mode.c_str(), 1);
// }

// NEW:
if (parsed_args.command == "run" || parsed_args.command == "serve" ||
    parsed_args.command == "bench" || parsed_args.command == "bench-embed" ||
    parsed_args.command == "image") {
    // Load effective config (shipped + user settings, with migration)
    static oflm::config::EffectiveConfig effective_cfg;
    try {
        effective_cfg = oflm::config::load_effective_config();
    } catch (const std::exception& e) {
        std::cerr << "[OFLM] Warning: " << e.what() << std::endl;
        // Continue with empty config — will use defaults
    }

    // Resolve settings (applies precedence: shipped < user < env < CLI)
    std::string tag = parsed_args.model_tag;
    if (tag == "model-faker") tag = ""; // serve default sentinel
    auto resolved = oflm::config::resolve_settings(effective_cfg, tag, parsed_args);

    // Apply settings (setenv for ENV_VAR delivery — replaces hardcoded setenv)
    oflm::config::apply_settings(resolved);

    // Store for later use by Runner/RestHandler
    // (Pass via a global or thread_local, or add to program_args_t)
    // Simplest: store in a static that Runner/RestHandler can access
    // Or: add resolved_settings to program_args_t
}
```

Actually, the cleanest approach: add `nlohmann::json resolved_settings` to `program_args_t` and populate it in main.cpp before dispatching to Runner/RestHandler.

In `src/include/program_args.hpp`:
```cpp
#include <nlohmann/json.hpp>
// ...
nlohmann::json resolved_settings;  // Populated by main.cpp before dispatch
oflm::config::EffectiveConfig effective_config;  // Also populated by main.cpp
```

Wait — `EffectiveConfig` is in `oflm::config` namespace and includes `config_store.hpp`. Adding it to `program_args.hpp` creates a dependency. Better to use a global or pass separately.

Simplest approach: use a global pointer.

In `src/config/settings_resolver.hpp`:
```cpp
/// Global effective config — set by main.cpp, read by Runner/RestHandler.
/// Valid only after load_effective_config() has been called.
extern EffectiveConfig* g_effective_config;

/// Global resolved settings — set by main.cpp after resolve_settings().
extern nlohmann::json* g_resolved_settings;
```

In `src/config/settings_resolver.cpp`:
```cpp
EffectiveConfig* g_effective_config = nullptr;
nlohmann::json* g_resolved_settings = nullptr;
```

In `main.cpp`:
```cpp
// After resolving:
static oflm::config::EffectiveConfig cfg_storage;
static nlohmann::json resolved_storage;
cfg_storage = oflm::config::load_effective_config();
resolved_storage = oflm::config::resolve_settings(cfg_storage, tag, parsed_args);
oflm::config::apply_settings(resolved_storage);
oflm::config::g_effective_config = &cfg_storage;
oflm::config::g_resolved_settings = &resolved_storage;
```

### In `src/include/AutoModel/automodel.hpp`

Find `_shared_load_model` declaration and add a parameter:

```cpp
// OLD:
// void _shared_load_model(const std::string& model_path, const nlohmann::json& model_info, int default_context_length, bool preemption);

// NEW:
void _shared_load_model(const std::string& model_path, const nlohmann::json& model_info,
                        int default_context_length, bool preemption,
                        const nlohmann::json& resolved_settings = nlohmann::json::object());
```

### In `src/common/AutoModel/automodel.cpp`

Update `_shared_load_model` implementation (around line 157-188):

```cpp
void AutoModel::_shared_load_model(const std::string& model_path, const nlohmann::json& model_info,
                                   int default_context_length, bool preemption,
                                   const nlohmann::json& resolved_settings) {
    // ... existing code ...

    // Context length: resolved settings override CLI override registry
    if (resolved_settings.contains("ctx_len")) {
        this->MAX_L = resolved_settings["ctx_len"].get<int>();
    } else if (default_context_length != -1) {
        this->MAX_L = default_context_length;
    } else {
        this->MAX_L = model_info["default_context_length"].get<int>();
    }

    // ... rest of existing code ...
}
```

### Update callers of `_shared_load_model`

Find all callers (in modeling_*.cpp files) and pass `g_resolved_settings`:

```cpp
// In each modeling_*.cpp load_model override:
_shared_load_model(model_path, model_info, ctx_length, preemption,
                   oflm::config::g_resolved_settings ? *oflm::config::g_resolved_settings : nlohmann::json::object());
```

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -20

# Verify prefill_mode env var is set via apply_settings (not hardcoded setenv)
OFLM_SETTINGS_PATH=/tmp/test_settings.json ./build/src/oflm run qwen3.8:27b --help 2>&1 | head -5

# Set a user preference and verify it's applied
echo '{"version":1,"settings":{"qwen3.8":{"27b":{"prefill_mode":"lean"}}}}' > /tmp/test_settings.json
OFLM_SETTINGS_PATH=/tmp/test_settings.json ./build/src/oflm run qwen3.8:27b 2>&1 | head -10 &
PID=$!
sleep 2
# Check if OFLM_OPEN_PREFILL_MODE was set (indirectly — check process env)
kill $PID 2>/dev/null
```

## Success criteria

- [ ] `main.cpp` loads EffectiveConfig and calls `apply_settings()` (replaces hardcoded setenv at :619-625)
- [ ] `_shared_load_model` accepts `resolved_settings` parameter
- [ ] `_shared_load_model` uses `resolved_settings["ctx_len"]` for MAX_L
- [ ] All callers of `_shared_load_model` pass resolved settings
- [ ] Global `g_effective_config` and `g_resolved_settings` pointers set by main.cpp
- [ ] Project compiles without errors
- [ ] `oflm run` still works (backward compat)
