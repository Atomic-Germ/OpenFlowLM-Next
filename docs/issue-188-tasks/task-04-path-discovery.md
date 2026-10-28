# Task 04: Path Discovery Functions

**Phase:** 2 (Config Store)
**Depends on:** None
**Files to modify:** `src/common/utils.cpp`, `src/include/utils/utils.hpp`
**Estimated complexity:** Small (~60 lines)

## Context

The current `find_model_list()` in `src/common/utils.cpp:63-118` has a 7-step search that includes the user registry (`~/.config/oflm/model_list.json`) as step 4 — which **replaces** the shipped registry entirely. This task splits it into two functions:
- `find_shipped_registry()` — finds the shipped `model_list.json` (no user registry step)
- `find_user_settings()` — finds the user's `settings.json` overlay

Read parent plan **§Phase 2 (Path discovery)** in `docs/issue-188-implementation-plan.md`.

## What to do

1. Add `find_shipped_registry()` and `find_user_settings()` to `src/common/utils.cpp`
2. Declare them in `src/include/utils/utils.hpp`
3. Keep `find_model_list()` as a backward-compatible wrapper

## Code to write

### In `src/include/utils/utils.hpp`

Add these declarations (find a good spot near the existing `find_model_list` declaration):

```cpp
/// Find the SHIPPED model_list.json (no user registry step).
/// Search order:
///   1. OFLM_CONFIG_PATH (whole-registry override)
///   2. <exe_dir>/model_list.json
///   3. ./model_list.json
///   4. <exe_dir>/../share/oflm/model_list.json
///   5. CMAKE_INSTALL_PREFIX/share/oflm/model_list.json
std::string find_shipped_registry();

/// Find the user settings.json overlay file.
/// Search order:
///   1. OFLM_SETTINGS_PATH (for testing)
///   2. <models_dir>/settings.json  (sibling of models directory)
/// Returns empty string if not found (overlay is optional).
std::string find_user_settings();
```

### In `src/common/utils.cpp`

Add these functions (place them right after the existing `find_model_list()` function, around line 118):

```cpp
std::string find_shipped_registry() {
    std::string install_prefix = CMAKE_INSTALL_PREFIX;

    // 1. Check OFLM_CONFIG_PATH environment variable (whole-registry override)
    const std::string env_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!env_path.empty()) {
        if (std::filesystem::exists(env_path)) {
            std::cerr << "[OFLM]  Using custom model list path: " << env_path << std::endl;
            return env_path;
        }
    }

    // 2. Portable development-tree location (next to the executable, then CWD).
    std::string exe_dir = get_executable_directory();
    std::string exe_relative_path = exe_dir + "/model_list.json";
    if (std::filesystem::exists(exe_relative_path)) {
        return exe_relative_path;
    }
    if (std::filesystem::exists("model_list.json")) {
        return "model_list.json";
    }

    // NOTE: No user registry step here! The user registry is now an overlay
    // (settings.json), not a replacement. This is the key change from the
    // old find_model_list() which checked the user dir at this point.

    // 3. Relocatable installed bundle, independent of its original prefix.
    std::string bundle_path = exe_dir + "/../share/oflm/model_list.json";
    if (std::filesystem::exists(bundle_path)) {
        return bundle_path;
    }

    // 4. Legacy configured prefix.
    std::string installed_path = install_prefix + "/share/oflm/model_list.json";
    if (std::filesystem::exists(installed_path)) {
        return installed_path;
    }

    throw std::runtime_error("model_list.json not found. Reinstall OpenFlowLM, or set OFLM_CONFIG_PATH "
                             "if you are running from a non-standard location.");
}

std::string find_user_settings() {
    // 1. OFLM_SETTINGS_PATH (for testing)
    const std::string env_path = getenv_oflm("OFLM_SETTINGS_PATH");
    if (!env_path.empty()) {
        return env_path;  // Return even if it doesn't exist — caller handles missing file
    }

    // 2. <models_dir>/settings.json (sibling of models directory)
    std::string models_dir = get_models_directory();
    std::string settings_path = models_dir + "/settings.json";
    return settings_path;  // Return the path; caller checks if it exists
}
```

### Update `find_model_list()` (backward compat)

Replace the existing `find_model_list()` body (lines 63-118) with a call to `find_shipped_registry()`:

```cpp
std::string find_model_list() {
    // Backward-compatible wrapper. New code should use find_shipped_registry()
    // + find_user_settings() to get the two-file overlay behavior.
    // If OFLM_CONFIG_PATH is set, this returns the whole-registry override.
    // Otherwise, this returns the shipped registry (no user overlay).
    return find_shipped_registry();
}
```

**IMPORTANT:** Do NOT delete the old `find_model_list()` — other code may call it. Just replace its body.

Also keep the `user_oflm_directories()` and `legacy_flm_directories()` functions — they're still used by `find_user_settings()` indirectly (via `get_models_directory()`).

## Test

```bash
# Build
cmake --build --preset linux-default --target oflm 2>&1 | tail -10

# Verify find_shipped_registry finds the shipped registry
./build/src/oflm list 2>&1 | head -5

# Verify OFLM_SETTINGS_PATH is recognized (should not crash)
OFLM_SETTINGS_PATH=/tmp/test_settings.json ./build/src/oflm list 2>&1 | head -5
```

## Success criteria

- [ ] `find_shipped_registry()` declared in `utils.hpp` and defined in `utils.cpp`
- [ ] `find_user_settings()` declared in `utils.hpp` and defined in `utils.cpp`
- [ ] `find_model_list()` is now a wrapper that calls `find_shipped_registry()`
- [ ] Project compiles without errors
- [ ] `oflm list` still works (backward compat)
- [ ] `OFLM_SETTINGS_PATH` env var is recognized
- [ ] `find_shipped_registry()` does NOT check `~/.config/oflm/model_list.json` (no user registry step)
- [ ] `find_user_settings()` returns `<models_dir>/settings.json` when `OFLM_SETTINGS_PATH` is not set
