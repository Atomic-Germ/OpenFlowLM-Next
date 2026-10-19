# Issue #188 Implementation Plan: Per-Model and User Settings

**Branch:** `issue-188-per-model-user-settings`
**Date:** 2026-10-19
**Status:** v3 — incorporates issue #188 discussion updates (Cyronius RFC summary, abondis conflict-detection suggestion)

---

## 0. What Changed

### v2 (Critical Review Response)

| # | Issue | Fix |
|---|-------|-----|
| C1 | Migration dropped modified entries (data loss) | Three-way migration: preserve user modifications as overrides |
| C2 | `OFLM_CONFIG_PATH` semantics change broke hermetic testing | Keep as whole-registry override; add `OFLM_SETTINGS_PATH` for overlay |
| C3 | `config` before Boost parser → inconsistent flags | `config` handled inside Boost parser like other C++ commands |
| C4 | Hardcoded `ResolvedSettings` struct defeated registry | Use `nlohmann::json`, iterate registry for resolution |
| C5 | `settings`/`models` sections ambiguous in settings.json | Split into separate `settings` and `models` sections |
| H1 | ANSI TUI fragile (Windows <10, TERM=dumb) | Three-tier UI: numbered menu → arrow picker → full TUI |
| H2 | Schema mismatch silently ignored settings on downgrade | List specific ignored keys in warning |
| H3 | Corrupted settings.json → C++ exception | Friendly error with fix instructions, continue with shipped defaults |
| H4 | Concurrent writes → lost updates | File locking (flock/LockFileEx) around load-modify-save |
| H5 | Strict mode blocked forward-compatibility | Warn on unknown keys, allow with confirmation |
| **NEW** | Engine option unreachable from binary (env-var bridge hack) | `DeliveryMechanism` in registry — every setting declares how it reaches its consumer |

### v3 (Issue #188 Discussion Updates, Oct 18-19)

| # | Source | Update | Response |
|---|--------|--------|----------|
| D1 | abondis (Oct 19) | "If there is a conflict with a user setting it can warn and offer to update the users config?" | **Added: Shipped-default change detection** (§2.6). When a release changes a shipped default that the user has overridden, warn on next load and offer `oflm config review` to reconcile. |
| D2 | Cyronius (Oct 18) | RFC: is there a better way? Is the risk acceptable? Is expanding model_list.json appropriate? | **RFC response** (§9). Two-file overlay is the simplest correct design; alternatives (conf.d, single-file, database) add complexity without solving real problems. Risk of conflicting defaults is mitigated by D1. Expanding model_list.json is a natural extension of its existing role. |

---

## 1. Executive Summary

Implement a two-file configuration system:

| File | Owner | Overwritten on release? | Purpose |
|------|-------|------------------------|---------|
| `model_list.json` (shipped) | Project | Yes | Model registry + app-wide defaults + settings schema |
| `~/.config/oflm/settings.json` | User | **Never** | User-added models + per-model preferences + app-wide preferences |

**Precedence (lowest to highest):**
1. Shipped defaults (in `model_list.json`)
2. User `settings.json`
3. `OFLM_*` environment variables (for settings that have them)
4. CLI flags (`--ctx-len`, `--pmode`, etc.)
5. Per-request API fields (serve mode only)

**Single source of truth:** A C++ settings registry (`src/config/settings_registry.{hpp,cpp}`) defines every user-facing setting **and how it reaches the code that consumes it**. From it we generate:
- The `settings_schema` block injected into shipped `model_list.json` at package time
- `oflm config --help` text
- Interactive picker options
- Validation rules
- Delivery instructions (direct / env var / API field)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  src/config/settings_registry.{hpp,cpp}                         │
│  THE source of truth. Every user-facing setting defined once:   │
│  key, type, options, description, CLI flag, env var, scope,     │
│  default, AND delivery mechanism (how it reaches the consumer). │
└────────────────────┬────────────────────────────────────────────┘
                     │ generates / drives
    ┌────────────────┼────────────┬──────────────┬─────────────┐
    ▼                ▼            ▼              ▼             ▼
┌─────────┐   ┌──────────┐  ┌──────────┐  ┌────────────┐ ┌──────────┐
│ Schema  │   │  --help  │  │Interactive│  │ Validation │ │ Delivery │
│  JSON   │   │   text   │  │  picker   │  │  + merge   │ │ (env var │
│(shipped │   │(oflm     │  │           │  │            │ │  bridge, │
│  model_ │   │  config) │  │           │  │            │ │  direct, │
│  list)  │   │          │  │           │  │            │ │  API)    │
└─────────┘   └──────────┘  └──────────┘  └────────────┘ └──────────┘
```

### 2.1 Configuration Reachability (NEW — addresses "engine option unreachable from binary")

**The problem this solves:** The codebase has an established hack — `OFLM_OPEN_PREFILL_MODE` is `setenv()`'d by `main.cpp:619-625` because `LM_Config` (a fixed C struct passed across the `OFLM_DLL` boundary) can't carry new fields without an ABI break. When a branch adds an engine option, the oflm binary literally cannot pass it without this env-var bridge. This happened with `prefill_mode`; it will happen again.

**The fix:** Every setting in the registry declares its **delivery mechanism**. The application layer reads this and delivers the value appropriately. No more ad-hoc `setenv()` bridges discovered at integration time.

```cpp
enum class DeliveryMechanism {
    DIRECT,     // Passed via function parameter / struct (same process, no DLL boundary)
    ENV_VAR,    // setenv() bridge — for values that must cross the OFLM_DLL boundary
    API_FIELD,  // Per-request override via REST API (serve mode)
    REGISTRY,   // Read from model registry entry at load time (existing pattern)
};

struct SettingDef {
    // ... identity ...
    std::string key;
    std::string cli_flag;
    std::string env_var;          // For ENV_VAR delivery; also the read-side name
    std::string description;
    SettingType type;
    // ... constraints ...
    std::vector<SettingOption> options;
    std::optional<long> min_value, max_value;
    std::vector<long> presets;
    std::string default_value;
    SettingScope scope;
    bool reload_required;
    std::string category;
    std::string registry_key;     // Maps to model_list.json entry key (e.g. "default_context_length")

    // === REACHABILITY ===
    DeliveryMechanism delivery;
    // For ENV_VAR: the consumer reads this env var (same as env_var field)
    // For DIRECT: the consumer function takes this as a parameter
    // For API_FIELD: the REST handler reads this field from the request JSON
    // For REGISTRY: the value lives in the model registry entry under registry_key
};
```

**How delivery works at runtime:**

```
resolve_settings() produces nlohmann::json of effective values
         │
         ▼
apply_settings() iterates the registry:
  for each setting with an effective value:
    switch (def.delivery):
      DIRECT:    pass to consumer function parameter
      ENV_VAR:   setenv(def.env_var, value)   ← the prefill_mode pattern, now systematic
      API_FIELD: store in per-request defaults (serve mode)
      REGISTRY:  already in the merged registry entry, consumer reads it
```

**When adding a new engine option (e.g. #184's decode_mode):**
1. Add to registry with `delivery: ENV_VAR` and `env_var: "OFLM_OPEN_DECODE_MODE"`
2. Engine reads `getenv_oflm("OFLM_OPEN_DECODE_MODE")` — one line
3. `apply_settings()` automatically setenv()s it — zero glue code
4. CLI flag, user settings, picker, help all work automatically

**Long-term (not this PR):** Replace `LM_Config` struct with a JSON blob passed across the DLL boundary. Then `DIRECT` becomes possible for everything. The registry's `delivery` field makes this migration mechanical: change `ENV_VAR` → `DIRECT` one setting at a time.

### 2.2 Settings Registry

```cpp
// src/config/settings_registry.hpp
#pragma once
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace oflm::config {

enum class SettingType { ENUM, INT, BOOL, STRING };
enum class SettingScope { GLOBAL, PER_MODEL, BOTH };
enum class DeliveryMechanism { DIRECT, ENV_VAR, API_FIELD, REGISTRY };

struct SettingOption {
    std::string value;
    std::string description;
};

struct SettingDef {
    std::string key;
    std::string cli_flag;
    std::string env_var;
    std::string description;
    SettingType type;
    std::vector<SettingOption> options;
    std::optional<long> min_value;
    std::optional<long> max_value;
    std::vector<long> presets;
    std::string default_value;
    SettingScope scope;
    bool reload_required;
    std::string category;
    std::string registry_key;
    DeliveryMechanism delivery;
};

const std::vector<SettingDef>& settings_registry();
std::optional<SettingDef> find_setting(const std::string& key);
std::vector<SettingDef> settings_for_scope(SettingScope scope);
nlohmann::json generate_schema_json();
std::vector<std::string> generate_help_lines();

// Compute the defaults_version hash from a shipped registry.
// Hashes: top-level "defaults" block + per-model default values from settings registry entries.
std::string compute_defaults_version(const nlohmann::json& shipped_registry);

} // namespace oflm::config
```

### 2.3 Initial Settings in the Registry

| Key | Type | Options/Range | Default | Scope | CLI Flag | Env Var | Delivery | Registry Key | Reload? |
|-----|------|---------------|---------|-------|----------|---------|----------|--------------|---------|
| `prefill_mode` | ENUM | `fast`, `lean` | `fast` | PER_MODEL | `--prefill-mode` | `OFLM_OPEN_PREFILL_MODE` | **ENV_VAR** | — | Yes |
| `pmode` | ENUM | `default`, `powersaver`, `balanced`, `performance`, `turbo` | `performance` | GLOBAL | `--pmode` | — | **DIRECT** (xrt-smi) | — | No |
| `ctx_len` | INT | min 512 | (from entry) | PER_MODEL | `--ctx-len` | — | **DIRECT** | `default_context_length` | Yes |
| `prefill_chunk_len` | INT | min 512 | (from entry) | PER_MODEL | `--prefill-chunk-len` | — | **DIRECT** | `max_prefill_len` | Yes |
| `think` | BOOL | true/false | (from entry) | PER_MODEL | — | — | **API_FIELD** | `details.think` | No |

Future settings (from #184):

| Key | Type | Options | Default | Scope | Delivery | Env Var | Reload? |
|-----|------|---------|---------|-------|----------|---------|---------|
| `decode_mode` | ENUM | `exact`, `fast` | `exact` | PER_MODEL | **ENV_VAR** | `OFLM_OPEN_DECODE_MODE` | Yes |
| `decode_block_size` | INT | min 1 | 1 | PER_MODEL | **ENV_VAR** | `OFLM_OPEN_DECODE_BLOCK` | Yes |

### 2.4 User Settings File Structure

`~/.config/oflm/settings.json`:

```json
{
  "version": 1,
  "defaults_version": "a3f2b8c1",
  "defaults": {
    "pmode": "balanced"
  },
  "settings": {
    "qwen3.8": {
      "27b": {
        "prefill_mode": "lean",
        "ctx_len": 16384
      }
    }
  },
  "models": {
    "my-finetune": {
      "3b": {
        "name": "my-finetune-3b",
        "url": "https://huggingface.co/user/my-finetune",
        "files": ["model.q4nx", "tokenizer.json"],
        "default_context_length": 8192,
        "max_prefill_len": 4096,
        "details": {
          "family": "qwen3",
          "format": "NPU2",
          "think": true,
          "parameter_size": "3B",
          "quantization_level": "q4nx"
        }
      }
    }
  }
}
```

**`defaults_version`** — hash of the shipped defaults the user's settings were last reconciled against. Used for shipped-default change detection (§2.6). Missing = first run with this feature, record current version silently.

**Two distinct sections — no ambiguity:**
- `settings.<type>.<size>` — sparse preference overlays, deep-merged onto shipped registry entries
- `models.<type>.<size>` — complete user-added model entries (written by `oflm add`), added to registry if not in shipped

**Deep-merge semantics:**
- `settings` overlays merge **recursively** onto `shipped.models.<type>.<size>`
- `models` entries are **added** (collision with shipped = error from `oflm add`)
- `defaults` merges key-by-key onto shipped `defaults`
- Arrays **replace**, never merge (e.g. `files`)

### 2.5 Shipped Registry Additions

`model_list.json` gains four top-level blocks:

```json
{
  "model_path": "models",
  "hf_owner": "Atomic-Germ",
  "version": 1,
  "defaults": {
    "pmode": "performance"
  },
  "defaults_version": "a3f2b8c1",
  "settings_schema": {
    "version": 1,
    "settings": [
      {
        "key": "prefill_mode",
        "type": "enum",
        "options": [
          {"value": "fast", "description": "Default prefill, higher memory"},
          {"value": "lean", "description": "Lower memory, ~15% slower"}
        ],
        "default": "fast",
        "scope": "per_model",
        "reload_required": true,
        "category": "performance",
        "delivery": "env_var",
        "env_var": "OFLM_OPEN_PREFILL_MODE",
        "description": "Memory/speed trade-off for prefill"
      }
    ]
  },
  "models": { ... }
}
```

- **`defaults_version`** — hash of the `defaults` block + per-model default values from settings registry entries. Used for shipped-default change detection (§2.6). Generated at package time.
- **`settings_schema`** — generated from the C++ registry at package time. The `delivery` field is included so external tools (TUI, web UI) know how settings propagate.

### 2.6 Shipped-Default Change Detection (NEW — abondis, Oct 19)

**The problem:** A release ships a new default for a setting (e.g. `prefill_mode` default changes from `fast` to `lean` for a model). The user has `prefill_mode: fast` in their `settings.json` from when they explicitly chose it — or from when it *was* the default. Either way, the user's override silently wins, and they never learn the shipped default changed. They're stuck on a stale preference.

**abondis's suggestion:** "If there is a conflict with a user setting it can warn and offer to update the users config."

**The fix:** Track which shipped defaults version the user's settings were written against. On load, if the shipped defaults have changed, detect user overrides that are now "interesting" (i.e. the user has an override for a key whose shipped default changed) and warn.

**How it works:**

1. The shipped registry includes a `defaults_version` field (a hash of the `defaults` block + all per-model default values from registry entries). Generated at package time alongside `settings_schema`.

```json
{
  "defaults": { "pmode": "performance" },
  "defaults_version": "a3f2b8c1",
  "settings_schema": { ... },
  "models": { ... }
}
```

2. `settings.json` records the `defaults_version` it was last reconciled against:

```json
{
  "version": 1,
  "defaults_version": "a3f2b8c1",
  "defaults": { "pmode": "balanced" },
  "settings": { ... },
  "models": { ... }
}
```

3. On load, compare `shipped.defaults_version` with `user.defaults_version`:
   - **Same** → no action
   - **Different** → shipped defaults changed since user last reconciled
   - **User file missing `defaults_version`** → first run with this feature; record current version, no warning (avoid noise on first upgrade)

4. When versions differ, find user overrides that may be affected. For each user setting in `settings.<type>.<size>.<key>` or `defaults.<key>`:
   - Check if the shipped default for that key exists and differs from the user's value
   - If yes → this is a "potential conflict": the user is overriding a shipped default

5. Print a one-time warning (not on every run — only when versions differ AND haven't been reconciled):

```
[OFLM] Shipped defaults changed in this release (was a3f2b8c1, now d4e5f6a7).
[OFLM] Your settings override these defaults:
[OFLM]   - prefill_mode for qwen3.8:27b: you have 'fast', shipped default is 'fast'
[OFLM]     (override is redundant — same as shipped default)
[OFLM]   - pmode (global): you have 'balanced', shipped default is 'performance'
[OFLM]     (override differs from shipped default)
[OFLM]
[OFLM] See all effective values:  oflm config show
[OFLM] Reconcile your overrides:  oflm config review
[OFLM]   (or edit ~/.config/oflm/settings.json directly)
```

6. **`oflm config review`** — interactive reconciliation (modifies settings.json):

```
$ oflm config review
Shipped defaults changed. Reconcile your overrides:

  prefill_mode for qwen3.8:27b
    Your value:      fast
    Shipped default: fast (same — override is redundant)
    [k] Keep  [r] Remove (use shipped default)  [s] Skip

  pmode (global)
    Your value:      balanced
    Shipped default: performance (different)
    [k] Keep  [r] Remove (use shipped default)  [s] Skip

Done. Removed 1 override, kept 1. settings.json updated.
```

After review, update `defaults_version` in settings.json to the current shipped version. The warning won't appear again until the next release changes defaults.

7. **Non-interactive reconciliation:** `oflm config review --accept-shipped` removes all overrides that match shipped defaults (redundant ones). `oflm config review --keep-all` just updates the version without changes.

**Edge cases:**
- User setting for a key not in shipped defaults → not a conflict, no warning
- User setting equals shipped default → redundant override, offer to remove
- User setting differs from shipped default → intentional override, show but default to "keep"
- `defaults_version` missing from shipped registry (old binary reading new file, or vice versa) → skip detection, no warning

**Where this lives:** Phase 2 (Config Store) — the version comparison and warning. Phase 4 (CLI) — the `config review` command. Phase 5 (UI) — the interactive review flow.

---

## 3. Phase-by-Phase Implementation

### Phase 1: Settings Registry (Source of Truth)

**New files:**
- `src/config/settings_registry.hpp`
- `src/config/settings_registry.cpp`

**Tasks:**
1. Define `SettingDef`, `SettingOption`, `SettingType`, `SettingScope`, `DeliveryMechanism`
2. Populate the registry with the 5 initial settings (table in §2.3)
3. Implement `generate_schema_json()` — serializes registry to `settings_schema` block format, plus `defaults_version` hash
4. Implement `generate_help_lines()` — formats settings for `--help` output
5. Implement lookup helpers (`find_setting`, `settings_for_scope`)

**CMake:** `src/config/*.cpp` files are picked up by the existing `file(GLOB ...)` at `src/CMakeLists.txt:293-295` if placed under `src/`. Verify the glob pattern includes `config/` subdirectory; if not, add explicitly.

**Test:** `src/test/settings_registry_test.cpp`
- Every setting has non-empty key, description, category, delivery mechanism
- ENUM settings have ≥2 options, each with a description
- INT settings have `min_value`
- ENV_VAR delivery settings have non-empty `env_var`
- `generate_schema_json()` produces valid JSON with all settings and correct `delivery` field
- `compute_defaults_version()` produces stable hash for same input, different hash when defaults change
- `find_setting()` works for all registered keys, returns nullopt for unknown
- Schema version is included and matches a constant

---

### Phase 2: Config Store (Load, Merge, Save, Migrate)

**New files:**
- `src/config/config_store.hpp`
- `src/config/config_store.cpp`

**Modified files:**
- `src/common/utils.cpp` — Add `find_shipped_registry()`, `find_user_settings()`; keep `find_model_list()` as wrapper for backward compat
- `src/include/utils/utils.hpp` — Declare new functions
- `src/include/model_list.hpp` — Add constructor accepting pre-loaded `nlohmann::json`

**New API:**

```cpp
// src/config/config_store.hpp
namespace oflm::config {

struct EffectiveConfig {
    nlohmann::json shipped_registry;   // Loaded shipped model_list.json
    nlohmann::json user_settings;      // Loaded settings.json (or empty object)
    nlohmann::json merged;             // Deep-merged result (shipped + settings overlays + user models)
    std::string shipped_path;
    std::string user_settings_path;
    bool migrated = false;
    std::vector<std::string> migration_log;
    // Shipped-default change detection (§2.6)
    bool defaults_changed = false;     // shipped.defaults_version != user.defaults_version
    std::string shipped_defaults_version;
    std::string user_defaults_version;
    std::vector<std::string> affected_overrides;  // Keys where user overrides shipped default
};

// File locking guard for concurrent access
class SettingsLock {
public:
    explicit SettingsLock(const std::string& path);  // Acquires flock/LockFileEx, 2s timeout
    ~SettingsLock();                                  // Releases
    bool locked() const;
};

// Load shipped registry + user settings, merge, run migration if needed.
// On corrupted settings.json: prints friendly error, continues with shipped only.
// Also performs shipped-default change detection (§2.6): compares defaults_version,
// populates defaults_changed and affected_overrides, prints warning if changed.
EffectiveConfig load_effective_config();

// Deep-merge overlay onto base. Objects merge recursively; arrays replace.
nlohmann::json deep_merge(const nlohmann::json& base, const nlohmann::json& overlay);

// Save user settings atomically (temp file in same dir + rename).
// Caller must hold SettingsLock.
void save_user_settings(const nlohmann::json& settings, const std::string& path);

// Reconcile user settings with current shipped defaults (§2.6).
// Mode: "interactive" | "accept-shipped" | "keep-all"
// Returns number of overrides removed.
// Updates defaults_version in settings.json to current shipped version.
int reconcile_settings(const std::string& settings_path, const std::string& mode);

// Three-way migration: legacy full-copy model_list.json → settings.json.
// 1. Tags not in shipped → user-added models → settings.json models
// 2. Tags in shipped, identical → dropped (stale copy)
// 3. Tags in shipped, different → preserved as overrides in settings.json settings
// Returns migration log lines.
std::vector<std::string> migrate_legacy_user_registry(
    const std::string& legacy_path,
    const nlohmann::json& shipped_registry,
    const std::string& settings_path);

} // namespace oflm::config
```

**Path discovery:**

```cpp
// src/common/utils.cpp

// Shipped registry — same search as current find_model_list but WITHOUT
// the user registry step (step 4 is removed).
std::string find_shipped_registry();
// Search order:
//   1. OFLM_CONFIG_PATH (whole-registry override — no user overlay applied)
//   2. <exe_dir>/model_list.json
//   3. ./model_list.json
//   4. <exe_dir>/../share/oflm/model_list.json
//   5. CMAKE_INSTALL_PREFIX/share/oflm/model_list.json

// User settings file — always the overlay location.
std::string find_user_settings();
// Search order:
//   1. OFLM_SETTINGS_PATH (for testing)
//   2. <models_dir>/settings.json  (sibling of models directory)
//      Linux: ~/.config/oflm/settings.json
//      Windows: %USERPROFILE%\.oflm\settings.json

// Backward compat: find_model_list() = find_shipped_registry()
// (unless OFLM_CONFIG_PATH is set, then it's a whole-registry override)
```

**OFLM_CONFIG_PATH semantics (C2 fix):**
- Set → loads that file as the **complete** registry. No user overlay. For hermetic testing/dev.
- Unset → loads shipped registry + overlays `settings.json`.

**Migration logic (C1 fix — three-way):**

```
if ~/.config/oflm/model_list.json exists AND settings.json does NOT exist:
    1. Load legacy file
    2. Load shipped registry
    3. For each model tag in legacy:
       a. Tag NOT in shipped registry:
            → User-added model → settings.json → models.<type>.<size>
            Log: "Carried over user-added model: <tag>"
       b. Tag IN shipped, entries identical (deep equality):
            → Stale copy → dropped
            (no log — expected for users who ran oflm add)
       c. Tag IN shipped, entries differ:
            → Preserved as override → settings.json → settings.<type>.<size>
            Log: "Carried over modified entry for <tag> (differs from shipped; preserved as override)"
    4. Top-level defaults (if any custom keys) → settings.json → defaults
    5. Save settings.json (with SettingsLock)
    6. Rename model_list.json → model_list.json.bak
    7. Log summary
```

**Corrupted settings.json handling (H3 fix):**

```
try {
    user_settings = nlohmann::json::parse(file);
} catch (const nlohmann::json::parse_error& e) {
    std::cerr << "[OFLM] ERROR: " << settings_path << " is not valid JSON:\n"
              << "[OFLM]   " << e.what() << "\n"
              << "[OFLM]   Fix the file manually, or move it aside to start fresh:\n"
              << "[OFLM]     mv " << settings_path << " " << settings_path << ".broken\n"
              << "[OFLM]   Continuing with shipped defaults only.\n";
    user_settings = nlohmann::json::object();
}
```

**File locking (H4 fix):**

```cpp
// POSIX
class SettingsLock {
    int lock_fd_ = -1;
    bool locked_ = false;
public:
    explicit SettingsLock(const std::string& path) {
        lock_fd_ = open((path + ".lock").c_str(), O_CREAT | O_RDWR, 0644);
        if (lock_fd_ < 0) return;
        struct flock fl{};
        fl.l_type = F_WRLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0;
        // Try immediately, then retry every 100ms for up to 2 seconds
        for (int i = 0; i < 21; i++) {
            if (fcntl(lock_fd_, F_SETLK, &fl) == 0) {
                locked_ = true;
                return;
            }
            if (i < 20) usleep(100000);
        }
    }
    ~SettingsLock() {
        if (lock_fd_ >= 0) {
            struct flock fl{};
            fl.l_type = F_UNLCK;
            fl.l_whence = SEEK_SET;
            fcntl(lock_fd_, F_SETLK, &fl);
            close(lock_fd_);
        }
    }
    bool locked() const { return locked_; }
};

// Windows: LockFileEx with LOCKFILE_FAIL_IMMEDIATELY, same retry loop
```

**Test:** `src/test/config_store_test.cpp`
- `deep_merge`: sparse overlay merges recursively; arrays replace; nulls don't overwrite
- `load_effective_config`: fresh install (no user file) → only shipped
- `load_effective_config`: with user file → merged correctly
- `load_effective_config`: corrupted JSON → friendly error, shipped only
- `migrate_legacy_user_registry`: old full-copy → settings.json + models split correctly
- `migrate_legacy_user_registry`: modified entries preserved as overrides (C1)
- `migrate_legacy_user_registry`: idempotent (won't re-migrate if settings.json exists)
- `save_user_settings`: atomic write, valid JSON, temp file cleaned up
- `SettingsLock`: concurrent access serializes

---

### Phase 3: Settings Resolution and Application

**New files:**
- `src/config/settings_resolver.hpp`
- `src/config/settings_resolver.cpp`

**Modified files:**
- `src/src/main.cpp` — Load EffectiveConfig, resolve settings, apply
- `src/common/AutoModel/automodel.cpp` — Consume resolved settings
- `src/runner/runner.cpp` — Consume resolved settings
- `src/server/rest_handler.cpp` — Consume resolved settings, per-request overrides

**C4 fix — no hardcoded struct, use JSON:**

```cpp
// src/config/settings_resolver.hpp
namespace oflm::config {

// Resolve effective settings for a model tag.
// Precedence: shipped defaults < user settings < env vars < CLI flags
// Returns JSON object: { "prefill_mode": "lean", "ctx_len": 16384, ... }
// Only includes settings that have a value from some source.
nlohmann::json resolve_settings(
    const EffectiveConfig& cfg,
    const std::string& model_tag,      // Empty for global-only settings
    const program_args_t& args);      // CLI flags

// Apply resolved settings using each setting's declared delivery mechanism.
// For ENV_VAR: setenv() each setting that has env_var defined.
// For DIRECT: returns the JSON for the caller to pass to consumers.
// For API_FIELD: returns the JSON for the REST handler to use as per-request defaults.
// For REGISTRY: no-op (already in merged registry entry).
void apply_settings(const nlohmann::json& resolved);

// Get a single setting's effective value with source tracking.
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

**Resolution algorithm:**

```cpp
nlohmann::json resolve_settings(cfg, model_tag, args) {
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
            if (const char* v = getenv_oflm(def.env_var.c_str()); v && *v) {
                resolved[def.key] = convert_type(v, def.type);
                continue;
            }
        }

        // 2. User settings
        // Check settings.<type>.<size>.<key> for per-model
        // Check defaults.<key> for global
        if (model_tag.empty()) {
            if (cfg.user_settings["defaults"].contains(def.key)) {
                resolved[def.key] = cfg.user_settings["defaults"][def.key];
                continue;
            }
        } else {
            auto [type, size] = split_tag(model_tag);
            if (cfg.user_settings["settings"][type][size].contains(def.key)) {
                resolved[def.key] = cfg.user_settings["settings"][type][size][def.key];
                continue;
            }
        }

        // 1. Shipped defaults
        if (cfg.shipped_registry["defaults"].contains(def.key)) {
            resolved[def.key] = cfg.shipped_registry["defaults"][def.key];
            continue;
        }
        // Per-model shipped default from registry entry
        if (!def.registry_key.empty() && !model_tag.empty()) {
            auto entry = get_merged_model_entry(cfg, model_tag);
            if (entry.contains(def.registry_key)) {
                resolved[def.key] = entry[def.registry_key];
                continue;
            }
        }
        // Built-in default from registry definition
        if (!def.default_value.empty()) {
            resolved[def.key] = convert_type(def.default_value, def.type);
        }
    }
    return resolved;
}

// Convert a string value to the setting's declared type for JSON storage.
// INT → int64, BOOL → bool, ENUM/STRING → string.
nlohmann::json convert_type(const std::string& value, SettingType type);
```

**Application (delivery) — the reachability fix:**

```cpp
void apply_settings(const nlohmann::json& resolved) {
    for (const auto& def : settings_registry()) {
        if (!resolved.contains(def.key)) continue;
        std::string value = resolved[def.key].get<std::string>();

        switch (def.delivery) {
            case DeliveryMechanism::ENV_VAR:
                // The systematic setenv bridge — replaces ad-hoc main.cpp:619-625
                setenv(def.env_var.c_str(), value.c_str(), 1);
                break;
            case DeliveryMechanism::DIRECT:
            case DeliveryMechanism::API_FIELD:
            case DeliveryMechanism::REGISTRY:
                // Value is in the resolved JSON; consumers read from there
                break;
        }
    }
}
```

**Consumer changes:**

```cpp
// automodel.cpp _shared_load_model — reads from resolved settings instead of
// ad-hoc registry reads:
void AutoModel::_shared_load_model(..., const nlohmann::json& resolved_settings) {
    // ctx_len: resolved_settings["ctx_len"] or model_info["default_context_length"]
    if (resolved_settings.contains("ctx_len")) {
        this->MAX_L = resolved_settings["ctx_len"].get<int>();
    } else {
        this->MAX_L = model_info["default_context_length"].get<int>();
    }
    // ...
}

// main.cpp — replaces the hardcoded setenv at line 619-625:
// Old: setenv("OFLM_OPEN_PREFILL_MODE", parsed_args.prefill_mode.c_str(), 1);
// New: apply_settings(resolved);  // handles all ENV_VAR settings systematically
```

**Test:** `src/test/settings_resolver_test.cpp`
- Precedence: CLI > env > user > shipped for each setting
- Per-model vs global scope filtering
- Delivery: ENV_VAR settings are setenv'd; DIRECT/REGISTRY are not
- Unknown settings in user file: warned but loaded (H5)
- Schema version mismatch: lists ignored keys (H2)

---

### Phase 4: CLI Commands (Inside Boost Parser)

**C3 fix — `config` is a Boost subcommand, not a pre-Boost handoff.**

**Modified files:**
- `src/include/utils/vm_args.hpp` — Add config-specific options and `config` command validation
- `src/include/program_args.hpp` — Add config fields to `program_args_t`
- `src/src/main.cpp` — Dispatch `config` subcommand
- `src/config/config_cmd.hpp` (new)
- `src/config/config_cmd.cpp` (new)

**New CLI options (in vm_args.hpp):**

The `config` action is **positional** — parsed from the existing `model_tag` positional argument when `command == "config"`. This matches the command syntax `oflm config <action>` and keeps the UX clean.

```cpp
// Added to the options_description:
("config-key", po::value<std::string>(), "Setting key for config set/unset")
("config-value", po::value<std::string>(), "Setting value for config set")
("config-model", po::value<std::string>(), "Model tag for per-model config operations")
("config-accept-shipped", po::bool_switch(), "config review: remove all redundant overrides")
("config-keep-all", po::bool_switch(), "config review: keep all overrides, just update version")

// In program_args_t:
std::string config_key;
std::string config_value;
std::string config_model;
bool config_accept_shipped = false;
bool config_keep_all = false;

// The action is parsed from parsed_args.model_tag when command == "config":
//   oflm config show       → command="config", model_tag="show" (action)
//   oflm config set ...    → command="config", model_tag="set" (action)
//   oflm config review     → command="config", model_tag="review" (action)
```

**Valid actions:** `show`, `set`, `unset`, `edit`, `browse`, `review`, `migrate`, `gen-schema`
**No action** (`oflm config` with no second positional) → guided interactive flow (Phase 5)

**Commands:**

```bash
# Guided interactive flow (no action)
oflm config                               # Opens guided flow (Phase 5)

# Show effective configuration (read-only)
oflm config show                          # All global settings + sources
oflm config show --config-model qwen3.8:27b   # One model's settings
oflm config show --json                   # Machine-readable (global --json flag works!)

# Set a preference (modifies settings.json)
oflm config set --config-key pmode --config-value balanced           # Global
oflm config set --config-model qwen3.8:27b --config-key prefill_mode --config-value lean
oflm config set --config-model qwen3.8:27b --config-key prefill_mode # No value → interactive picker

# Unset (revert to shipped default)
oflm config unset --config-key pmode
oflm config unset --config-model qwen3.8:27b --config-key prefill_mode

# Edit the file directly
oflm config edit                          # Open settings.json in $EDITOR

# Full TUI
oflm config browse

# Reconcile overrides after shipped defaults changed (§2.6)
oflm config review                        # Interactive: keep/remove/skip each override
oflm config review --accept-shipped       # Remove all redundant overrides (match shipped)
oflm config review --keep-all             # Just update defaults_version, no changes

# Internal (for packaging)
oflm config gen-schema                    # Output settings_schema JSON to stdout

# Manual migration trigger
oflm config migrate
```

**Command semantics — read vs. modify:**

| Command | Reads settings.json | Writes settings.json | Purpose |
|---------|---------------------|----------------------|---------|
| `config show` | Yes | **No** | See effective values and their sources |
| `config set` | Yes | **Yes** | Set a preference |
| `config unset` | Yes | **Yes** | Remove a preference (revert to shipped) |
| `config edit` | No (opens editor) | Via editor | Hand-edit the file |
| `config browse` | Yes | **Yes** (on save) | Full-screen TUI editor |
| `config review` | Yes | **Yes** | Reconcile overrides after shipped defaults changed |
| `config migrate` | Yes | **Yes** | Force legacy migration re-check |
| `config gen-schema` | No | **No** | Output schema JSON (internal/packaging) |

**Key distinction:**
- **See** → `config show` (read-only, safe to run anytime)
- **Change** → `config set` / `config unset` / `config edit` / `config browse` / `config review` (modifies settings.json)

**Why inside Boost:** Global flags (`--json`, `--quiet`) work consistently. Help text is auto-generated. Per-command validation follows the existing pattern. No special-casing in main.cpp.

**Dispatch in main.cpp (after `list` dispatch, around line 765):**

```cpp
if (parsed_args.command == "config") {
    return oflm::config::config_command_main(parsed_args);
}
```

**`config show` output:**

```
$ oflm config show --config-model qwen3.8:27b
prefill_mode      lean          (user)
pmode             performance   (shipped)
ctx_len           8192          (shipped)
prefill_chunk_len 4096          (shipped)
think             true          (shipped)
```

**Source tags** (from `SettingValue::Source`):

| Tag | Meaning |
|-----|---------|
| `(shipped)` | Value from shipped `model_list.json` defaults |
| `(user)` | Value from user's `settings.json` |
| `(env)` | Value from `OFLM_*` environment variable |
| `(cli)` | Value from command-line flag |
| `(api)` | Value from per-request API field (serve mode) |

**`config set` behavior (H5 fix — warn, don't reject):**

```
$ oflm config set --config-key decode_mode --config-value fast
[OFLM] WARNING: 'decode_mode' is not a known setting in this version of oflm.
[OFLM]   It will be stored but may not have any effect.
[OFLM]   Known settings: prefill_mode, pmode, ctx_len, prefill_chunk_len, think
Set anyway? [y/N] _
```

For known keys, validate the value against type/options. On invalid value, show valid options and exit 1.

**`config set` with no value → interactive picker** (Phase 5).

**`config review` implementation by PR:**
- **PR 1 (Phase 4):** Non-interactive modes only — `--accept-shipped`, `--keep-all`. Interactive mode prints "not yet available, use --accept-shipped or --keep-all".
- **PR 2 (Phase 5):** Interactive mode with Tier 1/2 pickers.

**Test:** Integration test via `oflm_smoke` extension; unit tests for validation logic.

---

### Phase 5: Interactive UI (Three-Tier)

**New files:**
- `src/config/config_ui.hpp`
- `src/config/config_ui.cpp`

**H1 fix — three tiers, graceful degradation:**

#### Tier 1: Numbered menu (always works)

No escape codes, no raw mode. Works on `TERM=dumb`, CI, SSH without TTY forwarding, Windows < 10.

```
Select setting for qwen3.8:27b:
  1. prefill_mode       fast (shipped)
  2. ctx_len            8192 (shipped)
  3. prefill_chunk_len  4096 (shipped)
  4. think              true (shipped)
  0. Back
> _
```

#### Tier 2: Arrow-key picker (if terminal supports raw input)

termios (POSIX) / Windows Console API. Detected via `isatty(STDIN_FILENO)` + `tcgetattr` success on POSIX; `GetConsoleMode` on Windows.

```
? prefill_mode for qwen3.8:27b (current: fast, source: shipped)
  > fast       Default prefill, higher memory
    lean       Lower memory, ~15% slower

  ↑/↓ navigate  Enter select  Esc cancel
```

#### Tier 3: Full-screen TUI (`oflm config browse` only)

ANSI escape codes. Detected via `TERM` env var (not "dumb", not empty) + `isatty(STDOUT_FILENO)` + Windows VT processing enabled.

If Tier 3 unavailable:
```
[OFLM] Full-screen mode requires a modern terminal.
[OFLM] Use 'oflm config' for the interactive guided flow, or
[OFLM] 'oflm config set --config-key <key> --config-value <value>' for direct setting.
```

**Capability detection:**

```cpp
enum class UiTier { NUMBERED_MENU, ARROW_PICKER, FULL_TUI };

UiTier detect_ui_tier() {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO))
        return UiTier::NUMBERED_MENU;

#ifdef _WIN32
    // Try to enable VT processing
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    if (!GetConsoleMode(hOut, &mode)) return UiTier::NUMBERED_MENU;
    if (!(mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
        if (!SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
            return UiTier::ARROW_PICKER;  // Console API works, VT doesn't
    }
    return UiTier::FULL_TUI;
#else
    const char* term = getenv("TERM");
    if (!term || std::string(term) == "dumb") return UiTier::NUMBERED_MENU;

    struct termios t;
    if (tcgetattr(STDIN_FILENO, &t) != 0) return UiTier::NUMBERED_MENU;

    // Check for ANSI support — most modern terminals
    if (getenv("COLORTERM") || std::string(term).find("xterm") != std::string::npos
        || std::string(term).find("screen") != std::string::npos
        || std::string(term).find("tmux") != std::string::npos)
        return UiTier::FULL_TUI;

    return UiTier::ARROW_PICKER;
#endif
}
```

**Picker implementation (Tier 2):**

```cpp
class Picker {
public:
    struct Item { std::string label; std::string description; };
    Picker(std::string title, std::vector<Item> items, size_t current = SIZE_MAX);
    // Returns selected index, or SIZE_MAX if cancelled (Esc)
    size_t run();
};
```

**Guided flow (`oflm config` with no action):**

Chains pickers: action → model → setting → value. Uses Tier 2 picker, falls back to Tier 1 numbered menu.

```
$ oflm config
? What do you want to do?          [Tier 2 picker, or numbered menu on dumb terminal]
  > Edit global settings
    Edit a model's settings
    Show effective configuration
    Exit
```

**Full TUI (Tier 3, `oflm config browse`):**

```
┌─ oflm config ─────────────────────────────────────────────┐
│                                                           │
│  GLOBAL                                                   │
│    pmode ........................ performance (shipped)   │
│                                                           │
│  qwen3.8:27b                                              │
│    prefill_mode ................. lean (USER)             │
│    ctx_len ...................... 8192 (shipped)          │
│    prefill_chunk_len ............ 4096 (shipped)          │
│    think ........................ true (shipped)          │
│                                                           │
│  my-finetune:3b (user-added)                              │
│    url .......................... https://...             │
│                                                           │
│  [↑↓/jk] navigate  [Enter] edit  [d] delete  [s] save     │
│  [q] quit (prompts if unsaved)                            │
└───────────────────────────────────────────────────────────┘
```

**Test:**
- Tier detection unit tests (mock isatty, getenv)
- Picker logic test with abstracted terminal interface (mock input)
- Smoke test: `echo "q" | oflm config browse` exits 0 (catches crashes)
- Smoke test: `printf "0\n" | oflm config` exits 0 (Tier 1 fallback)

---

### Phase 6: Schema Generation at Package Time

**New files:**
- `utilities/release/inject-schema.py` — Runs `oflm config gen-schema`, merges `settings_schema` + `defaults_version` into model_list.json

**Modified files:**
- `src/CMakeLists.txt` — Add `settings_schema.json` generation to build dir (not source)
- `.github/workflows/release.yml` — Inject schema into build-dir copy before cpack

**Approach:** inject-schema.py runs in the release workflow only. The CMake post-build approach was considered but rejected — it changes the install source for model_list.json and risks breaking the Windows post-build copy (src/CMakeLists.txt:881-891) and other consumers.

`oflm config gen-schema` outputs a JSON object with both blocks:
```json
{
  "settings_schema": { "version": 1, "settings": [...] },
  "defaults_version": "a3f2b8c1"
}
```

inject-schema.py merges both into model_list.json.

```yaml
# .github/workflows/release.yml — add before cpack step
- name: Inject settings schema
  run: |
    ./build/src/oflm config gen-schema > /tmp/schema.json
    python3 utilities/release/inject-schema.py \
      --input src/model_list.json \
      --schema /tmp/schema.json \
      --output src/model_list.json
    # In the release branch, modifying the source file is fine
    # (the branch is disposable; the tag is cut after this step)
```

**H2 fix — schema version mismatch at runtime:**

```cpp
// In load_effective_config():
auto shipped_schema_version = shipped["settings_schema"]["version"].get<int>();
auto binary_schema_version = SCHEMA_VERSION;  // From registry

if (shipped_schema_version > binary_schema_version) {
    // Binary is older than shipped schema (downgrade scenario)
    auto known_keys = get_all_setting_keys();  // From registry
    auto shipped_keys = get_schema_setting_keys(shipped["settings_schema"]);
    auto unknown = diff(shipped_keys, known_keys);
    std::cerr << "[OFLM] WARNING: model_list.json uses settings schema v"
              << shipped_schema_version << " but this binary supports v"
              << binary_schema_version << ".\n";
    if (!unknown.empty()) {
        std::cerr << "[OFLM]   Unknown settings will be ignored: "
                  << join(unknown, ", ") << "\n";
    }
    std::cerr << "[OFLM]   Upgrade oflm or remove these keys from settings.json\n";
}
// If shipped < binary: normal upgrade, use binary's registry (no warning)
// If no schema in shipped: dev build, use binary's registry (no warning)
```

---

### Phase 7: `oflm add` Changes

**Modified files:**
- `utilities/oflm-add/oflm_add/__init__.py` — `register()`, path resolution, argparse

**Changes:**

1. **Path resolution:** `user_registry_path()` → `user_settings_path()`
   - `--config` → `OFLM_SETTINGS_PATH` → `<models_dir>/settings.json`
   - No longer writes to `model_list.json`

2. **`register()` rewrite:**
   ```python
   def register(entry, settings_path, system_registry, force=False, name_override=None):
       # Load user settings (create if missing)
       settings = load_or_create(settings_path)
       
       tag = name_override or f"{entry['type']}:{entry['size']}"
       model_type, size = tag.split(":", 1)
       
       # Collision check: shipped registry
       if model_type in system_registry["models"] and size in system_registry["models"][model_type]:
           if not name_override:
               error(f"Model {tag} already exists in shipped registry. Use --name to install under a different name.")
       
       # Collision check: user settings
       if model_type in settings.get("models", {}) and size in settings["models"][model_type]:
           if not force:
               error(f"Model {tag} already installed. Use --force to overwrite.")
       
       # Write complete entry to settings.models
       settings.setdefault("models", {}).setdefault(model_type, {})[size] = entry
       save_json_atomic(settings_path, settings)
   ```

3. **New CLI flags:**
   - `--name <tag>` — Install under a different tag
   - `--force` — Overwrite existing user model

4. **Final message:**
   ```
   Model my-finetune:3b installed to ~/.config/oflm/settings.json
   Run with: oflm run my-finetune:3b
   ```

**Test:** `utilities/oflm-add/tests/test_settings_json.py`
- `register()` writes to `settings.json` → `models` section
- Does NOT touch `model_list.json`
- Collision with shipped model → error with `--name` suggestion
- Collision with user model → error without `--force`, succeeds with `--force`
- `--name` installs under custom tag
- Settings in `settings` section are preserved (not clobbered by `oflm add`)

---

### Phase 8: Integration with Engine Settings

**Modified files:**
- `src/src/main.cpp` — Load EffectiveConfig; replace hardcoded `setenv` at :619-625 with `apply_settings()`
- `src/common/AutoModel/automodel.cpp` — `_shared_load_model` takes `resolved_settings` JSON param
- `src/runner/runner.cpp` — Pass resolved settings to engine
- `src/server/rest_handler.cpp` — Use resolved settings as per-request defaults; API `think` field overrides

**Flow:**

```
main.cpp:
  1. EffectiveConfig cfg = load_effective_config();     // shipped + user + migration
  2. nlohmann::json resolved = resolve_settings(cfg, tag, args);  // + env + CLI
  3. apply_settings(resolved);                          // setenv for ENV_VAR settings
  4. model_list models(cfg.merged, models_dir);          // merged registry
  5. Runner(models, downloader, args, resolved).run();  // resolved passed to engine
```

**Engine consumption:**

```cpp
// automodel.cpp
// Old: this->MAX_L = model_info["default_context_length"];
// New: check resolved_settings first
if (resolved.contains("ctx_len")) {
    this->MAX_L = resolved["ctx_len"].get<int>();
} else {
    this->MAX_L = model_info["default_context_length"].get<int>();
}

// prefill_mode: no longer read from env var directly in engine
// The env var is set by apply_settings() before engine construction
// Engine code stays the same (reads OFLM_OPEN_PREFILL_MODE)
// But the SOURCE of the value is now the settings system, not just CLI
```

**Per-request API overrides (serve mode):**

```cpp
// rest_handler.cpp configure_chat_engine_parameters()
// Existing: request["think"] → configure_parameter("enable_think", ...)
// New: resolved settings provide the DEFAULT; request field overrides

bool default_think = resolved_settings.value("think", true);
bool request_think = request.value("think", default_think);  // API field wins
configure_parameter("enable_think", request_think);
```

---

### Phase 9: Testing

**New test files:**

| File | Tests |
|------|-------|
| `src/test/settings_registry_test.cpp` | Registry integrity, schema generation, delivery field present |
| `src/test/config_store_test.cpp` | Deep merge, load, three-way migration, corrupted JSON, file locking |
| `src/test/settings_resolver_test.cpp` | Precedence chain, scope filtering, delivery (setenv), unknown key warning |
| `src/test/config_ui_test.cpp` | Tier detection, picker logic (mocked terminal) |
| `utilities/oflm-add/tests/test_settings_json.py` | register() writes to settings.json, collision detection, --name/--force |

**Modified test files:**
- `src/CMakeLists.txt` — Register new tests (around line 1373)
- `src/test/CMakeLists.txt` — Add host-only test suites

**Smoke tests (extend oflm_smoke):**
```cmake
# Test config commands don't crash (read-only)
add_test(NAME oflm_config_show COMMAND oflm config show)
add_test(NAME oflm_config_gen_schema COMMAND oflm config gen-schema)

# Test config set/unset with isolated settings file (not the user's real one)
add_test(NAME oflm_config_set_unset
    COMMAND sh -c "export OFLM_SETTINGS_PATH=${CMAKE_BINARY_DIR}/test_settings.json && \
                   oflm config set --config-key pmode --config-value balanced && \
                   oflm config show --json | grep -q '\"pmode\":\"balanced\"' && \
                   oflm config unset --config-key pmode && \
                   oflm config show --json | grep -q '\"pmode\":\"performance\"'")

# TUI smoke test (Tier 1 fallback via piped input)
add_test(NAME oflm_config_browse_smoke
    COMMAND sh -c "echo q | oflm config browse")

# Guided flow smoke test (Tier 1 fallback)
add_test(NAME oflm_config_guided_smoke
    COMMAND sh -c "printf '0\n' | oflm config")
```

**Test scenarios:**

| Scenario | Expected |
|----------|----------|
| Fresh install, no user file | Shipped defaults only |
| User sets preference via `config set` | Persisted in settings.json, applied on next run |
| Precedence: CLI > env > user > shipped | Correct winner for each source |
| Migration: old full-copy with user models | User models → settings.json models, .bak created |
| Migration: modified shipped entry | Preserved as override in settings.json settings (C1) |
| Migration: stale identical entries | Dropped silently |
| Migration: idempotent | Second run doesn't re-migrate |
| Corrupted settings.json | Friendly error, continues with shipped |
| `oflm add` writes to settings.json | model_list.json untouched |
| `oflm add` collision with shipped | Error, suggests --name |
| `oflm add` collision with user model | Error without --force, succeeds with --force |
| Schema generation | Matches registry, includes delivery field |
| Deep merge | Sparse overlay, arrays replace, objects recurse |
| ENV_VAR delivery | setenv called with correct value (prefill_mode pattern) |
| DIRECT delivery | Value in resolved JSON, no setenv |
| Concurrent `config set` | File lock serializes, no lost update |
| TUI on dumb terminal | Falls back to numbered menu, doesn't crash |
| Schema version mismatch (downgrade) | Lists ignored keys in warning |

---

### Phase 10: Documentation

**New files:**
- `docs/config.md` — User guide: two-file system, commands, precedence, migration, examples

**Modified files:**
- `src/create_new_model.md` — Document settings registry, how to add new settings (including delivery mechanism)
- `src/include/utils/vm_args.hpp` — `print_help()` includes config subcommand

**Man pages:** Deferred. `--help` output from registry is primary documentation.

---

## 4. File-by-File Change Summary

### New Files

| File | Purpose |
|------|---------|
| `src/config/settings_registry.hpp` | SettingDef (with DeliveryMechanism), registry declaration |
| `src/config/settings_registry.cpp` | Registry implementation, schema/help generation |
| `src/config/config_store.hpp` | EffectiveConfig, SettingsLock, load/merge/save/migrate API |
| `src/config/config_store.cpp` | Two-file loading, deep merge, three-way migration, file locking |
| `src/config/settings_resolver.hpp` | resolve_settings, apply_settings, get_setting_value |
| `src/config/settings_resolver.cpp` | Precedence resolution, delivery mechanism application |
| `src/config/config_cmd.hpp` | Config subcommand declarations |
| `src/config/config_cmd.cpp` | show/set/unset/edit/browse/migrate/gen-schema implementations |
| `src/config/config_ui.hpp` | UiTier, Picker, guided flow, TUI declarations |
| `src/config/config_ui.cpp` | Three-tier UI: numbered menu, arrow picker, ANSI TUI |
| `src/test/settings_registry_test.cpp` | Registry unit tests |
| `src/test/config_store_test.cpp` | Merge/migration/locking unit tests |
| `src/test/settings_resolver_test.cpp` | Precedence/delivery unit tests |
| `src/test/config_ui_test.cpp` | Tier detection/picker unit tests |
| `utilities/release/inject-schema.py` | Package-time schema injection |
| `utilities/oflm-add/tests/test_settings_json.py` | oflm-add settings.json tests |
| `docs/config.md` | User-facing configuration guide |

### Modified Files

| File | Changes |
|------|---------|
| `src/common/utils.cpp` | Add `find_shipped_registry()`, `find_user_settings()`; `find_model_list()` delegates |
| `src/include/utils/utils.hpp` | Declare new path-finding functions |
| `src/include/model_list.hpp` | Add constructor accepting pre-loaded `nlohmann::json` |
| `src/src/main.cpp` | Load EffectiveConfig; replace `setenv` at :619-625 with `apply_settings()`; dispatch `config` |
| `src/include/program_args.hpp` | Add config_key, config_value, config_model, config_accept_shipped, config_keep_all fields |
| `src/include/utils/vm_args.hpp` | Add config options; allow `config` command; per-command validation |
| `src/common/AutoModel/automodel.cpp` | `_shared_load_model` takes resolved_settings; reads ctx_len from it |
| `src/runner/runner.cpp` | Pass resolved_settings to engine; read prefill_chunk_len from it |
| `src/server/rest_handler.cpp` | Use resolved_settings as defaults; API fields override |
| `src/CMakeLists.txt` | Register new tests; verify config/ glob |
| `src/test/CMakeLists.txt` | Add new host-only test suites |
| `src/model_list.json` | Add `version`, `defaults`, `defaults_version`, `settings_schema` blocks (via inject-schema.py) |
| `utilities/oflm-add/oflm_add/__init__.py` | `register()` writes to settings.json; add `--name`/`--force`; collision detection |
| `.github/workflows/release.yml` | Run inject-schema.py before packaging |
| `.github/workflows/cmake-build-test.yml` | Run new tests |

---

## 5. Migration Path for Existing Users

### 5.1 First Run After Upgrade

```
$ oflm run qwen3.8:27b
[oflm] Migrating config: found legacy ~/.config/oflm/model_list.json
[oflm]   Carried over 3 user-added models to settings.json:
[oflm]     - my-finetune:3b
[oflm]     - custom-qwen:8b
[oflm]     - test-model:latest
[oflm]   Carried over 1 modified entry as override:
[oflm]     - qwen3.8:27b (differs from shipped; preserved in settings.json)
[oflm]   Dropped 33 stale shipped entries (identical to shipped versions)
[oflm]   Backup saved to ~/.config/oflm/model_list.json.bak
[oflm]   Future releases will update shipped defaults automatically
[oflm] Model qwen3.8:27b loaded (ctx_len=8192, prefill_mode=fast)
```

### 5.2 Three-Way Migration Rules

| In legacy file | In shipped? | Action |
|----------------|-------------|--------|
| Model tag | No | → `settings.json` → `models` (user-added model) |
| Model tag | Yes, identical | Dropped (stale copy) |
| Model tag | Yes, different | → `settings.json` → `settings` (preserved as override) |
| Top-level `model_path`, `hf_owner` | — | Dropped (shipped values used) |
| Top-level custom keys | — | → `settings.json` → `defaults` |

### 5.3 Rollback

```
mv ~/.config/oflm/settings.json ~/.config/oflm/settings.json.new
mv ~/.config/oflm/model_list.json.bak ~/.config/oflm/model_list.json
# Downgrade oflm
```

---

## 6. Open Questions / Deferred

| Question | Decision | Rationale |
|----------|----------|-----------|
| Custom keys in interactive picker? | Strict — only schema-defined keys. `config set` warns but allows unknown keys | Prevents typos in UI; allows forward-compat via CLI |
| Man pages? | Deferred — `--help` from registry is primary docs | Man page generation is separate build tooling |
| FTXUI? | No — three-tier custom UI, no new dependency | FTXUI requires CMake/vcpkg/CI/Docker changes across 5 places |
| `oflm config diff`? | Deferred — `config show` with source tags covers it | Can add later |
| `oflm remove` command? | Deferred — out of scope | Follow-up PR |
| `oflm config edit --model`? | Dropped — editor can't generically focus sections | `config edit` opens whole file |
| LM_Config → JSON blob across DLL? | Future refactor — registry makes it mechanical | Change `delivery: ENV_VAR` → `delivery: DIRECT` per setting |
| Settings versioning? | `version: 1`; future migrations chained; unknown version → warn + best-effort load | Allows format evolution |

---

## 7. Rollout Plan

| PR | Phases | Description |
|----|--------|-------------|
| 1 | 1-4 | Registry + config store + resolver + CLI commands (core, no UI). `config review` non-interactive modes only; `config set` without value errors "use --config-value". |
| 2 | 5 | Interactive UI (three-tier). Enables `config set` picker, `config review` interactive, guided flow, `config browse` TUI. |
| 3 | 6-7 | Schema injection + oflm-add changes |
| 4 | 8 | Engine integration (apply_settings, resolved settings to consumers) |
| 5 | 9-10 | Tests + documentation |

**Dependency note:** Phase 4's interactive features (`config set` picker, `config review` interactive, guided flow) depend on Phase 5's UI primitives. PR 1 ships the non-interactive fallbacks; PR 2 upgrades them to interactive.

Or all in one PR on the feature branch.

---

## 8. Success Criteria

- [ ] `oflm config set --config-model qwen3.8:27b --config-key prefill_mode --config-value lean` persists
- [ ] `oflm run qwen3.8:27b` uses the persisted setting (via ENV_VAR delivery)
- [ ] `oflm run qwen3.8:27b --prefill-mode fast` overrides the persisted setting
- [ ] **Any new engine option can be added to the registry with `delivery: ENV_VAR` and automatically reaches the engine** (the reachability requirement)
- [ ] `oflm add <url>` writes to `settings.json`, never touches shipped `model_list.json`
- [ ] Old full-copy `model_list.json` migrates with three-way logic (preserves modifications)
- [ ] `oflm config show` displays effective values with sources
- [ ] `oflm config` (no args) opens guided flow (Tier 2, falls back to Tier 1)
- [ ] `oflm config browse` opens TUI (Tier 3, falls back gracefully)
- [ ] `oflm config gen-schema` output matches the C++ registry including `delivery` field
- [ ] Corrupted settings.json → friendly error, continues with shipped
- [ ] Concurrent `config set` → file lock prevents lost updates
- [ ] Schema version mismatch → lists ignored keys
- [ ] Shipped-default change detection → warns when defaults_version differs, lists affected overrides
- [ ] `oflm config review` reconciles overrides interactively
- [ ] `oflm config review --accept-shipped` removes redundant overrides
- [ ] All existing tests pass; new tests cover merge, migration, precedence, delivery, oflm-add, conflict detection

---

## 9. RFC Response to Issue #188 Discussion

Cyronius asked (Oct 18): *"Is there a better way, is this an acceptable risk, and is it appropriate to expand the use of model_list.json to include default settings?"*

### Is there a better way?

**Alternatives considered and rejected:**

| Alternative | Why rejected |
|-------------|--------------|
| **conf.d / drop-in directory** (`~/.config/oflm/settings.d/*.json`) | More moving parts, harder to see effective config, harder to atomically update. Single file with deep-merge is simpler and matches the issue's proposal. abondis suggested this (Oct 18, point 4) — the response is that `oflm add` writing to a separate `models` section within one file achieves the same isolation without directory-scanning complexity. |
| **Single file (status quo)** | The current bug: user file replaces shipped entirely. Users miss new models and fixes. This is the problem we're solving. |
| **Database (SQLite)** | Overkill for ~50 settings and ~50 models. JSON is human-editable, diff-able, and already the format of model_list.json. |
| **Environment variables only** | Already the status quo for debug knobs. Doesn't scale to per-model preferences; no persistence story. |

**The two-file overlay is the simplest design that:**
- Lets releases update shipped defaults without touching user data
- Lets users override any setting per-model or globally
- Keeps everything human-editable and diff-able
- Requires no new dependencies

### Is the risk acceptable?

**The risk Cyronius identified:** "We ship a new default that has a setting that doesn't play well with the other model settings a user has."

**Mitigations:**

1. **Layered precedence** — User settings always win over shipped defaults. A bad shipped default can be overridden per-model.

2. **Shipped-default change detection (§2.6)** — When a release changes a shipped default that the user has overridden, the user is warned and offered `oflm config review` to reconcile. They're never silently stuck on a stale preference.

3. **Schema validation** — The settings registry declares types and valid values. A shipped default that fails validation is caught at package time (inject-schema.py runs `oflm config gen-schema` and validates).

4. **Migration preserves user data** — Three-way migration (C1 fix) never drops user modifications. The `.bak` file is always kept.

5. **Rollback is trivial** — `settings.json` can be deleted to revert to shipped defaults. `model_list.json.bak` restores the pre-migration state.

**Residual risk:** A shipped default might be wrong for a specific user's hardware/setup, and the user might not notice the warning. This is acceptable — the warning is printed, `config show` shows sources, and the user can always override.

### Is it appropriate to expand model_list.json?

**Yes.** `model_list.json` already carries per-model defaults:
- `default_context_length` (per entry)
- `max_prefill_len` (per entry)
- `details.think` (per entry)

Adding a top-level `defaults` block (app-wide defaults) and `settings_schema` (setting definitions) is a **natural extension** of its existing role. The file is already the registry of "what models exist and how they should be configured by default." Settings are just more configuration.

The alternative — a separate `defaults.json` — would mean two shipped files to keep in sync, two lookups at runtime, and more complexity in packaging. Not worth it.

### Response to abondis (Oct 19)

*"If there is a conflict with a user setting it can warn and offer to update the users config?"*

**Implemented as §2.6 — Shipped-Default Change Detection.** When shipped defaults change between releases and the user has overrides, we warn on load and offer `oflm config review` to reconcile. The user can keep their override, remove it (use the new shipped default), or skip. Non-interactive modes (`--accept-shipped`, `--keep-all`) are available for scripting.
