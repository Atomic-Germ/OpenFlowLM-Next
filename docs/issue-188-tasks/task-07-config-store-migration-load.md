# Task 07: Config Store — Migration, Load, Reconcile

**Phase:** 2 (Config Store)
**Depends on:** Task 06
**Files to modify:** `src/config/config_store.cpp` (replace stubs with real implementations)
**Estimated complexity:** Medium (~200 lines)

## Context

This task implements the three most important functions in the config store:
1. `load_effective_config()` — loads both files, merges, migrates, detects default changes
2. `migrate_legacy_user_registry()` — three-way migration from old full-copy model_list.json
3. `reconcile_settings()` — reconcile user overrides with changed shipped defaults

Read parent plan **§Phase 2** and **§2.6 (Shipped-Default Change Detection)** and **§5 (Migration Path)** in `docs/issue-188-implementation-plan.md`.

## What to do

Replace the three stub functions in `src/config/config_store.cpp` with real implementations.

## Code to write

Replace the stub section at the bottom of `src/config/config_store.cpp` with:

```cpp
// ============================================================================
// migrate_legacy_user_registry (three-way migration)
// ============================================================================

std::vector<std::string> migrate_legacy_user_registry(
    const std::string& legacy_path,
    const nlohmann::json& shipped_registry,
    const std::string& settings_path)
{
    std::vector<std::string> log;

    // Load legacy file
    std::ifstream legacy_file(legacy_path);
    if (!legacy_file.is_open()) {
        log.push_back("Legacy file not found: " + legacy_path);
        return log;
    }
    nlohmann::json legacy;
    try {
        legacy = nlohmann::json::parse(legacy_file);
    } catch (const std::exception& e) {
        log.push_back("Failed to parse legacy file: " + std::string(e.what()));
        return log;
    }

    // Build the new settings.json structure
    nlohmann::json settings;
    settings["version"] = 1;
    settings["defaults"] = nlohmann::json::object();
    settings["settings"] = nlohmann::json::object();
    settings["models"] = nlohmann::json::object();

    // Record the current shipped defaults version
    if (shipped_registry.contains("defaults_version")) {
        settings["defaults_version"] = shipped_registry["defaults_version"];
    } else {
        settings["defaults_version"] = compute_defaults_version(shipped_registry);
    }

    int user_models = 0;
    int modified_entries = 0;
    int stale_entries = 0;

    // Process each model in the legacy file
    if (legacy.contains("models")) {
        for (const auto& [type, sizes] : legacy["models"].items()) {
            for (const auto& [size, entry] : sizes.items()) {
                std::string tag = type + ":" + size;

                // Check if this tag exists in the shipped registry
                bool in_shipped = shipped_registry.contains("models") &&
                                  shipped_registry["models"].contains(type) &&
                                  shipped_registry["models"][type].contains(size);

                if (!in_shipped) {
                    // Case 1: User-added model → settings.json models
                    settings["models"][type][size] = entry;
                    user_models++;
                    log.push_back("  Carried over user-added model: " + tag);
                } else {
                    // Tag exists in shipped — compare entries
                    const auto& shipped_entry = shipped_registry["models"][type][size];
                    if (entry == shipped_entry) {
                        // Case 2: Identical → stale copy, drop silently
                        stale_entries++;
                    } else {
                        // Case 3: Different → preserve as override in settings section
                        // Only carry over the setting-related keys (not url, files, etc.)
                        nlohmann::json override_entry = nlohmann::json::object();
                        for (const auto& def : settings_registry()) {
                            if (def.registry_key.empty()) continue;
                            // Check if the legacy entry has this key (possibly nested)
                            // For simplicity, check top-level and details.* keys
                            if (def.registry_key.find('.') != std::string::npos) {
                                // Nested key like "details.think"
                                auto parts = split_tag(def.registry_key);
                                // Actually split on '.' not ':'
                                auto dot_pos = def.registry_key.find('.');
                                std::string parent = def.registry_key.substr(0, dot_pos);
                                std::string child = def.registry_key.substr(dot_pos + 1);
                                if (entry.contains(parent) && entry[parent].contains(child)) {
                                    override_entry[def.key] = entry[parent][child];
                                }
                            } else if (entry.contains(def.registry_key)) {
                                override_entry[def.key] = entry[def.registry_key];
                            }
                        }
                        if (!override_entry.empty()) {
                            settings["settings"][type][size] = override_entry;
                            modified_entries++;
                            log.push_back("  Carried over modified entry as override: " + tag);
                        } else {
                            stale_entries++;
                        }
                    }
                }
            }
        }
    }

    // Migrate top-level custom keys to defaults (skip known non-default keys)
    static const std::set<std::string> skip_keys = {
        "model_path", "hf_owner", "version", "defaults", "settings_schema",
        "defaults_version", "models"
    };
    if (legacy.contains("defaults")) {
        for (auto it = legacy["defaults"].begin(); it != legacy["defaults"].end(); ++it) {
            settings["defaults"][it.key()] = it.value();
        }
    }

    // Save settings.json (with lock)
    {
        SettingsLock lock(settings_path);
        if (!lock.locked()) {
            log.push_back("WARNING: Could not acquire settings lock, saving anyway");
        }
        save_user_settings(settings, settings_path);
    }

    // Rename legacy file to .bak
    std::string bak_path = legacy_path + ".bak";
    std::error_code ec;
    std::filesystem::rename(legacy_path, bak_path, ec);
    if (ec) {
        log.push_back("WARNING: Could not rename " + legacy_path + " to " + bak_path);
    } else {
        log.push_back("  Backup saved to " + bak_path);
    }

    // Summary
    std::ostringstream summary;
    summary << "  Migrated " << user_models << " user-added models, "
            << modified_entries << " modified entries (preserved as overrides), "
            << stale_entries << " stale entries (dropped)";
    log.push_back(summary.str());

    return log;
}

// ============================================================================
// load_effective_config
// ============================================================================

EffectiveConfig load_effective_config() {
    EffectiveConfig cfg;

    // 1. Find and load shipped registry
    cfg.shipped_path = utils::find_shipped_registry();
    {
        std::ifstream file(cfg.shipped_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open shipped registry: " + cfg.shipped_path);
        }
        try {
            cfg.shipped_registry = nlohmann::json::parse(file);
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Shipped registry is not valid JSON: " + cfg.shipped_path +
                                     " (" + e.what() + ")");
        }
    }

    // 2. Find and load user settings (optional)
    cfg.user_settings_path = utils::find_user_settings();
    cfg.user_settings = nlohmann::json::object();

    // Check for legacy migration: old full-copy model_list.json exists, settings.json doesn't
    std::string legacy_path;
    {
        // Check user directories for old model_list.json
        std::string models_dir = utils::get_models_directory();
        std::string legacy_candidate = models_dir + "/model_list.json";
        if (std::filesystem::exists(legacy_candidate) &&
            !std::filesystem::exists(cfg.user_settings_path)) {
            legacy_path = legacy_candidate;
        }
    }

    if (!legacy_path.empty()) {
        std::cerr << "[OFLM] Migrating config: found legacy " << legacy_path << std::endl;
        cfg.migration_log = migrate_legacy_user_registry(legacy_path, cfg.shipped_registry, cfg.user_settings_path);
        cfg.migrated = true;
        for (const auto& line : cfg.migration_log) {
            std::cerr << "[oflm]" << line << std::endl;
        }
        std::cerr << "[oflm]   Future releases will update shipped defaults automatically" << std::endl;

        // Reload user settings (now migrated)
        std::ifstream settings_file(cfg.user_settings_path);
        if (settings_file.is_open()) {
            try {
                cfg.user_settings = nlohmann::json::parse(settings_file);
            } catch (...) {
                cfg.user_settings = nlohmann::json::object();
            }
        }
    } else if (std::filesystem::exists(cfg.user_settings_path)) {
        // Load existing settings.json
        std::ifstream file(cfg.user_settings_path);
        try {
            cfg.user_settings = nlohmann::json::parse(file);
        } catch (const nlohmann::json::parse_error& e) {
            // H3 fix: friendly error, continue with shipped only
            std::cerr << "[OFLM] ERROR: " << cfg.user_settings_path << " is not valid JSON:" << std::endl;
            std::cerr << "[OFLM]   " << e.what() << std::endl;
            std::cerr << "[OFLM]   Fix the file manually, or move it aside to start fresh:" << std::endl;
            std::cerr << "[OFLM]     mv " << cfg.user_settings_path << " " << cfg.user_settings_path << ".broken" << std::endl;
            std::cerr << "[OFLM]   Continuing with shipped defaults only." << std::endl;
            cfg.user_settings = nlohmann::json::object();
        }
    }

    // 3. Deep-merge: start with shipped, apply user settings overlays, add user models
    cfg.merged = cfg.shipped_registry;

    // Merge defaults
    if (cfg.user_settings.contains("defaults")) {
        if (!cfg.merged.contains("defaults")) cfg.merged["defaults"] = nlohmann::json::object();
        cfg.merged["defaults"] = deep_merge(cfg.merged["defaults"], cfg.user_settings["defaults"]);
    }

    // Merge per-model settings overlays
    if (cfg.user_settings.contains("settings")) {
        for (const auto& [type, sizes] : cfg.user_settings["settings"].items()) {
            for (const auto& [size, overlay] : sizes.items()) {
                if (!cfg.merged["models"].contains(type)) cfg.merged["models"][type] = nlohmann::json::object();
                if (!cfg.merged["models"][type].contains(size)) cfg.merged["models"][type][size] = nlohmann::json::object();
                cfg.merged["models"][type][size] = deep_merge(cfg.merged["models"][type][size], overlay);
            }
        }
    }

    // Add user-added models (complete entries, not merged)
    if (cfg.user_settings.contains("models")) {
        for (const auto& [type, sizes] : cfg.user_settings["models"].items()) {
            for (const auto& [size, entry] : sizes.items()) {
                if (!cfg.merged["models"].contains(type)) cfg.merged["models"][type] = nlohmann::json::object();
                cfg.merged["models"][type][size] = entry;
            }
        }
    }

    // 4. Shipped-default change detection (§2.6)
    cfg.shipped_defaults_version = cfg.shipped_registry.value("defaults_version",
        compute_defaults_version(cfg.shipped_registry));
    cfg.user_defaults_version = cfg.user_settings.value("defaults_version", std::string());

    if (!cfg.user_defaults_version.empty() && cfg.user_defaults_version != cfg.shipped_defaults_version) {
        cfg.defaults_changed = true;

        // Find affected overrides: user settings that override shipped defaults
        // Global defaults
        if (cfg.user_settings.contains("defaults")) {
            for (auto it = cfg.user_settings["defaults"].begin(); it != cfg.user_settings["defaults"].end(); ++it) {
                std::string shipped_val;
                if (cfg.shipped_registry.contains("defaults") &&
                    cfg.shipped_registry["defaults"].contains(it.key())) {
                    shipped_val = cfg.shipped_registry["defaults"][it.key()].dump();
                }
                std::string user_val = it.value().dump();
                std::ostringstream oss;
                oss << "  - " << it.key() << " (global): you have " << user_val
                    << ", shipped default is " << (shipped_val.empty() ? "(none)" : shipped_val);
                if (shipped_val == user_val) {
                    oss << " [redundant]";
                }
                cfg.affected_overrides.push_back(oss.str());
            }
        }
        // Per-model settings
        if (cfg.user_settings.contains("settings")) {
            for (const auto& [type, sizes] : cfg.user_settings["settings"].items()) {
                for (const auto& [size, settings] : sizes.items()) {
                    for (auto it = settings.begin(); it != settings.end(); ++it) {
                        std::ostringstream oss;
                        oss << "  - " << it.key() << " for " << type << ":" << size
                            << ": you have " << it.value().dump();
                        cfg.affected_overrides.push_back(oss.str());
                    }
                }
            }
        }

        // Print warning
        std::cerr << "[OFLM] Shipped defaults changed in this release (was "
                  << cfg.user_defaults_version << ", now " << cfg.shipped_defaults_version << ")." << std::endl;
        if (!cfg.affected_overrides.empty()) {
            std::cerr << "[OFLM] Your settings override these defaults:" << std::endl;
            for (const auto& line : cfg.affected_overrides) {
                std::cerr << "[OFLM]" << line << std::endl;
            }
        }
        std::cerr << "[OFLM]" << std::endl;
        std::cerr << "[OFLM] See all effective values:  oflm config show" << std::endl;
        std::cerr << "[OFLM] Reconcile your overrides:  oflm config review" << std::endl;
        std::cerr << "[OFLM]   (or edit " << cfg.user_settings_path << " directly)" << std::endl;
    } else if (cfg.user_defaults_version.empty() && std::filesystem::exists(cfg.user_settings_path)) {
        // First run with this feature — record current version silently
        // (Don't warn on first upgrade to avoid noise)
        nlohmann::json updated = cfg.user_settings;
        updated["defaults_version"] = cfg.shipped_defaults_version;
        SettingsLock lock(cfg.user_settings_path);
        if (lock.locked()) {
            save_user_settings(updated, cfg.user_settings_path);
        }
    }

    // 5. Schema version mismatch check (H2 fix)
    if (cfg.shipped_registry.contains("settings_schema")) {
        int shipped_schema_version = cfg.shipped_registry["settings_schema"].value("version", 0);
        // Binary schema version comes from generate_schema_json
        int binary_schema_version = generate_schema_json().value("version", 0);
        if (shipped_schema_version > binary_schema_version) {
            // Downgrade scenario: shipped schema is newer than binary
            std::cerr << "[OFLM] WARNING: model_list.json uses settings schema v"
                      << shipped_schema_version << " but this binary supports v"
                      << binary_schema_version << "." << std::endl;
            // List unknown keys
            std::set<std::string> known_keys;
            for (const auto& def : settings_registry()) known_keys.insert(def.key);
            std::vector<std::string> unknown;
            if (cfg.shipped_registry["settings_schema"].contains("settings")) {
                for (const auto& s : cfg.shipped_registry["settings_schema"]["settings"]) {
                    if (s.contains("key") && !known_keys.count(s["key"].get<std::string>())) {
                        unknown.push_back(s["key"].get<std::string>());
                    }
                }
            }
            if (!unknown.empty()) {
                std::cerr << "[OFLM]   Unknown settings will be ignored:";
                for (const auto& k : unknown) std::cerr << " " << k;
                std::cerr << std::endl;
            }
            std::cerr << "[OFLM]   Upgrade oflm or remove these keys from " << cfg.user_settings_path << std::endl;
        }
    }

    return cfg;
}

// ============================================================================
// reconcile_settings
// ============================================================================

int reconcile_settings(const std::string& settings_path, const std::string& mode) {
    // Load current state
    auto cfg = load_effective_config();
    if (!cfg.defaults_changed) {
        std::cout << "No shipped-default changes detected. Nothing to reconcile." << std::endl;
        return 0;
    }

    int removed = 0;

    if (mode == "keep-all") {
        // Just update defaults_version, no changes
        nlohmann::json updated = cfg.user_settings;
        updated["defaults_version"] = cfg.shipped_defaults_version;
        SettingsLock lock(settings_path);
        if (lock.locked()) {
            save_user_settings(updated, settings_path);
        }
        std::cout << "Kept all overrides. defaults_version updated." << std::endl;
        return 0;
    }

    if (mode == "accept-shipped") {
        // Remove all overrides that match shipped defaults (redundant ones)
        nlohmann::json updated = cfg.user_settings;

        // Global defaults
        if (updated.contains("defaults") && cfg.shipped_registry.contains("defaults")) {
            auto& user_defaults = updated["defaults"];
            auto& shipped_defaults = cfg.shipped_registry["defaults"];
            std::vector<std::string> to_remove;
            for (auto it = user_defaults.begin(); it != user_defaults.end(); ++it) {
                if (shipped_defaults.contains(it.key()) && shipped_defaults[it.key()] == it.value()) {
                    to_remove.push_back(it.key());
                }
            }
            for (const auto& key : to_remove) {
                user_defaults.erase(key);
                removed++;
            }
        }

        updated["defaults_version"] = cfg.shipped_defaults_version;
        SettingsLock lock(settings_path);
        if (lock.locked()) {
            save_user_settings(updated, settings_path);
        }
        std::cout << "Removed " << removed << " redundant override(s)." << std::endl;
        return removed;
    }

    if (mode == "interactive") {
        // Interactive mode — implemented in Task 17 (Phase 5)
        // For now, print what would happen and ask user to use non-interactive modes
        std::cerr << "Interactive review not yet available." << std::endl;
        std::cerr << "Use 'oflm config review --accept-shipped' to remove redundant overrides," << std::endl;
        std::cerr << "or 'oflm config review --keep-all' to keep all and update version." << std::endl;
        return -1;
    }

    std::cerr << "Unknown reconcile mode: " << mode << std::endl;
    return -1;
}

} // namespace oflm::config
```

## Test

```bash
# Build
cmake --build --preset linux-default --target oflm 2>&1 | tail -10

# Manual test: create a fake legacy registry and verify migration
mkdir -p /tmp/oflm_test
cat > /tmp/oflm_test/model_list.json << 'EOF'
{
  "model_path": "models",
  "hf_owner": "Test",
  "models": {
    "my-test-model": {
      "1b": {
        "name": "my-test-model-1b",
        "url": "https://example.com",
        "default_context_length": 4096
      }
    }
  }
}
EOF

# Run with OFLM_SETTINGS_PATH pointing to a temp location
# (Full automated tests come in Task 08)
```

## Success criteria

- [ ] `migrate_legacy_user_registry()` implemented (three-way: user models → models, modified → settings, identical → drop)
- [ ] `load_effective_config()` implemented (load both files, merge, migrate, detect changes, schema version check)
- [ ] `reconcile_settings()` implemented (modes: "accept-shipped", "keep-all"; "interactive" is a stub for Task 17)
- [ ] Corrupted settings.json → friendly error, continues with shipped
- [ ] Legacy migration creates `.bak` file
- [ ] Shipped-default change detection prints warning with affected overrides
- [ ] Schema version mismatch (downgrade) lists unknown keys
- [ ] Project compiles without errors
