/// \file config_store.hpp
/// \brief Two-file config system: load, merge, save, migrate user settings
/// \note The shipped model_list.json is always read. The user's settings.json
///       is a sparse overlay that is deep-merged on top. Nothing we ship ever
///       writes to settings.json.
#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace oflm::config {

/// The result of loading and merging the two config files.
struct EffectiveConfig {
    nlohmann::json shipped_registry;   ///< Loaded shipped model_list.json
    nlohmann::json user_settings;      ///< Loaded settings.json (empty object if not found)
    nlohmann::json merged;             ///< Deep-merged result (shipped + settings overlays + user models)
    std::string shipped_path;          ///< Path to the shipped registry file
    std::string user_settings_path;    ///< Path to the user settings file
    bool migrated = false;             ///< True if legacy migration ran this session
    std::vector<std::string> migration_log;  ///< Log lines from migration

    // Shipped-default change detection (§2.6)
    bool defaults_changed = false;     ///< shipped.defaults_version != user.defaults_version
    std::string shipped_defaults_version;  ///< From shipped registry
    std::string user_defaults_version;     ///< From user settings (empty if not set)
    std::vector<std::string> affected_overrides;  ///< Human-readable list of overrides affected by default change
};

/// RAII file lock for concurrent access to settings.json.
/// POSIX: flock(). Windows: LockFileEx().
/// Holds the lock for the entire load-modify-save cycle.
class SettingsLock {
public:
    /// Acquire lock on <path>.lock. Waits up to 2 seconds.
    explicit SettingsLock(const std::string& path);
    ~SettingsLock();

    SettingsLock(const SettingsLock&) = delete;
    SettingsLock& operator=(const SettingsLock&) = delete;

    /// True if the lock was acquired.
    bool locked() const { return locked_; }

private:
    int lock_fd_ = -1;       ///< POSIX file descriptor
    void* lock_handle_ = nullptr;  ///< Windows HANDLE
    bool locked_ = false;
};

/// Load shipped registry + user settings, deep-merge, run migration if needed.
/// On corrupted settings.json: prints friendly error, continues with shipped only.
/// Also performs shipped-default change detection: compares defaults_version,
/// populates defaults_changed and affected_overrides, prints warning if changed.
EffectiveConfig load_effective_config();

/// Deep-merge overlay onto base. Objects merge recursively; arrays replace; nulls don't overwrite.
nlohmann::json deep_merge(const nlohmann::json& base, const nlohmann::json& overlay);

/// Save user settings atomically (write to temp file in same dir, then rename).
/// Caller must hold SettingsLock.
void save_user_settings(const nlohmann::json& settings, const std::string& path);

/// Reconcile user settings with current shipped defaults (§2.6).
/// Mode: "interactive" | "accept-shipped" | "keep-all"
/// Returns number of overrides removed.
/// Updates defaults_version in settings.json to current shipped version.
int reconcile_settings(const std::string& settings_path, const std::string& mode);

/// Three-way migration: legacy full-copy model_list.json → settings.json.
/// 1. Tags not in shipped → user-added models → settings.json models
/// 2. Tags in shipped, identical → dropped (stale copy)
/// 3. Tags in shipped, different → preserved as overrides in settings.json settings
/// Returns migration log lines.
std::vector<std::string> migrate_legacy_user_registry(
    const std::string& legacy_path,
    const nlohmann::json& shipped_registry,
    const std::string& settings_path);

/// Get the merged model entry for a tag (e.g. "qwen3.8:27b").
/// Returns the deep-merged entry from shipped + user settings overlay.
/// Returns empty JSON object if tag not found.
nlohmann::json get_merged_model_entry(const EffectiveConfig& cfg, const std::string& tag);

/// Split a model tag into (type, size). E.g. "qwen3.8:27b" → ("qwen3.8", "27b").
/// If no colon, returns (tag, "").
std::pair<std::string, std::string> split_tag(const std::string& tag);

} // namespace oflm::config
