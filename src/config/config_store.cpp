/// \file config_store.cpp
/// \brief Config store implementation — merge, save, lock, helpers
/// \note Migration and load_effective_config are in Task 07.
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include "utils/utils.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#endif

namespace oflm::config {

// ============================================================================
// deep_merge
// ============================================================================

nlohmann::json deep_merge(const nlohmann::json& base, const nlohmann::json& overlay) {
    if (!overlay.is_object()) {
        // Non-object overlay replaces entirely (arrays, strings, etc.)
        return overlay;
    }
    if (!base.is_object()) {
        return overlay;
    }

    nlohmann::json result = base;
    for (auto it = overlay.begin(); it != overlay.end(); ++it) {
        const std::string& key = it.key();
        const auto& overlay_val = it.value();

        if (result.contains(key)) {
            if (result[key].is_object() && overlay_val.is_object()) {
                // Recursively merge objects
                result[key] = deep_merge(result[key], overlay_val);
            } else if (overlay_val.is_null()) {
                // Null in overlay means "don't overwrite" — keep base value
                // (This allows explicit null to mean "use default" in some contexts)
            } else {
                // Arrays, strings, numbers, bools: overlay replaces
                result[key] = overlay_val;
            }
        } else {
            result[key] = overlay_val;
        }
    }
    return result;
}

// ============================================================================
// SettingsLock
// ============================================================================

SettingsLock::SettingsLock(const std::string& path) {
#ifdef _WIN32
    std::string lock_path = path + ".lock";
    HANDLE h = CreateFileA(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE,
                           0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    lock_handle_ = h;

    // Try to acquire lock with 2-second timeout
    for (int i = 0; i < 21; i++) {
        OVERLAPPED overlapped = {};
        if (LockFileEx(h, LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlapped)) {
            locked_ = true;
            return;
        }
        if (i < 20) Sleep(100);
    }
#else
    std::string lock_path = path + ".lock";
    lock_fd_ = open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
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
#endif
}

SettingsLock::~SettingsLock() {
#ifdef _WIN32
    if (lock_handle_) {
        OVERLAPPED overlapped = {};
        UnlockFileEx(lock_handle_, 0, 1, 0, &overlapped);
        CloseHandle(lock_handle_);
    }
#else
    if (lock_fd_ >= 0) {
        struct flock fl{};
        fl.l_type = F_UNLCK;
        fl.l_whence = SEEK_SET;
        fcntl(lock_fd_, F_SETLK, &fl);
        close(lock_fd_);
    }
#endif
}

// ============================================================================
// save_user_settings (atomic write)
// ============================================================================

void save_user_settings(const nlohmann::json& settings, const std::string& path) {
    // Write to temp file in same directory (same filesystem for atomic rename)
    std::filesystem::path target(path);
    std::string temp_path = path + ".tmp." + std::to_string(
#ifdef _WIN32
        GetCurrentProcessId()
#else
        getpid()
#endif
    );

    {
        std::ofstream out(temp_path, std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open temp file for writing: " + temp_path);
        }
        out << settings.dump(2) << std::endl;
        out.flush();
        if (!out.good()) {
            out.close();
            std::filesystem::remove(temp_path);
            throw std::runtime_error("Failed to write settings to: " + temp_path);
        }
    }

    // Atomic rename (same filesystem)
    std::error_code ec;
    std::filesystem::rename(temp_path, path, ec);
    if (ec) {
        // Rename failed — try copy + remove as fallback
        std::filesystem::copy_file(temp_path, path, std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(temp_path);
        if (ec) {
            throw std::runtime_error("Failed to save settings to: " + path + " (" + ec.message() + ")");
        }
    }
}

// ============================================================================
// split_tag
// ============================================================================

std::pair<std::string, std::string> split_tag(const std::string& tag) {
    auto pos = tag.find(':');
    if (pos == std::string::npos) {
        return {tag, ""};
    }
    return {tag.substr(0, pos), tag.substr(pos + 1)};
}

// ============================================================================
// get_merged_model_entry
// ============================================================================

nlohmann::json get_merged_model_entry(const EffectiveConfig& cfg, const std::string& tag) {
    auto [type, size] = split_tag(tag);

    // Start with shipped entry
    nlohmann::json entry;
    if (cfg.shipped_registry.contains("models") &&
        cfg.shipped_registry["models"].contains(type) &&
        cfg.shipped_registry["models"][type].contains(size)) {
        entry = cfg.shipped_registry["models"][type][size];
    }

    // Deep-merge user settings overlay (from settings.<type>.<size>)
    if (cfg.user_settings.contains("settings") &&
        cfg.user_settings["settings"].contains(type) &&
        cfg.user_settings["settings"][type].contains(size)) {
        entry = deep_merge(entry, cfg.user_settings["settings"][type][size]);
    }

    // User-added model (from models.<type>.<size>) — replaces entirely if not in shipped
    if (entry.is_null() || entry.empty()) {
        if (cfg.user_settings.contains("models") &&
            cfg.user_settings["models"].contains(type) &&
            cfg.user_settings["models"][type].contains(size)) {
            entry = cfg.user_settings["models"][type][size];
        }
    }

    return entry;
}

// ============================================================================
// STUBS — implemented in Task 07
// ============================================================================

EffectiveConfig load_effective_config() {
    throw std::runtime_error("load_effective_config not implemented — see Task 07");
}

std::vector<std::string> migrate_legacy_user_registry(
    const std::string& legacy_path,
    const nlohmann::json& shipped_registry,
    const std::string& settings_path)
{
    throw std::runtime_error("migrate_legacy_user_registry not implemented — see Task 07");
}

int reconcile_settings(const std::string& settings_path, const std::string& mode) {
    throw std::runtime_error("reconcile_settings not implemented — see Task 07");
}

} // namespace oflm::config
