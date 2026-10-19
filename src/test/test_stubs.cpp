/// \file test_stubs.cpp
/// \brief Stubs for unit tests — replaces utils.cpp and config_store.cpp to avoid XRT dependency
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <nlohmann/json.hpp>

// Forward declare EffectiveConfig to avoid including config_store.hpp
// (which transitively includes XRT headers)
namespace oflm::config {
struct EffectiveConfig;
std::pair<std::string, std::string> split_tag(const std::string& tag);
nlohmann::json get_merged_model_entry(const EffectiveConfig& cfg, const std::string& tag);
nlohmann::json deep_merge(const nlohmann::json& base, const nlohmann::json& overlay);
}

// We need the full EffectiveConfig definition for get_merged_model_entry.
// Provide a minimal version that matches the real one's relevant fields.
// Actually, let's just include the real header but guard against XRT.
// The issue is utils/utils.hpp → ... → buffer.hpp → oflm_rt::ext::bo
// Solution: provide our own minimal EffectiveConfig-compatible access.

// Actually, the simplest approach: copy the implementations here.
// split_tag and get_merged_model_entry don't depend on XRT.

namespace utils {

std::string getenv_oflm(const char* oflm_name) {
    if (const char* v = std::getenv(oflm_name)) {
        if (*v) return std::string(v);
    }
    if (std::strncmp(oflm_name, "OFLM_", 5) == 0) {
        std::string legacy = std::string("FLM_") + (oflm_name + 5);
        if (const char* v = std::getenv(legacy.c_str())) {
            if (*v) return std::string(v);
        }
    }
    return std::string();
}

std::string find_shipped_registry() {
    throw std::runtime_error("stub: find_shipped_registry not available in tests");
}

std::string find_user_settings() {
    return std::string();
}

std::string get_models_directory() {
    return std::string("/tmp");
}

std::string get_executable_directory() {
    return std::string(".");
}

std::string get_user_directory() {
    return std::string("/tmp");
}

int get_server_port(int user_port) {
    return user_port > 0 ? user_port : 52625;
}

} // namespace utils

// Now provide the config functions.
// We need the real EffectiveConfig definition. Include it directly
// but avoid the XRT chain by NOT including utils/utils.hpp.
// config_store.hpp includes nlohmann/json.hpp and declares EffectiveConfig
// without any XRT dependency — the XRT comes from utils/utils.hpp which
// config_store.cpp includes, not config_store.hpp.
// So we CAN include config_store.hpp safely.

#include "config/config_store.hpp"

namespace oflm::config {

std::pair<std::string, std::string> split_tag(const std::string& tag) {
    auto pos = tag.find(':');
    if (pos == std::string::npos) {
        return {tag, ""};
    }
    return {tag.substr(0, pos), tag.substr(pos + 1)};
}

nlohmann::json deep_merge(const nlohmann::json& base, const nlohmann::json& overlay) {
    if (!overlay.is_object()) return overlay;
    if (!base.is_object()) return overlay;
    nlohmann::json result = base;
    for (auto it = overlay.begin(); it != overlay.end(); ++it) {
        const std::string& key = it.key();
        const auto& overlay_val = it.value();
        if (result.contains(key)) {
            if (result[key].is_object() && overlay_val.is_object()) {
                result[key] = deep_merge(result[key], overlay_val);
            } else if (overlay_val.is_null()) {
                // null doesn't overwrite
            } else {
                result[key] = overlay_val;
            }
        } else {
            result[key] = overlay_val;
        }
    }
    return result;
}

nlohmann::json get_merged_model_entry(const EffectiveConfig& cfg, const std::string& tag) {
    auto [type, size] = split_tag(tag);
    nlohmann::json entry;
    if (cfg.shipped_registry.contains("models") &&
        cfg.shipped_registry["models"].contains(type) &&
        cfg.shipped_registry["models"][type].contains(size)) {
        entry = cfg.shipped_registry["models"][type][size];
    }
    if (cfg.user_settings.contains("settings") &&
        cfg.user_settings["settings"].contains(type) &&
        cfg.user_settings["settings"][type].contains(size)) {
        entry = deep_merge(entry, cfg.user_settings["settings"][type][size]);
    }
    if (entry.is_null() || entry.empty()) {
        if (cfg.user_settings.contains("models") &&
            cfg.user_settings["models"].contains(type) &&
            cfg.user_settings["models"][type].contains(size)) {
            entry = cfg.user_settings["models"][type][size];
        }
    }
    return entry;
}

} // namespace oflm::config
