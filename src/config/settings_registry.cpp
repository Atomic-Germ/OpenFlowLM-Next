/// \file settings_registry.cpp
/// \brief Settings registry implementation — populates the registry and generates schema/help
#include "config/settings_registry.hpp"
#include "settings_registry_generated.hpp"
#include <algorithm>
#include <sstream>
#include <functional>

namespace oflm::config {

std::string scope_to_string(SettingScope scope) {
    switch (scope) {
        case SettingScope::GLOBAL: return "global";
        case SettingScope::PER_MODEL: return "per_model";
        case SettingScope::BOTH: return "both";
    }
    return "unknown";
}

std::string type_to_string(SettingType type) {
    switch (type) {
        case SettingType::ENUM: return "enum";
        case SettingType::INT: return "int";
        case SettingType::BOOL: return "bool";
        case SettingType::STRING: return "string";
    }
    return "unknown";
}

std::string delivery_to_string(DeliveryMechanism d) {
    switch (d) {
        case DeliveryMechanism::DIRECT: return "direct";
        case DeliveryMechanism::ENV_VAR: return "env_var";
        case DeliveryMechanism::API_FIELD: return "api_field";
        case DeliveryMechanism::REGISTRY: return "registry";
    }
    return "unknown";
}

nlohmann::json setting_to_schema_json(const SettingDef& def) {
    nlohmann::json j;
    j["key"] = def.key;
    j["type"] = type_to_string(def.type);
    j["description"] = def.description;
    j["default"] = def.default_value;
    j["scope"] = scope_to_string(def.scope);
    j["reload_required"] = def.reload_required;
    j["category"] = def.category;
    j["delivery"] = delivery_to_string(def.delivery);
    if (!def.env_var.empty()) j["env_var"] = def.env_var;
    if (!def.cli_flag.empty()) j["cli_flag"] = def.cli_flag;
    if (!def.registry_key.empty()) j["registry_key"] = def.registry_key;
    if (def.type == SettingType::ENUM) {
        nlohmann::json opts = nlohmann::json::array();
        for (const auto& opt : def.options) {
            opts.push_back({{"value", opt.value}, {"description", opt.description}});
        }
        j["options"] = opts;
    }
    if (def.type == SettingType::INT) {
        if (def.min_value) j["min"] = *def.min_value;
        if (def.max_value) j["max"] = *def.max_value;
        if (!def.presets.empty()) {
            nlohmann::json p = nlohmann::json::array();
            for (auto v : def.presets) p.push_back(v);
            j["presets"] = p;
        }
    }
    return j;
}

// Schema version — bump when the schema format changes
const int SCHEMA_VERSION = 1;

const std::vector<SettingDef>& settings_registry() {
    return build_registry();
}

std::optional<SettingDef> find_setting(const std::string& key) {
    for (const auto& def : settings_registry()) {
        if (def.key == key) return def;
    }
    return std::nullopt;
}

std::vector<SettingDef> settings_for_scope(SettingScope scope) {
    std::vector<SettingDef> result;
    for (const auto& def : settings_registry()) {
        if (def.scope == scope || def.scope == SettingScope::BOTH) {
            result.push_back(def);
        }
    }
    return result;
}

nlohmann::json generate_schema_json() {
    nlohmann::json schema;
    schema["version"] = 1;
    nlohmann::json settings = nlohmann::json::array();
    for (const auto& def : settings_registry()) {
        settings.push_back(setting_to_schema_json(def));
    }
    schema["settings"] = settings;
    return schema;
}

std::string compute_defaults_version(const nlohmann::json& shipped_registry) {
    // Hash: top-level "defaults" block + per-model default values from settings registry entries
    nlohmann::json hash_input = nlohmann::json::object();
    if (shipped_registry.contains("defaults")) {
        hash_input["defaults"] = shipped_registry["defaults"];
    }
    // Collect per-model defaults for settings that have a registry_key
    nlohmann::json per_model = nlohmann::json::object();
    if (shipped_registry.contains("models")) {
        for (const auto& def : settings_registry()) {
            if (def.registry_key.empty()) continue;
            for (const auto& [type, sizes] : shipped_registry["models"].items()) {
                for (const auto& [size, entry] : sizes.items()) {
                    if (entry.contains(def.registry_key)) {
                        per_model[type + ":" + size + "." + def.key] = entry[def.registry_key];
                    }
                }
            }
        }
    }
    hash_input["per_model_defaults"] = per_model;
    // Simple hash: use std::hash on the serialized JSON
    std::string serialized = hash_input.dump();
    std::hash<std::string> hasher;
    size_t h = hasher(serialized);
    std::ostringstream oss;
    oss << std::hex << h;
    return oss.str();
}

nlohmann::json generate_full_schema_json(const nlohmann::json& shipped_registry) {
    nlohmann::json result;
    result["settings_schema"] = generate_schema_json();
    result["defaults_version"] = compute_defaults_version(shipped_registry);
    return result;
}

std::vector<std::string> generate_help_lines() {
    std::vector<std::string> lines;
    lines.push_back("Config settings (managed via 'oflm config' or ~/.config/oflm/settings.json):");
    lines.push_back("");
    for (const auto& def : settings_registry()) {
        std::ostringstream oss;
        oss << "  " << def.key;
        if (!def.cli_flag.empty()) oss << " (" << def.cli_flag << ")";
        oss << " — " << def.description;
        lines.push_back(oss.str());
        if (def.type == SettingType::ENUM) {
            std::ostringstream opts;
            opts << "    Options:";
            for (const auto& opt : def.options) {
                opts << " " << opt.value;
            }
            lines.push_back(opts.str());
        }
        if (!def.default_value.empty()) {
            lines.push_back("    Default: " + def.default_value);
        }
        lines.push_back("");
    }
    return lines;
}

std::vector<std::string> get_all_setting_keys() {
    std::vector<std::string> keys;
    for (const auto& def : settings_registry()) {
        keys.push_back(def.key);
    }
    return keys;
}

nlohmann::json convert_type(const std::string& value, SettingType type) {
    switch (type) {
        case SettingType::INT:
            try { return nlohmann::json(std::stol(value)); }
            catch (...) { return nlohmann::json(value); }
        case SettingType::BOOL:
            if (value == "true" || value == "1" || value == "yes" || value == "on") return nlohmann::json(true);
            if (value == "false" || value == "0" || value == "no" || value == "off") return nlohmann::json(false);
            return nlohmann::json(value);
        case SettingType::ENUM:
        case SettingType::STRING:
        default:
            return nlohmann::json(value);
    }
}

} // namespace oflm::config
