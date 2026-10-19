/// \file config_cmd.cpp
/// \brief Implementation of `oflm config` CLI commands
#include "config/config_cmd.hpp"
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include "config/settings_resolver.hpp"
#include "config/config_ui.hpp"
#include "utils/utils.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace oflm::config {

// Source tag strings (consistent with plan §Phase 4)
// The source tag beside a value. The user cares about one thing first: is
// this the value I chose, or the one the application ships? So the file the
// user owns reads (custom) and the shipped registry reads (global). The
// runtime overrides keep their own tags, because they answer a different
// question: why is this not the value I set?
const char* source_tag(SettingValue::Source s) {
    switch (s) {
        case SettingValue::SHIPPED: return "(global)";
        case SettingValue::USER: return "(custom)";
        case SettingValue::ENV: return "(env)";
        case SettingValue::CLI: return "(cli)";
        case SettingValue::API: return "(api)";
    }
    return "(?)";
}

// Validate a value against a setting definition
static bool validate_value(const SettingDef& def, const std::string& value, std::string& error) {
    switch (def.type) {
        case SettingType::ENUM:
            for (const auto& opt : def.options) {
                if (opt.value == value) return true;
            }
            error = "Valid options:";
            for (const auto& opt : def.options) error += " " + opt.value;
            return false;
        case SettingType::INT: {
            try {
                long v = std::stol(value);
                if (def.min_value && v < *def.min_value) {
                    error = "Value must be >= " + std::to_string(*def.min_value);
                    return false;
                }
                if (def.max_value && v > *def.max_value) {
                    error = "Value must be <= " + std::to_string(*def.max_value);
                    return false;
                }
                return true;
            } catch (...) {
                error = "Value must be an integer";
                return false;
            }
        }
        case SettingType::BOOL:
            return value == "true" || value == "false" || value == "0" || value == "1" ||
                   value == "yes" || value == "no" || value == "on" || value == "off";
        case SettingType::STRING:
            return true;
    }
    return false;
}

// H5 fix: warn on unknown keys, allow with confirmation
static bool check_unknown_key(const std::string& key) {
    if (find_setting(key)) return true; // Known key
    std::cerr << "[OFLM] WARNING: '" << key << "' is not a known setting in this version of oflm." << std::endl;
    std::cerr << "[OFLM]   It will be stored but may not have any effect." << std::endl;
    std::cerr << "[OFLM]   Known settings:";
    for (const auto& k : get_all_setting_keys()) std::cerr << " " << k;
    std::cerr << std::endl;
    std::cerr << "Set anyway? [y/N] ";
    std::string response;
    std::getline(std::cin, response);
    return response == "y" || response == "Y" || response == "yes";
}

// Get the settings path for a given model tag
static std::string get_settings_path_for_model(const std::string& model_tag, const EffectiveConfig& cfg) {
    auto [type, size] = split_tag(model_tag);
    
    // Check if this is a user-added model (in settings.json models section)
    if (cfg.user_settings.contains("models") &&
        cfg.user_settings["models"].contains(type) &&
        cfg.user_settings["models"][type].contains(size)) {
        return cfg.user_settings_path;
    }
    
    // For per-model settings, use the settings section
    return cfg.user_settings_path;
}

// Set a configuration value
static int config_set(const program_args_t& args) {
    if (args.config_value.empty()) {
        auto def = find_setting(args.config_key);
        if (def && def->type == SettingType::ENUM) {
            std::vector<PickerItem> items;
            for (const auto& opt : def->options) {
                items.push_back(PickerItem{opt.value, opt.description});
            }
            size_t idx = arrow_picker("Select value for " + args.config_key + ":", items);
            if (idx == SIZE_MAX) return 1; // cancelled
            
            // Use the selected value
            EffectiveConfig cfg = load_effective_config();
            SettingsLock lock(cfg.user_settings_path);
            if (!lock.locked()) {
                std::cerr << "Error: Could not acquire lock on settings file" << std::endl;
                return 1;
            }
            
            nlohmann::json settings_json = cfg.user_settings;
            if (settings_json.is_null()) {
                settings_json = nlohmann::json::object();
            }
            
            if (args.config_model.empty()) {
                if (!settings_json.contains("defaults")) {
                    settings_json["defaults"] = nlohmann::json::object();
                }
                settings_json["defaults"][args.config_key] = items[idx].label;
            } else {
                auto model_parts = split_tag(args.config_model);
                std::string type = model_parts.first;
                std::string size = model_parts.second;
                if (!settings_json.contains("settings")) {
                    settings_json["settings"] = nlohmann::json::object();
                }
                if (!settings_json["settings"].contains(type)) {
                    settings_json["settings"][type] = nlohmann::json::object();
                }
                if (!settings_json["settings"][type].contains(size)) {
                    settings_json["settings"][type][size] = nlohmann::json::object();
                }
                settings_json["settings"][type][size][args.config_key] = items[idx].label;
            }
            
            if (cfg.shipped_registry.contains("defaults") &&
                cfg.shipped_registry["defaults"].contains("defaults_version")) {
                settings_json["defaults_version"] = cfg.shipped_registry["defaults"]["defaults_version"];
            }
            
            save_user_settings(settings_json, cfg.user_settings_path);
            std::cout << "Set '" << args.config_key << "' = '" << items[idx].label << "'" << std::endl;
            return 0;
        }
        std::cout << "Setting '" << args.config_key << "' is not an ENUM type or not found" << std::endl;
        return 1;
    }
    
    EffectiveConfig cfg = load_effective_config();
    
    // Check if key is known (H5 fix)
    bool key_known = check_unknown_key(args.config_key);
    if (!key_known) {
        return 1;
    }
    
    // Find the setting definition
    auto def_opt = find_setting(args.config_key);
    if (!def_opt) {
        std::cerr << "Error: Setting '" << args.config_key << "' not found in registry" << std::endl;
        return 1;
    }
    const SettingDef& def = *def_opt;
    
    // Validate the value
    std::string validation_error;
    if (!validate_value(def, args.config_value, validation_error)) {
        std::cerr << "Error: Invalid value for '" << args.config_key << "': " << validation_error << std::endl;
        return 1;
    }
    
    // Convert value to proper type for JSON
    nlohmann::json value_json = convert_type(args.config_value, def.type);
    
    SettingsLock lock(cfg.user_settings_path);
    if (!lock.locked()) {
        std::cerr << "Error: Could not acquire lock on settings file" << std::endl;
        return 1;
    }
    
    // Load existing settings or create empty
    nlohmann::json settings = cfg.user_settings;
    if (settings.is_null()) {
        settings = nlohmann::json::object();
    }
    
    // Set the value based on scope
    bool app_wide = false;
    if (args.config_model.empty()) {
        // Global setting: defaults.<key>
        app_wide = true;
        if (!settings.contains("defaults")) {
            settings["defaults"] = nlohmann::json::object();
        }
        settings["defaults"][args.config_key] = value_json;
        
        // Update defaults_version
        if (cfg.shipped_registry.contains("defaults") &&
            cfg.shipped_registry["defaults"].contains("defaults_version")) {
            settings["defaults_version"] = cfg.shipped_registry["defaults"]["defaults_version"];
        }
    } else {
        // Per-model setting: settings.<type>.<size>.<key>
        auto [type, size] = split_tag(args.config_model);

        // A global setting that this model does not override for itself is
        // an app-wide setting: writing it to the model would leave every
        // other model on the old value. Write it to the defaults block
        // instead, so it reaches all of them.
        bool is_global = def.scope == SettingScope::GLOBAL;
        bool model_overrides_it = cfg.user_settings.contains("settings") &&
                                  cfg.user_settings["settings"].contains(type) &&
                                  cfg.user_settings["settings"][type].contains(size) &&
                                  cfg.user_settings["settings"][type][size].contains(args.config_key);

        if (is_global && !model_overrides_it) {
            app_wide = true;
            if (!settings.contains("defaults")) {
                settings["defaults"] = nlohmann::json::object();
            }
            settings["defaults"][args.config_key] = value_json;
        } else {
            if (!settings.contains("settings")) {
                settings["settings"] = nlohmann::json::object();
            }
            if (!settings["settings"].contains(type)) {
                settings["settings"][type] = nlohmann::json::object();
            }
            if (!settings["settings"][type].contains(size)) {
                settings["settings"][type][size] = nlohmann::json::object();
            }

            settings["settings"][type][size][args.config_key] = value_json;
        }

        // Update defaults_version
        if (cfg.shipped_registry.contains("defaults") &&
            cfg.shipped_registry["defaults"].contains("defaults_version")) {
            settings["defaults_version"] = cfg.shipped_registry["defaults"]["defaults_version"];
        }
    }
    
    // Save the settings
    try {
        save_user_settings(settings, cfg.user_settings_path);
        std::cout << "Set '" << args.config_key << "' = '" << args.config_value << "'";
        if (app_wide) {
            std::cout << " (app-wide: every model now uses this)";
        }
        std::cout << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error saving settings: " << e.what() << std::endl;
        return 1;
    }
}

// Unset a configuration value
static int config_unset(const program_args_t& args) {
    EffectiveConfig cfg = load_effective_config();
    
    SettingsLock lock(cfg.user_settings_path);
    if (!lock.locked()) {
        std::cerr << "Error: Could not acquire lock on settings file" << std::endl;
        return 1;
    }
    
    // Load existing settings or create empty
    nlohmann::json settings = cfg.user_settings;
    if (settings.is_null()) {
        std::cout << "No user settings to modify" << std::endl;
        return 0;
    }
    
    bool removed = false;
    
    if (args.config_model.empty()) {
        // Global setting: remove from defaults
        if (settings.contains("defaults") &&
            settings["defaults"].contains(args.config_key)) {
            settings["defaults"].erase(args.config_key);
            removed = true;
        }
    } else {
        // Per-model setting: remove from settings.<type>.<size>
        auto [type, size] = split_tag(args.config_model);
        
        if (settings.contains("settings") &&
            settings["settings"].contains(type) &&
            settings["settings"][type].contains(size) &&
            settings["settings"][type][size].contains(args.config_key)) {
            settings["settings"][type][size].erase(args.config_key);
            removed = true;
            
            // If the model entry is now empty, remove it
            if (settings["settings"][type][size].empty()) {
                settings["settings"][type].erase(size);
                // If the type entry is now empty, remove it
                if (settings["settings"][type].empty()) {
                    settings["settings"].erase(type);
                }
            }
        }
    }
    
    if (removed) {
        // Update defaults_version
        if (cfg.shipped_registry.contains("defaults") &&
            cfg.shipped_registry["defaults"].contains("defaults_version")) {
            settings["defaults_version"] = cfg.shipped_registry["defaults"]["defaults_version"];
        }
        
        // Save the settings
        try {
            save_user_settings(settings, cfg.user_settings_path);
            std::cout << "Unset '" << args.config_key << "'" << std::endl;
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error saving settings: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "Setting '" << args.config_key << "' not found in user settings" << std::endl;
        return 0;
    }
}

// Show configuration
int config_show(const program_args_t& args) {
    EffectiveConfig cfg = load_effective_config();
    
    if (args.json_output) {
        // Output JSON format
        nlohmann::json output;
        
        if (!args.config_model.empty()) {
            // Show per-model settings
            auto model_entry = get_merged_model_entry(cfg, args.config_model);
            if (model_entry.is_null() || model_entry.empty()) {
                std::cout << "{}" << std::endl;
                return 0;
            }
            
            // Convert to flat key-value with sources
            nlohmann::json model_settings;
            for (const auto& def : settings_registry()) {
                // Everything that bears on this model: the per-model settings
                // and the app-wide ones. The app-wide settings are listed
                // because the guided flow offers them here too, and a view
                // that shows fewer than the menu is confusing.
                auto value_opt = get_setting_value(cfg, args.config_model, def.key, args);
                if (value_opt) {
                    model_settings[def.key] = {
                        {"value", value_opt->value},
                        {"source", source_tag(value_opt->source)}
                    };
                }
            }
            
            output = model_settings;
        } else {
            // Show global settings
            nlohmann::json global_settings;
            for (const auto& def : settings_registry()) {
                if (def.scope != SettingScope::GLOBAL && def.scope != SettingScope::BOTH) {
                    continue;
                }
                
                auto value_opt = get_setting_value(cfg, "", def.key, args);
                if (value_opt) {
                    global_settings[def.key] = {
                        {"value", value_opt->value},
                        {"source", source_tag(value_opt->source)}
                    };
                }
            }
            
            output = global_settings;
        }
        
        std::cout << output.dump(2) << std::endl;
    } else {
        // Output human-readable format
        if (!args.config_model.empty()) {
            std::cout << "Settings for model: " << args.config_model << std::endl;
            std::cout << "key                      value                      source" << std::endl;
            std::cout << "------------------------ ------------------------ ------------------------" << std::endl;
            
            bool found_any = false;
            for (const auto& def : settings_registry()) {
                // Everything that bears on this model, as in the JSON view.
                auto value_opt = get_setting_value(cfg, args.config_model, def.key, args);
                if (value_opt) {
                    std::cout << std::left << std::setw(24) << def.key << " "
                              << std::setw(26) << value_opt->value << " "
                              << source_tag(value_opt->source) << std::endl;
                    found_any = true;
                }
            }
            
            if (!found_any) {
                std::cout << "No settings found for this model" << std::endl;
            }
        } else {
            std::cout << "Global settings:" << std::endl;
            std::cout << "key                      value                      source" << std::endl;
            std::cout << "------------------------ ------------------------ ------------------------" << std::endl;
            
            bool found_any = false;
            for (const auto& def : settings_registry()) {
                if (def.scope != SettingScope::GLOBAL && def.scope != SettingScope::BOTH) {
                    continue;
                }
                
                auto value_opt = get_setting_value(cfg, "", def.key, args);
                if (value_opt) {
                    std::cout << std::left << std::setw(24) << def.key << " "
                              << std::setw(26) << value_opt->value << " "
                              << source_tag(value_opt->source) << std::endl;
                    found_any = true;
                }
            }
            
            if (!found_any) {
                std::cout << "No global settings found" << std::endl;
            }
        }
    }
    
    return 0;
}

// Edit configuration file
static int config_edit(const program_args_t& args) {
    EffectiveConfig cfg = load_effective_config();
    
    // Get editor from environment
    const char* editor_env = std::getenv("EDITOR");
#ifdef _WIN32
    const char* editor_env_win = std::getenv("EDITOR");
    if (!editor_env) editor_env = editor_env_win;
#endif
    std::string editor = editor_env ? editor_env : "vi";
    
    // Create a temporary file with the settings
    std::string temp_path = cfg.user_settings_path + ".tmp";
    
    try {
        // If settings.json doesn't exist, create an empty template
        nlohmann::json settings;
        if (!std::filesystem::exists(cfg.user_settings_path)) {
            settings = nlohmann::json::object();
        } else {
            settings = cfg.user_settings;
        }
        
        // Write to temp file
        std::ofstream out(temp_path);
        if (!out.is_open()) {
            std::cerr << "Error: Could not open temp file for editing" << std::endl;
            return 1;
        }
        out << settings.dump(2) << std::endl;
        out.close();
        
        // Build editor command
        std::string command = editor + " " + temp_path;
        
        // Execute editor
        int result = std::system(command.c_str());
        
        if (result != 0) {
            std::cerr << "Editor exited with code " << result << std::endl;
            std::filesystem::remove(temp_path);
            return 1;
        }
        
        // Read back the edited file
        std::ifstream in(temp_path);
        if (!in.is_open()) {
            std::cerr << "Error: Could not read edited file" << std::endl;
            std::filesystem::remove(temp_path);
            return 1;
        }
        
        nlohmann::json edited_settings;
        try {
            in >> edited_settings;
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid JSON in edited file: " << e.what() << std::endl;
            in.close();
            std::filesystem::remove(temp_path);
            return 1;
        }
        
        in.close();
        
        // Validate the JSON structure
        if (!edited_settings.is_object()) {
            std::cerr << "Error: Settings must be a JSON object" << std::endl;
            std::filesystem::remove(temp_path);
            return 1;
        }
        
        // Save the settings
        SettingsLock lock(cfg.user_settings_path);
        if (!lock.locked()) {
            std::cerr << "Error: Could not acquire lock on settings file" << std::endl;
            std::filesystem::remove(temp_path);
            return 1;
        }
        
        save_user_settings(edited_settings, cfg.user_settings_path);
        std::cout << "Settings saved" << std::endl;
        
        // Clean up temp file
        std::filesystem::remove(temp_path);
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error during edit: " << e.what() << std::endl;
        if (std::filesystem::exists(temp_path)) {
            std::filesystem::remove(temp_path);
        }
        return 1;
    }
}

// Generate schema
static int config_gen_schema(const program_args_t& args) {
    // The schema comes from the code, so it can be produced without a
    // registry: the packaging step runs this before the registry is in
    // place, and a missing file must not stop it. What the registry would
    // have said (the app-wide defaults, the version hash) is filled in when
    // there is one.
    nlohmann::json shipped;
    try {
        shipped = load_effective_config().shipped_registry;
    } catch (const std::exception&) {
        shipped = nlohmann::json::object();
    }
    nlohmann::json schema = generate_full_schema_json(shipped);
    std::cout << schema.dump(2) << std::endl;
    return 0;
}

// Migrate settings
static int config_migrate(const program_args_t& args) {
    EffectiveConfig cfg = load_effective_config();
    
    // Force migration check
    if (cfg.migrated) {
        std::cout << "Migration already completed in this session" << std::endl;
    } else {
        std::cout << "Migration check completed" << std::endl;
    }
    
    return 0;
}

/// Main entry point for `oflm config <action>`
int config_command_main(const program_args_t& args) {
    // The action is the positional argument after `config`. The parser leaves
    // model_tag at its default sentinel when there is none.
    const std::string& action = args.model_tag;

    if (action.empty() || action == "model-faker") {
        // No action given, open the guided flow
        return guided_flow();
    }

    if (action == "show") {
        return config_show(args);
    } else if (action == "set") {
        return config_set(args);
    } else if (action == "unset") {
        return config_unset(args);
    } else if (action == "edit") {
        return config_edit(args);
    } else if (action == "browse") {
        return browse_tui();
    } else if (action == "gen-schema") {
        return config_gen_schema(args);
    } else if (action == "migrate") {
        return config_migrate(args);
    } else if (action == "review") {
        std::string mode = "interactive";
        if (args.config_accept_shipped) mode = "accept-shipped";
        else if (args.config_keep_all) mode = "keep-all";

        if (mode == "interactive") {
            // Interactive mode not yet available (Task 17)
            std::cerr << "Interactive review not yet available." << std::endl;
            std::cerr << "Use 'oflm config review --accept-shipped' to remove redundant overrides," << std::endl;
            std::cerr << "or 'oflm config review --keep-all' to keep all and update version." << std::endl;
            return 1;
        }

        std::string settings_path = utils::find_user_settings();
        int removed = reconcile_settings(settings_path, mode);
        return removed >= 0 ? 0 : 1;
    }

    std::cerr << "Error: Unknown config action '" << action << "'" << std::endl;
    std::cerr << "Valid actions: show, set, unset, edit, gen-schema, migrate, review" << std::endl;
    return 1;
}

int config_command_main_guarded(const program_args_t& args) {
    try {
        return config_command_main(args);
    } catch (const std::exception& e) {
        // A missing or unreadable registry is a normal condition for a config
        // command, not a crash: say what went wrong and stop.
        std::cerr << "[OFLM] " << e.what() << std::endl;
        return 1;
    }
}

} // namespace oflm::config
