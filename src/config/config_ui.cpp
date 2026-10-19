#include "config/config_ui.hpp"
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include "config/settings_resolver.hpp"
#include "config/config_cmd.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdint>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define ISATTY _isatty
#define FILENO _fileno
#else
#include <unistd.h>
#include <termios.h>
#define ISATTY isatty
#define FILENO fileno
#endif

namespace oflm::config {


UiTier detect_ui_tier() {
    if (!ISATTY(FILENO(stdin)) || !ISATTY(FILENO(stdout)))
        return UiTier::NUMBERED_MENU;

#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    if (!GetConsoleMode(hOut, &mode)) return UiTier::NUMBERED_MENU;
    if (!(mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
        if (!SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
            return UiTier::ARROW_PICKER;
    }
    return UiTier::FULL_TUI;
#else
    const char* term = getenv("TERM");
    if (!term || std::string(term) == "dumb") return UiTier::NUMBERED_MENU;

    struct termios t;
    if (tcgetattr(STDIN_FILENO, &t) != 0) return UiTier::NUMBERED_MENU;

    if (getenv("COLORTERM") ||
        std::string(term).find("xterm") != std::string::npos ||
        std::string(term).find("screen") != std::string::npos ||
        std::string(term).find("tmux") != std::string::npos)
        return UiTier::FULL_TUI;

    return UiTier::ARROW_PICKER;
#endif
}

size_t numbered_menu(const std::string& title, const std::vector<PickerItem>& items) {
    std::cout << title << ":\n";
    for (size_t i = 0; i < items.size(); i++) {
        std::cout << "  " << (i + 1) << ". " << items[i].label;
        if (!items[i].description.empty()) {
            std::cout << "  " << items[i].description;
        }
        std::cout << "\n";
    }
    std::cout << "  0. Back\n";
    std::cout << "> " << std::flush;

    std::string line;
    if (!std::getline(std::cin, line)) return SIZE_MAX;

    try {
        int choice = std::stoi(line);
        if (choice == 0) return SIZE_MAX;
        if (choice < 1 || choice > (int)items.size()) return SIZE_MAX;
        return (size_t)(choice - 1);
    } catch (...) {
        return SIZE_MAX;
    }
}

// Stubs — implemented in Tasks 16-17
size_t arrow_picker(const std::string& title, const std::vector<PickerItem>& items, size_t current) {
    if (items.empty()) return SIZE_MAX;
    if (detect_ui_tier() < UiTier::ARROW_PICKER) {
        // Fall back to numbered menu
        return numbered_menu(title, items);
    }

    size_t selected = (current != SIZE_MAX && current < items.size()) ? current : 0;

#ifdef _WIN32
    // Windows Console API implementation
    HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD orig_mode;
    GetConsoleMode(hIn, &orig_mode);
    SetConsoleMode(hIn, orig_mode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT));

    auto render = [&]() {
        // Clear screen and render
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(hOut, &csbi);
        // Simple render: print title and items
        std::cout << "\033[H\033[2J" << title << "\n";
        for (size_t i = 0; i < items.size(); i++) {
            if (i == selected) std::cout << "  > " << items[i].label;
            else std::cout << "    " << items[i].label;
            if (!items[i].description.empty()) std::cout << "  " << items[i].description;
            std::cout << "\n";
        }
        std::cout << "\n  ↑/↓ navigate  Enter select  Esc cancel\n" << std::flush;
    };

    render();
    while (true) {
        INPUT_RECORD record;
        DWORD read;
        ReadConsoleInput(hIn, &record, 1, &read);
        if (record.EventType != KEY_EVENT) continue;
        if (!record.Event.KeyEvent.bKeyDown) continue;

        if (record.Event.KeyEvent.wVirtualKeyCode == VK_UP && selected > 0) {
            selected--;
            render();
        } else if (record.Event.KeyEvent.wVirtualKeyCode == VK_DOWN && selected < items.size() - 1) {
            selected++;
            render();
        } else if (record.Event.KeyEvent.wVirtualKeyCode == VK_RETURN) {
            SetConsoleMode(hIn, orig_mode);
            return selected;
        } else if (record.Event.KeyEvent.wVirtualKeyCode == VK_ESCAPE) {
            SetConsoleMode(hIn, orig_mode);
            return SIZE_MAX;
        }
    }
#else
    // POSIX termios implementation
    struct termios orig, raw;
    tcgetattr(STDIN_FILENO, &orig);
    raw = orig;
    raw.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);

    auto render = [&]() {
        std::cout << "\033[H\033[2J" << title << "\n";
        for (size_t i = 0; i < items.size(); i++) {
            if (i == selected) std::cout << "  > " << items[i].label;
            else std::cout << "    " << items[i].label;
            if (!items[i].description.empty()) std::cout << "  " << items[i].description;
            std::cout << "\n";
        }
        std::cout << "\n  ↑/↓ navigate  Enter select  Esc cancel\n" << std::flush;
    };

    render();
    while (true) {
        char c;
        if (read(STDIN_FILENO, &c, 1) != 1) break;
        if (c == '\n' || c == '\r') {
            tcsetattr(STDIN_FILENO, TCSANOW, &orig);
            return selected;
        } else if (c == 27) { // Esc
            // Read rest of escape sequence
            char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) == 1 && seq[0] == '[') {
                if (read(STDIN_FILENO, &seq[1], 1) == 1) {
                    if (seq[1] == 'A' && selected > 0) selected--; // Up
                    else if (seq[1] == 'B' && selected < items.size() - 1) selected++; // Down
                    else if (seq[1] == 'C' || seq[1] == 'D') { /* left/right — ignore */ }
                }
            } else {
                // Plain Esc
                tcsetattr(STDIN_FILENO, TCSANOW, &orig);
                return SIZE_MAX;
            }
            render();
        } else if (c == 'q' || c == 'Q') {
            tcsetattr(STDIN_FILENO, TCSANOW, &orig);
            return SIZE_MAX;
        }
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &orig);
    return SIZE_MAX;
#endif
}

int guided_flow() {
    auto tier = detect_ui_tier();
    auto pick = [&](const std::string& title, const std::vector<PickerItem>& items) -> size_t {
        if (tier >= UiTier::ARROW_PICKER) return arrow_picker(title, items);
        return numbered_menu(title, items);
    };

    // Step 1: Choose action
    std::vector<PickerItem> actions = {
        {"Edit global settings", ""},
        {"Edit a model's settings", ""},
        {"Show effective configuration", ""},
        {"Exit", ""},
    };
    size_t action = pick("What do you want to do?", actions);
    if (action == SIZE_MAX || action == 3) return 0;

    // Step 2: Choose model (or global)
    EffectiveConfig cfg = load_effective_config();
    std::vector<PickerItem> models;
    models.push_back({"[Global settings]", ""});
    // Add all models from merged registry
    for (auto it_type = cfg.merged["models"].begin(); it_type != cfg.merged["models"].end(); ++it_type) {
        const std::string& type = it_type.key();
        for (auto it_size = it_type->begin(); it_size != it_type->end(); ++it_size) {
            const std::string& size = it_size.key();
            std::string tag = type + ":" + size;
            bool is_user = cfg.user_settings.contains("models") &&
                           cfg.user_settings["models"].contains(type) &&
                           cfg.user_settings["models"][type].contains(size);
            models.push_back({tag, is_user ? "(user-added)" : ""});
        }
    }

    size_t model_idx = pick("Select model:", models);
    if (model_idx == SIZE_MAX) return 0;
    std::string model_tag = (model_idx == 0) ? "" : models[model_idx].label;

    if (action == 2) {
        // Show effective configuration
        program_args_t show_args;
        show_args.config_model = model_tag;
        return config_show(show_args);
    }

    // Step 3: Choose setting
    std::vector<PickerItem> settings;
    for (const auto& def : settings_registry()) {
        if (def.scope == SettingScope::PER_MODEL && model_tag.empty()) continue;
        settings.push_back({def.key, def.description});
    }
    size_t setting_idx = pick("Select setting:", settings);
    if (setting_idx == SIZE_MAX) return 0;
    std::string key = settings[setting_idx].label;

    // Step 4: Choose value (or unset)
    auto def = find_setting(key);
    if (def && def->type == SettingType::ENUM) {
        std::vector<PickerItem> options;
        for (const auto& opt : def->options) {
            options.push_back(PickerItem{opt.value, opt.description});
        }
        size_t opt_idx = pick("Select value for " + key + ":", options);
        if (opt_idx == SIZE_MAX) return 0;
        
        // Set the value
        SettingsLock lock(cfg.user_settings_path);
        if (!lock.locked()) {
            std::cerr << "Error: Could not acquire lock on settings file" << std::endl;
            return 1;
        }
        
        nlohmann::json settings_json = cfg.user_settings;
        if (settings_json.is_null()) {
            settings_json = nlohmann::json::object();
        }
        
        if (model_tag.empty()) {
            // Global setting
            if (!settings_json.contains("defaults")) {
                settings_json["defaults"] = nlohmann::json::object();
            }
            settings_json["defaults"][key] = options[opt_idx].label;
        } else {
            // Per-model setting
            auto model_parts = split_tag(model_tag);
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
            settings_json["settings"][type][size][key] = options[opt_idx].label;
        }
        
        // Update defaults_version
        if (cfg.shipped_registry.contains("defaults") &&
            cfg.shipped_registry["defaults"].contains("defaults_version")) {
            settings_json["defaults_version"] = cfg.shipped_registry["defaults"]["defaults_version"];
        }
        
        save_user_settings(settings_json, cfg.user_settings_path);
        std::cout << "Set '" << key << "' = '" << options[opt_idx].label << "'" << std::endl;
    } else {
        std::cerr << "Error: Setting '" << key << "' is not an ENUM type or not found" << std::endl;
        return 1;
    }

    return 0;
}

int browse_tui() {
    if (detect_ui_tier() < UiTier::FULL_TUI) {
        std::cerr << "[OFLM] Full-screen mode requires a modern terminal." << std::endl;
        std::cerr << "[OFLM] Use 'oflm config' for the interactive guided flow, or" << std::endl;
        std::cerr << "[OFLM] 'oflm config set --config-key <key> --config-value <value>' for direct setting." << std::endl;
        return 1;
    }

    // Basic TUI: show all settings, allow quit
    EffectiveConfig cfg = load_effective_config();
    std::cout << "\033[2J\033[H"; // Clear screen
    std::cout << "┌─ oflm config ─────────────────────────────────────────────┐\n";
    std::cout << "│  Press 'q' to quit                                        │\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n";
    
    // Render global settings
    std::cout << "\nGlobal settings:\n";
    std::cout << "key                      value                      source\n";
    std::cout << "------------------------ ------------------------ ------------------------\n";
    
    bool found_any = false;
    for (const auto& def : settings_registry()) {
        if (def.scope != SettingScope::GLOBAL && def.scope != SettingScope::BOTH) {
            continue;
        }
        
        auto value_opt = get_setting_value(cfg, "", def.key, program_args_t());
        if (value_opt) {
            std::cout << std::left << std::setw(24) << def.key << " "
                      << std::setw(26) << value_opt->value << " "
                      << source_tag(value_opt->source) << "\n";
            found_any = true;
        }
    }
    
    if (!found_any) {
        std::cout << "No global settings found\n";
    }
    
    // Simple input loop
    char c;
    while (true) {
        if (read(STDIN_FILENO, &c, 1) != 1) break;
        if (c == 'q' || c == 'Q') break;
    }
    std::cout << "\033[2J\033[H"; // Clear on exit
    return 0;
}

} // namespace oflm::config
