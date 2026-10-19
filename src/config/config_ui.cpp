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
    // The title may already end in a colon; do not double it.
    std::cout << title;
    if (title.empty() || title.back() != ':') std::cout << ':';
    std::cout << "\n";
    for (size_t i = 0; i < items.size(); i++) {
        std::cout << "  " << (i + 1) << ". " << items[i].label;
        if (!items[i].description.empty()) {
            std::cout << "  " << items[i].description;
        }
        std::cout << "\n";
    }
    std::cout << "  0. Back\n";
    std::cout << "\n  [number] choose   [0] back   [q] quit\n";
    std::cout << "> " << std::flush;

    std::string line;
    if (!std::getline(std::cin, line)) return SIZE_MAX;

    // Trim, so " q " works as well as "q".
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) line.erase(line.begin());
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
    if (line == "q" || line == "Q") return SIZE_MAX;   // quit the whole flow

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
        std::cout << "\n  ↑/↓ navigate  Enter select  Esc back  q quit\n" << std::flush;
    };

    render();
    while (true) {
        INPUT_RECORD record;
        DWORD read;
        ReadConsoleInput(hIn, &record, 1, &read);
        if (record.EventType != KEY_EVENT) continue;
        if (!record.Event.KeyEvent.bKeyDown) continue;

        // Up at the top wraps to the bottom, and down at the bottom wraps to
        // the top: a list that stops dead at the ends makes you look for the
        // arrow that does nothing.
        if (record.Event.KeyEvent.wVirtualKeyCode == VK_UP) {
            selected = (selected + items.size() - 1) % items.size();
            render();
        } else if (record.Event.KeyEvent.wVirtualKeyCode == VK_DOWN) {
            selected = (selected + 1) % items.size();
            render();
        } else if (record.Event.KeyEvent.wVirtualKeyCode == VK_RETURN) {
            SetConsoleMode(hIn, orig_mode);
            return selected;
        } else if (record.Event.KeyEvent.wVirtualKeyCode == VK_ESCAPE) {
            SetConsoleMode(hIn, orig_mode);
            return SIZE_MAX;
        } else if (record.Event.KeyEvent.wVirtualKeyCode == 'Q') {
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
        std::cout << "\n  ↑/↓ navigate  Enter select  Esc back  q quit\n" << std::flush;
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
                    // Up at the top wraps to the bottom, and down at the
                    // bottom wraps to the top.
                    if (seq[1] == 'A') selected = (selected + items.size() - 1) % items.size();      // Up
                    else if (seq[1] == 'B') selected = (selected + 1) % items.size();                  // Down
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

// What a setting currently holds, and where that value came from, for the
// menu to show: "8192 (global)" or "balanced (custom)".
static std::string current_value_note(const EffectiveConfig& cfg,
                                       const std::string& model_tag,
                                       const std::string& key) {
    program_args_t no_args;
    auto v = get_setting_value(cfg, model_tag, key, no_args);
    if (!v) return "";
    return v->value + " " + source_tag(v->source);
}

// The bare value a setting holds, without where it came from.
static std::string current_value_only(const EffectiveConfig& cfg,
                                      const std::string& model_tag,
                                      const std::string& key) {
    program_args_t no_args;
    auto v = get_setting_value(cfg, model_tag, key, no_args);
    if (!v) return "";
    return v->value;
}

// After putting something on the screen, wait for the user to acknowledge it.
// The next menu redraws the whole screen, so without this the output is wiped
// before it can be read.
static void wait_for_ack() {
    std::cout << "\n  [press Enter to continue]" << std::flush;
    std::string line;
    std::getline(std::cin, line);
    std::cout << std::endl;
}

int guided_flow() {
    auto tier = detect_ui_tier();
    auto pick = [&](const std::string& title, const std::vector<PickerItem>& items) -> size_t {
        if (tier >= UiTier::ARROW_PICKER) return arrow_picker(title, items);
        return numbered_menu(title, items);
    };
    // The same, but starting the cursor on the entry that is in effect.
    auto pick_from = [&](const std::string& title, const std::vector<PickerItem>& items,
                         size_t current) -> size_t {
        if (tier >= UiTier::ARROW_PICKER) return arrow_picker(title, items, current);
        return numbered_menu(title, items);
    };

    // The flow is a small state machine. Cancelling a step (Esc, or 0 in the
    // numbered menu) goes back to the step before it; cancelling the first one
    // leaves the flow. A linear chain of prompts would have quit the program
    // on the first Esc, which is not what a menu is expected to do.
    enum class Step { ACTION, MODEL, SETTING, VALUE };
    Step step = Step::ACTION;

    EffectiveConfig cfg = load_effective_config();

    // The model list is rebuilt whenever the config changes, so the markers
    // stay true: "(user-added)" for a model the user installed, and the
    // number of settings they have customised.
    std::vector<PickerItem> models;
    auto rebuild_models = [&]() {
        models.clear();
        models.push_back({"[Global settings]", ""});
        for (auto it_type = cfg.merged["models"].begin(); it_type != cfg.merged["models"].end(); ++it_type) {
            const std::string& type = it_type.key();
            for (auto it_size = it_type->begin(); it_size != it_type->end(); ++it_size) {
                const std::string& size = it_size.key();
                std::string tag = type + ":" + size;
                std::string note;
                bool is_user = cfg.user_settings.contains("models") &&
                               cfg.user_settings["models"].contains(type) &&
                               cfg.user_settings["models"][type].contains(size);
                if (is_user) note = "(user-added)";
                if (cfg.user_settings.contains("settings") &&
                    cfg.user_settings["settings"].contains(type) &&
                    cfg.user_settings["settings"][type].contains(size)) {
                    size_t n = cfg.user_settings["settings"][type][size].size();
                    if (!note.empty()) note += " ";
                    note += "(" + std::to_string(n) + " custom)";
                }
                models.push_back({tag, note});
            }
        }
    };
    rebuild_models();

    std::vector<PickerItem> settings;
    std::vector<PickerItem> options;
    size_t action = 0;
    std::string model_tag;
    std::string key;
    // Where the setting step goes back to: the model list when a model was
    // chosen, the action list when the global settings were.
    Step settings_back_to = Step::ACTION;

    for (;;) {
        switch (step) {
        case Step::ACTION: {
            std::vector<PickerItem> actions = {
                {"Edit global settings", ""},
                {"Edit a model's settings", ""},
                {"Show effective configuration", ""},
                {"Exit", ""},
            };
            size_t a = pick("What do you want to do?", actions);
            if (a == SIZE_MAX) return 0;   // nothing above this step
            if (a == 3) return 0;          // Exit
            action = a;
            if (action == 0) {
                // The global settings: no model to choose.
                model_tag.clear();
                settings.clear();
                for (const auto& def : settings_registry()) {
                    if (def.scope == SettingScope::PER_MODEL) continue;
                    // Show what it holds now, and whether that is the user's
                    // own value or the shipped one, so returning to this list
                    // shows the state rather than a bare list of names.
                    settings.push_back({def.key, current_value_note(cfg, model_tag, def.key)});
                }
                settings_back_to = Step::ACTION;
                step = Step::SETTING;
            } else {
                step = Step::MODEL;
            }
            break;
        }

        case Step::MODEL: {
            size_t m = pick("Select model:", models);
            if (m == SIZE_MAX) { step = Step::ACTION; break; }   // back
            model_tag = (m == 0) ? "" : models[m].label;

            if (action == 2) {
                // Show effective configuration, then back to the top.
                program_args_t show_args;
                show_args.config_model = model_tag;
                config_show(show_args);
                // The next menu redraws the screen; let the user read this
                // first.
                wait_for_ack();
                step = Step::ACTION;
                break;
            }

            settings.clear();
            for (const auto& def : settings_registry()) {
                if (def.scope == SettingScope::PER_MODEL && model_tag.empty()) continue;
                // The value it holds now, and whether that is the user's own
                // value or the shipped one.
                settings.push_back({def.key, current_value_note(cfg, model_tag, def.key)});
            }
            settings_back_to = Step::MODEL;
            step = Step::SETTING;
            break;
        }

        case Step::SETTING: {
            size_t s = pick("Select setting:", settings);
            if (s == SIZE_MAX) { step = settings_back_to; break; }   // back
            key = settings[s].label;

            options.clear();
            auto def = find_setting(key);
            if (def && def->type == SettingType::ENUM) {
                for (const auto& opt : def->options) {
                    options.push_back(PickerItem{opt.value, opt.description});
                }
            }
            step = Step::VALUE;
            break;
        }

        case Step::VALUE: {
            if (options.empty()) {
                // Not an enum: there is nothing to pick here.
                std::cerr << "Setting '" << key << "' is not an ENUM type; set it with"
                          << " 'oflm config set --config-key " << key << " --config-value <value>'"
                          << std::endl;
                step = Step::SETTING;
                break;
            }

            // Show which option is in effect, and whether that is the user's
            // own value or the shipped one, and start the cursor on it.
            std::string current = current_value_note(cfg, model_tag, key);
            size_t current_idx = SIZE_MAX;
            for (size_t i = 0; i < options.size(); i++) {
                if (options[i].label == current_value_only(cfg, model_tag, key)) {
                    current_idx = i;
                    break;
                }
            }
            std::string title = "Select value for " + key;
            if (!current.empty()) title += " (current: " + current + ")";
            size_t o = pick_from(title, options, current_idx);
            if (o == SIZE_MAX) { step = Step::SETTING; break; }  // back

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
                settings_json["defaults"][key] = options[o].label;
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
                settings_json["settings"][type][size][key] = options[o].label;
            }

            // Remember the shipped defaults version this file was written against.
            if (cfg.shipped_registry.contains("defaults_version")) {
                settings_json["defaults_version"] = cfg.shipped_registry["defaults_version"];
            }

            save_user_settings(settings_json, cfg.user_settings_path);
            std::cout << "Set '" << key << "' = '" << options[o].label << "'" << std::endl;

            // The next menu redraws the screen; let the user read this first.
            wait_for_ack();

            // Back to the menu we came from -- the model list for a model's
            // setting, the action list for a global one -- so the user lands
            // where they were, with the value now showing as their own.
            // The config in memory is refreshed so the list shows the new value.
            cfg = load_effective_config();
            rebuild_models();
            step = settings_back_to;
            break;
        }
        }
    }
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
