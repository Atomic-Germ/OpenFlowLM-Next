#include "config/config_ui.hpp"
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include "config/settings_resolver.hpp"
#include "config/config_cmd.hpp"
#include "config/config_keys.hpp"
#include "config/config_paths.hpp"
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
#include <poll.h>
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

namespace {

// The platform side of the key reader: raw mode in, one key event out. The
// two platforms differ in how the terminal is put in raw mode and how a key
// is taken from it; what a key MEANS is the same on both, and is handled
// once, in the picker below.
#ifdef _WIN32
class ConsoleKeys {
public:
    ConsoleKeys() {
        hIn_ = GetStdHandle(STD_INPUT_HANDLE);
        if (GetConsoleMode(hIn_, &orig_)) {
            SetConsoleMode(hIn_, orig_ & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT));
            raw_ = true;
        }
    }
    ~ConsoleKeys() { if (raw_) SetConsoleMode(hIn_, orig_); }

    UiKeyEvent read() {
        UiKeyEvent ev;
        for (;;) {
            INPUT_RECORD rec;
            DWORD got = 0;
            if (!ReadConsoleInput(hIn_, &rec, 1, &got) || got == 0) { ev.key = UiKey::None; return ev; }
            if (rec.EventType != KEY_EVENT) continue;
            const KEY_EVENT_RECORD& ke = rec.Event.KeyEvent;
            if (!ke.bKeyDown) continue;

            switch (ke.wVirtualKeyCode) {
                case VK_UP:     ev.key = UiKey::Up; return ev;
                case VK_DOWN:   ev.key = UiKey::Down; return ev;
                case VK_LEFT:   ev.key = UiKey::Left; return ev;
                case VK_RIGHT:  ev.key = UiKey::Right; return ev;
                case VK_HOME:   ev.key = UiKey::Home; return ev;
                case VK_END:    ev.key = UiKey::End; return ev;
                case VK_PRIOR:  ev.key = UiKey::PageUp; return ev;
                case VK_NEXT:   ev.key = UiKey::PageDown; return ev;
                case VK_RETURN: ev.key = UiKey::Enter; return ev;
                case VK_ESCAPE: ev.key = UiKey::Escape; return ev;
                default: break;
            }
            if (ke.uChar.AsciiChar) {
                ev.key = UiKey::Char;
                ev.ch = ke.uChar.AsciiChar;
                return ev;
            }
        }
    }

private:
    HANDLE hIn_ = nullptr;
    DWORD orig_ = 0;
    bool raw_ = false;
};
#else
class TerminalKeys {
public:
    TerminalKeys() {
        if (tcgetattr(STDIN_FILENO, &orig_) == 0) {
            struct termios raw = orig_;
            raw.c_lflag &= ~(ICANON | ECHO);
            raw.c_cc[VMIN] = 1;
            raw.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &raw);
            raw_ = true;
        }
    }
    ~TerminalKeys() { if (raw_) tcsetattr(STDIN_FILENO, TCSANOW, &orig_); }

    UiKeyEvent read() {
        UiKeyEvent ev;
        for (;;) {
            // An escape on its own is not a decision yet: the rest of the
            // sequence may still be on its way, and a terminal is free to
            // deliver it in two pieces. So the buffer is only parsed when it
            // holds more than a lone ESC.
            if (!pending_.empty() && !(pending_.size() == 1 && pending_[0] == 0x1b)) {
                size_t n = parse_ui_key(pending_.data(), pending_.size(), &ev);
                if (n > 0) {
                    pending_.erase(0, n);
                    return ev;
                }
            }
            // Nothing complete yet. Wait for the next byte: for the first one
            // as long as it takes, and for a short while after an escape --
            // a lone ESC has to be told apart from the start of a sequence,
            // and the terminal sends the rest of a sequence right away.
            struct pollfd pfd {};
            pfd.fd = STDIN_FILENO;
            pfd.events = POLLIN;
            int pr = poll(&pfd, 1, pending_.empty() ? -1 : 50);
            if (pr == 0) {
                // The wait expired and nothing more came: the escape stands
                // alone. Whatever followed it stays pending.
                if (!pending_.empty() && pending_[0] == 0x1b) {
                    pending_.erase(0, 1);
                    ev.key = UiKey::Escape;
                    ev.ch = 0;
                    return ev;
                }
                ev.key = UiKey::None;
                return ev;
            }
            if (pr < 0) { ev.key = UiKey::None; return ev; }

            char buf[64];
            ssize_t got = ::read(STDIN_FILENO, buf, sizeof(buf));
            if (got <= 0) { ev.key = UiKey::None; return ev; }
            pending_.append(buf, static_cast<size_t>(got));
        }
    }

private:
    // The bytes that arrived but do not make a whole event yet: an escape
    // sequence can be split across two reads, and the tail of what the user
    // typed while the program was busy must not be lost.
    std::string pending_;
    struct termios orig_ {};
    bool raw_ = false;
};
#endif

} // namespace

// Stubs — implemented in Tasks 16-17
size_t arrow_picker(const std::string& title, const std::vector<PickerItem>& items, size_t current) {
    if (items.empty()) return SIZE_MAX;
    if (detect_ui_tier() < UiTier::ARROW_PICKER) {
        // Fall back to numbered menu
        return numbered_menu(title, items);
    }

    size_t selected = (current != SIZE_MAX && current < items.size()) ? current : 0;

    auto render = [&]() {
        std::cout << "\033[H\033[2J" << title << "\n";
        for (size_t i = 0; i < items.size(); i++) {
            if (i == selected) std::cout << "  > " << items[i].label;
            else std::cout << "    " << items[i].label;
            if (!items[i].description.empty()) std::cout << "  " << items[i].description;
            std::cout << "\n";
        }
        std::cout << "\n  ↑/↓ or j/k move   Enter or l select   Esc or h back   q quit\n" << std::flush;
    };

    // What PageUp and PageDown move by.
    const size_t page = 10;

    // The terminal is put back the way it was when this scope ends, on every
    // way out of it -- including an exception.
#ifdef _WIN32
    ConsoleKeys keys;
#else
    TerminalKeys keys;
#endif

    render();
    for (;;) {
        UiKeyEvent ev = keys.read();
        UiKey k = ev.key;

        if (k == UiKey::Char) {
            if (ev.ch == 'q' || ev.ch == 'Q') return SIZE_MAX;
            // The vim style: j is down, k is up, h is back, l is into.
            UiKey vim = vim_key_for_char(ev.ch);
            if (vim != UiKey::None) k = vim;
        }

        switch (k) {
            case UiKey::Enter:  return selected;
            case UiKey::Escape: return SIZE_MAX;
            case UiKey::Left:   return SIZE_MAX;    // h: back
            case UiKey::Right:  return selected;    // l: into
            case UiKey::Up:
                // Up at the top wraps to the bottom, and down at the bottom
                // wraps to the top: a list that stops dead at the ends makes
                // you look for the arrow that does nothing.
                selected = (selected + items.size() - 1) % items.size();
                render();
                break;
            case UiKey::Down:
                selected = (selected + 1) % items.size();
                render();
                break;
            case UiKey::Home:
                selected = 0;
                render();
                break;
            case UiKey::End:
                selected = items.size() - 1;
                render();
                break;
            case UiKey::PageUp:
                selected = (selected + items.size() - page % items.size()) % items.size();
                render();
                break;
            case UiKey::PageDown:
                selected = (selected + page) % items.size();
                render();
                break;
            default:
                break;   // a key the picker does not use
        }
    }
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
    // number of settings whose value the user chose for this model — the
    // same ones the view shows as (custom), whether they were set for the
    // model alone or app-wide.
    std::vector<PickerItem> models;
    auto rebuild_models = [&]() {
        models.clear();
        models.push_back({"[Global settings]", ""});
        program_args_t no_args;
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
                size_t n = 0;
                for (const auto& def : settings_registry()) {
                    auto v = get_setting_value(cfg, tag, def.key, no_args);
                    if (v && v->source == SettingValue::USER) n++;
                }
                if (n) {
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
                {"Show the paths in use", ""},
                {"Exit", ""},
            };
            size_t a = pick("What do you want to do?", actions);
            if (a == SIZE_MAX) return 0;   // nothing above this step
            if (a == 4) return 0;          // Exit
            action = a;
            if (action == 3) {
                // The paths in use: where the registry, the kernels, the
                // models and the settings are read from and written to.
                print_paths(std::cout);
                std::cout << std::endl;
                // The next menu redraws the screen; let the user read this.
                wait_for_ack();
                break;                      // and stay in this menu
            }
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

            // Where the value goes: a global setting that this model does not
            // override is an app-wide setting, so it is written to the
            // defaults block and every model follows. A setting this model
            // overrides for itself stays with the model.
            bool model_overrides_it = false;
            auto def = find_setting(key);
            bool is_global = def && def->scope == SettingScope::GLOBAL;
            if (!model_tag.empty() && is_global) {
                auto mp = split_tag(model_tag);
                model_overrides_it = cfg.user_settings.contains("settings") &&
                                    cfg.user_settings["settings"].contains(mp.first) &&
                                    cfg.user_settings["settings"][mp.first].contains(mp.second) &&
                                    cfg.user_settings["settings"][mp.first][mp.second].contains(key);
            }

            if (model_tag.empty() || (is_global && !model_overrides_it)) {
                // Global setting: it applies to every model
                if (!settings_json.contains("defaults")) {
                    settings_json["defaults"] = nlohmann::json::object();
                }
                settings_json["defaults"][key] = options[o].label;
                if (!model_tag.empty()) {
                    std::cout << "  (app-wide: every model now uses this)" << std::endl;
                }
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
    
    // The screen says "press q to quit", so the key has to arrive on its own:
    // in canonical mode the read waits for a whole line, and the line editor
    // eats the arrow keys on the way.
#ifdef _WIN32
    ConsoleKeys keys;
#else
    TerminalKeys keys;
#endif
    for (;;) {
        UiKeyEvent ev = keys.read();
        if (ev.key == UiKey::None) break;                       // end of input
        if (ev.key == UiKey::Escape) break;
        if (ev.key == UiKey::Char && (ev.ch == 'q' || ev.ch == 'Q')) break;
    }
    std::cout << "\033[2J\033[H"; // Clear on exit
    return 0;
}

} // namespace oflm::config
