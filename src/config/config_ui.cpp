#include "config/config_ui.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdint>

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
int guided_flow() { return 1; } // Task 17
int browse_tui() { return 1; }   // Task 17

} // namespace oflm::config
