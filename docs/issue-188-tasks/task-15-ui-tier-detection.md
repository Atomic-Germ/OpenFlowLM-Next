# Task 15: UI Tier Detection + Numbered Menu

**Phase:** 5 (Interactive UI)
**Depends on:** Task 11
**Files to create:** `src/config/config_ui.hpp`, `src/config/config_ui.cpp` (partial)
**Estimated complexity:** Medium (~120 lines)

## Context

Three-tier UI system: Tier 1 (numbered menu, always works), Tier 2 (arrow-key picker), Tier 3 (full-screen TUI). This task implements tier detection and Tier 1.

## What to do

Create `src/config/config_ui.hpp` and `src/config/config_ui.cpp` with:
1. `UiTier` enum and `detect_ui_tier()`
2. `Picker` class (Tier 2 — stub for now, implemented in Task 16)
3. `numbered_menu()` function (Tier 1)

## Code

### `config_ui.hpp`

```cpp
#pragma once
#include <string>
#include <vector>
#include <cstddef>

namespace oflm::config {

enum class UiTier { NUMBERED_MENU, ARROW_PICKER, FULL_TUI };

/// Detect the best available UI tier for the current terminal.
UiTier detect_ui_tier();

/// One item in a picker/menu.
struct PickerItem {
    std::string label;
    std::string description;
};

/// Arrow-key picker (Tier 2). Returns selected index, or SIZE_MAX if cancelled.
/// Implemented in Task 16.
size_t arrow_picker(const std::string& title, const std::vector<PickerItem>& items, size_t current = SIZE_MAX);

/// Numbered menu (Tier 1). Always works. Returns selected index, or SIZE_MAX if cancelled.
size_t numbered_menu(const std::string& title, const std::vector<PickerItem>& items);

/// Run the guided flow (oflm config with no action).
/// Implemented in Task 17.
int guided_flow();

/// Run the full-screen TUI (oflm config browse).
/// Implemented in Task 17.
int browse_tui();

} // namespace oflm::config
```

### `config_ui.cpp` (partial — tier detection + numbered menu)

```cpp
#include "config/config_ui.hpp"
#include <iostream>
#include <cstdlib>
#include <string>

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
size_t arrow_picker(const std::string&, const std::vector<PickerItem>&, size_t) {
    return SIZE_MAX; // Task 16
}
int guided_flow() { return 1; } // Task 17
int browse_tui() { return 1; }   // Task 17

} // namespace oflm::config
```

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -10

# Test tier detection (should print tier info if we add a debug command, or just verify compile)
# Test numbered menu with piped input
echo "0" | ./build/src/oflm config 2>&1 | head -10
```

## Success criteria

- [ ] `config_ui.hpp` and `config_ui.cpp` created
- [ ] `detect_ui_tier()` implemented (POSIX + Windows)
- [ ] `numbered_menu()` implemented
- [ ] `arrow_picker()`, `guided_flow()`, `browse_tui()` are stubs
- [ ] Project compiles
