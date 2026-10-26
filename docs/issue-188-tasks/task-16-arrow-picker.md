# Task 16: Arrow-Key Picker (Tier 2)

**Phase:** 5 (Interactive UI)
**Depends on:** Task 15
**Files to modify:** `src/config/config_ui.cpp` (replace arrow_picker stub)
**Estimated complexity:** Medium (~100 lines)

## Context

Implement the Tier 2 arrow-key picker using termios (POSIX) and Windows Console API. This is the interactive selector used when `oflm config set` has no value.

## What to do

Replace the `arrow_picker()` stub in `src/config/config_ui.cpp` with a real implementation.

## Implementation

```cpp
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
```

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -5

# Test with piped input (should fall back to numbered menu)
printf "1\n" | ./build/src/oflm config 2>&1 | head -10
```

## Success criteria

- [ ] `arrow_picker()` implemented (POSIX termios + Windows Console API)
- [ ] Falls back to numbered menu if tier < ARROW_PICKER
- [ ] Up/down navigation works
- [ ] Enter selects, Esc cancels
- [ ] Project compiles
