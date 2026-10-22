# Task 17: Guided Flow + TUI + Config Review Interactive

**Phase:** 5 (Interactive UI)
**Depends on:** Task 16, Task 13
**Files to modify:** `src/config/config_ui.cpp` (replace stubs), `src/config/config_cmd.cpp` (wire interactive features)
**Estimated complexity:** Large (~250 lines)

## Context

Implement the three interactive features:
1. **Guided flow** (`oflm config` with no action) — chains pickers: action → model → setting → value
2. **Full TUI** (`oflm config browse`) — ANSI-based browse/edit interface
3. **Config review interactive** — reconcile overrides with keep/remove/skip

Also wire `config set` without value to use the picker.

## What to do

1. Implement `guided_flow()` in `config_ui.cpp`
2. Implement `browse_tui()` in `config_ui.cpp`
3. Update `config_cmd.cpp` to call picker when `set` has no value
4. Update `config_cmd.cpp` review action to support interactive mode

## Implementation sketch

### guided_flow()

```cpp
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
    auto cfg = load_effective_config();
    std::vector<PickerItem> models;
    models.push_back({"[Global settings]", ""});
    // Add all models from merged registry
    for (const auto& [type, sizes] : cfg.merged["models"].items()) {
        for (const auto& [size, entry] : sizes.items()) {
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
        // (Call the show logic from config_cmd)
        return 0; // Simplified for now
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
            options.push_back({opt.value, opt.description});
        }
        size_t opt_idx = pick("Select value for " + key + ":", options);
        if (opt_idx == SIZE_MAX) return 0;
        // Call config set logic
        // (Set the value in settings.json)
    }

    return 0;
}
```

### browse_tui()

Simplified ANSI TUI. Full implementation is complex — provide a basic version:

```cpp
int browse_tui() {
    if (detect_ui_tier() < UiTier::FULL_TUI) {
        std::cerr << "[OFLM] Full-screen mode requires a modern terminal." << std::endl;
        std::cerr << "[OFLM] Use 'oflm config' for the interactive guided flow, or" << std::endl;
        std::cerr << "[OFLM] 'oflm config set --config-key <key> --config-value <value>' for direct setting." << std::endl;
        return 1;
    }

    // Basic TUI: show all settings, allow quit
    auto cfg = load_effective_config();
    std::cout << "\033[2J\033[H"; // Clear screen
    std::cout << "┌─ oflm config ─────────────────────────────────────────────┐\n";
    std::cout << "│  Press 'q' to quit, 'h' for help                          │\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n";
    // ... render settings ...

    // Simple input loop
    char c;
    while (read(STDIN_FILENO, &c, 1) == 1) {
        if (c == 'q' || c == 'Q') break;
    }
    std::cout << "\033[2J\033[H"; // Clear on exit
    return 0;
}
```

### Wire config set picker

In `config_cmd.cpp`, when `config set` has no `--config-value`:

```cpp
if (args.config_value.empty()) {
    auto def = find_setting(args.config_key);
    if (def && def->type == SettingType::ENUM) {
        std::vector<PickerItem> items;
        for (const auto& opt : def->options) {
            items.push_back({opt.value, opt.description});
        }
        size_t idx = arrow_picker("Select value for " + args.config_key + ":", items);
        if (idx == SIZE_MAX) return 1; // cancelled
        // Use items[idx].label as the value
    }
}
```

### Wire config review interactive

In `config_cmd.cpp`, update the review action:

```cpp
if (mode == "interactive") {
    // Interactive reconciliation using pickers
    auto cfg = load_effective_config();
    if (!cfg.defaults_changed) {
        std::cout << "No shipped-default changes detected." << std::endl;
        return 0;
    }

    std::cout << "Shipped defaults changed. Reconcile your overrides:\n" << std::endl;

    // For each affected override, ask keep/remove/skip
    // ... (simplified: use reconcile_settings with "accept-shipped" or "keep-all")
    // Full interactive implementation uses pickers for each override

    // For now, delegate to non-interactive modes
    std::cout << "Interactive mode: use --accept-shipped or --keep-all for now." << std::endl;
    return 1;
}
```

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -10

# Test guided flow (Tier 1 with piped input)
printf "0\n" | ./build/src/oflm config 2>&1 | head -10

# Test browse (should fall back gracefully on non-TTY)
echo "q" | ./build/src/oflm config browse 2>&1 | head -5

# Test config set with picker (needs TTY, so test with piped input for Tier 1)
export OFLM_SETTINGS_PATH=/tmp/test_settings.json
printf "2\n" | ./build/src/oflm config set --config-key prefill_mode 2>&1 | head -10
```

## Success criteria

- [ ] `guided_flow()` implemented (chains pickers)
- [ ] `browse_tui()` implemented (basic ANSI TUI, falls back gracefully)
- [ ] `config set` without value uses picker
- [ ] `config review` interactive mode wired (may delegate to non-interactive for now)
- [ ] `oflm config` (no action) opens guided flow
- [ ] `oflm config browse` works or falls back gracefully
- [ ] Project compiles
