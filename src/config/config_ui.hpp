#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include "settings_resolver.hpp"

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

/// Format a source tag for display
const char* source_tag(SettingValue::Source s);

} // namespace oflm::config
