#pragma once
/// \file config_keys.hpp
/// \brief Key parsing for the interactive config pickers.
/// \note The parser is a pure function over a byte buffer, with no terminal
///       and no platform code in it, so it can be tested without a tty. The
///       terminal side (raw mode, reading, the ESC timeout) lives in
///       config_ui.cpp.
#include <cstddef>

namespace oflm::config {

/// A key the pickers understand.
enum class UiKey {
    None,        ///< nothing was read
    Up, Down, Left, Right,
    Home, End, PageUp, PageDown,
    Enter,
    Escape,
    Char,        ///< a printable character: see UiKeyEvent::ch
    Other        ///< a key the pickers do not use (a function key, a mouse report, ...)
};

/// One key event.
struct UiKeyEvent {
    UiKey key = UiKey::None;
    char ch = 0;   ///< only meaningful when key == UiKey::Char
};

/// Parse one key event out of a byte buffer.
///
/// The sequences understood are the ones a terminal actually sends:
///   CSI   ESC [ <params> <final>   -- the cursor and function keys
///   SS3   ESC O <final>            -- the same keys in application mode
/// A single ESC is an Escape, and any other escape sequence is consumed
/// whole and reported as Other, so its bytes never leak into the next key.
///
/// @param buf  the bytes to parse
/// @param len  how many bytes are there
/// @param out  the event, filled in when the return value is not 0
/// @return the number of bytes consumed, or 0 when the buffer does not hold
///         a complete event yet (an escape sequence still arriving).
size_t parse_ui_key(const char* buf, size_t len, UiKeyEvent* out);

/// The movement a character stands for in the vim style: j is down, k is up,
/// h is left (back) and l is right (into).
/// @return UiKey::None when the character is not a movement key.
UiKey vim_key_for_char(char c);

} // namespace oflm::config
