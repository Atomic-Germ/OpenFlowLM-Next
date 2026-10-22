/// \file config_keys.cpp
/// \brief Key parsing for the interactive config pickers.
#include "config/config_keys.hpp"

#include <cstdlib>
#include <cstring>

namespace oflm::config {

namespace {

// The CSI final byte that stands for a key. The two families a terminal uses
// for the same key are both here: the "normal" one (ESC [ A) and the
// "application" one (ESC O A), which is what a terminal sends when it is in
// application keypad mode. The old parser only knew the first family, so an
// arrow key in that mode was taken for a plain Escape -- and took the whole
// config flow down with it.
UiKey csi_key(char final) {
    switch (final) {
        case 'A': return UiKey::Up;
        case 'B': return UiKey::Down;
        case 'C': return UiKey::Right;
        case 'D': return UiKey::Left;
        case 'H': return UiKey::Home;
        case 'F': return UiKey::End;
        default:  return UiKey::Other;
    }
}

// ESC [ <n> ~ : the numeric function keys. The number is what tells them
// apart, the final byte is always '~'.
UiKey csi_tilde_key(const char* params, size_t len) {
    char buf[32];
    size_t n = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
    memcpy(buf, params, n);
    buf[n] = '\0';
    switch (std::atoi(buf)) {
        case 1: case 7: return UiKey::Home;
        case 4: case 8: return UiKey::End;
        case 5: return UiKey::PageUp;
        case 6: return UiKey::PageDown;
        default: return UiKey::Other;
    }
}

} // namespace

size_t parse_ui_key(const char* buf, size_t len, UiKeyEvent* out) {
    if (!buf || len == 0) return 0;

    auto emit = [&](UiKey k, char c = 0) {
        if (out) { out->key = k; out->ch = c; }
    };

    unsigned char c0 = static_cast<unsigned char>(buf[0]);

    if (c0 == 0x1b) {                                  // ESC
        if (len == 1) { emit(UiKey::Escape); return 1; }

        const char intro = buf[1];
        if (intro == '[') {                             // CSI
            size_t i = 2;
            // The parameter bytes. A modifier form (ESC [ 1 ; 5 A) has more
            // of them, and the key is the same one.
            while (i < len && ((buf[i] >= '0' && buf[i] <= '9') || buf[i] == ';' ||
                               buf[i] == '?' || buf[i] == '<' || buf[i] == ':')) {
                i++;
            }
            if (i >= len) return 0;                    // still arriving
            const char final = buf[i];
            if (final == '~') {
                emit(csi_tilde_key(buf + 2, i - 2));
            } else {
                emit(csi_key(final));
            }
            return i + 1;
        }
        if (intro == 'O') {                             // SS3, application mode
            if (len < 3) return 0;                      // still arriving
            emit(csi_key(buf[2]));
            return 3;
        }
        // ESC followed by anything else: Alt+key, or a stray byte. Both are
        // of no use here; the two bytes are consumed together so the second
        // one is not read again as a key of its own.
        emit(UiKey::Other);
        return 2;
    }

    if (c0 == '\r' || c0 == '\n') { emit(UiKey::Enter); return 1; }
    if (c0 == 0x7f || c0 == 0x08) { emit(UiKey::Other); return 1; }   // backspace
    if (c0 < 0x20) { emit(UiKey::Other); return 1; }                // control

    emit(UiKey::Char, static_cast<char>(c0));
    return 1;
}

UiKey vim_key_for_char(char c) {
    switch (c) {
        case 'j': return UiKey::Down;
        case 'k': return UiKey::Up;
        case 'h': return UiKey::Left;
        case 'l': return UiKey::Right;
        default:  return UiKey::None;
    }
}

} // namespace oflm::config
