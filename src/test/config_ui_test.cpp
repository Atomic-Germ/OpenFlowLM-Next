/// \file config_ui_test.cpp
/// \brief Unit tests for the key parsing of the interactive config pickers.
#include "config/config_keys.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

#define TEST(name) static void name(); \
    static struct name##_registrar { name##_registrar() { tests.push_back({#name, name}); } } name##_instance; \
    static void name()

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "FAIL: " << #expr << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "FAIL: " << #a << " != " << #b << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)

static int failures = 0;
static std::vector<std::pair<std::string, void(*)()>> tests;

using oflm::config::UiKey;
using oflm::config::UiKeyEvent;

/// Parse one event out of a byte string. Fails the test when the bytes do
/// not make a whole event.
static UiKeyEvent one(const std::string& bytes) {
    UiKeyEvent ev;
    size_t n = oflm::config::parse_ui_key(bytes.data(), bytes.size(), &ev);
    if (n != bytes.size()) {
        std::cerr << "FAIL: " << bytes.size() << " bytes consumed, expected "
                  << bytes.size() << std::endl;
        failures++;
    }
    return ev;
}

// --- the cursor keys, in the two forms a terminal sends them -----------------

TEST(csi_up_down_left_right) {
    ASSERT_EQ(one("\x1b[A").key, UiKey::Up);
    ASSERT_EQ(one("\x1b[B").key, UiKey::Down);
    ASSERT_EQ(one("\x1b[C").key, UiKey::Right);
    ASSERT_EQ(one("\x1b[D").key, UiKey::Left);
}

TEST(ss3_up_down_left_right) {
    // The application cursor keys. A terminal in application keypad mode
    // sends these, and the old parser took them for a plain Escape -- which
    // closed the whole config flow.
    ASSERT_EQ(one("\x1bOA").key, UiKey::Up);
    ASSERT_EQ(one("\x1bOB").key, UiKey::Down);
    ASSERT_EQ(one("\x1bOC").key, UiKey::Right);
    ASSERT_EQ(one("\x1bOD").key, UiKey::Left);
}

TEST(home_and_end) {
    ASSERT_EQ(one("\x1b[H").key, UiKey::Home);
    ASSERT_EQ(one("\x1b[F").key, UiKey::End);
    ASSERT_EQ(one("\x1bOH").key, UiKey::Home);
    ASSERT_EQ(one("\x1bOF").key, UiKey::End);
    ASSERT_EQ(one("\x1b[1~").key, UiKey::Home);
    ASSERT_EQ(one("\x1b[4~").key, UiKey::End);
    ASSERT_EQ(one("\x1b[7~").key, UiKey::Home);
    ASSERT_EQ(one("\x1b[8~").key, UiKey::End);
}

TEST(page_up_and_down) {
    ASSERT_EQ(one("\x1b[5~").key, UiKey::PageUp);
    ASSERT_EQ(one("\x1b[6~").key, UiKey::PageDown);
}

TEST(modified_cursor_keys_are_the_same_key) {
    // Ctrl+Up, Shift+Down and the other modifier forms end in the same final
    // byte: the picker wants the key, not the modifier.
    ASSERT_EQ(one("\x1b[1;5A").key, UiKey::Up);
    ASSERT_EQ(one("\x1b[1;2B").key, UiKey::Down);
    ASSERT_EQ(one("\x1b[1;5C").key, UiKey::Right);
    ASSERT_EQ(one("\x1b[1;5D").key, UiKey::Left);
}

// --- what is not a key ------------------------------------------------------

TEST(enter_and_escape) {
    ASSERT_EQ(one("\r").key, UiKey::Enter);
    ASSERT_EQ(one("\n").key, UiKey::Enter);
    ASSERT_EQ(one("\x1b").key, UiKey::Escape);
}

TEST(plain_characters) {
    UiKeyEvent ev = one("j");
    ASSERT_EQ(ev.key, UiKey::Char);
    ASSERT_EQ(ev.ch, 'j');
}

TEST(unknown_sequences_are_swallowed_whole) {
    // The bytes of a sequence the picker does not use must not be read again
    // as keys of their own: shift-tab, a mouse report, a bracketed paste.
    ASSERT_EQ(one("\x1b[Z").key, UiKey::Other);
    ASSERT_EQ(one("\x1b[200~").key, UiKey::Other);
    ASSERT_EQ(one("\x1b[<0;10;20M").key, UiKey::Other);
    ASSERT_EQ(one("\x1b[99~").key, UiKey::Other);
}

TEST(modified_function_keys_are_the_same_key) {
    // Ctrl+Home and Shift+PageUp: the number still says which key it is, the
    // modifier is only what follows it.
    ASSERT_EQ(one("\x1b[1;5~").key, UiKey::Home);
    ASSERT_EQ(one("\x1b[5;2~").key, UiKey::PageUp);
    ASSERT_EQ(one("\x1b[6;2~").key, UiKey::PageDown);
}

TEST(incomplete_sequences_consume_nothing) {
    // The rest of an escape sequence may not have arrived yet.
    UiKeyEvent ev;
    ASSERT_EQ(oflm::config::parse_ui_key("\x1b", 1, &ev), 1u);          // a lone escape
    ASSERT_EQ(oflm::config::parse_ui_key("\x1b[", 2, &ev), 0u);          // CSI, not finished
    ASSERT_EQ(oflm::config::parse_ui_key("\x1bO", 2, &ev), 0u);          // SS3, not finished
    ASSERT_EQ(oflm::config::parse_ui_key("\x1b[1;", 4, &ev), 0u);        // parameters, not finished
    ASSERT_EQ(oflm::config::parse_ui_key("", 0, &ev), 0u);
}

TEST(control_characters_are_not_characters) {
    ASSERT_EQ(one("\x7f").key, UiKey::Other);   // backspace
    ASSERT_EQ(one("\x03").key, UiKey::Other);   // ctrl-c
}

// --- the vim style ----------------------------------------------------------

TEST(vim_keys) {
    ASSERT_EQ(oflm::config::vim_key_for_char('j'), UiKey::Down);
    ASSERT_EQ(oflm::config::vim_key_for_char('k'), UiKey::Up);
    ASSERT_EQ(oflm::config::vim_key_for_char('h'), UiKey::Left);
    ASSERT_EQ(oflm::config::vim_key_for_char('l'), UiKey::Right);
    ASSERT_EQ(oflm::config::vim_key_for_char('q'), UiKey::None);
    ASSERT_EQ(oflm::config::vim_key_for_char('J'), UiKey::None);
    ASSERT_EQ(oflm::config::vim_key_for_char(' '), UiKey::None);
}

int main() {
    std::cout << "Running " << tests.size() << " tests..." << std::endl;
    int passed = 0;
    for (auto& [name, fn] : tests) {
        failures = 0;
        fn();
        if (failures == 0) {
            std::cout << "  PASS: " << name << std::endl;
            passed++;
        } else {
            std::cout << "  FAIL: " << name << " (" << failures << " failures)" << std::endl;
        }
    }
    std::cout << passed << "/" << tests.size() << " tests passed" << std::endl;
    return passed == (int)tests.size() ? 0 : 1;
}
