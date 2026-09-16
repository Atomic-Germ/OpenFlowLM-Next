/// \file test_downloader_pinning.cpp
/// \brief Pinned per-file download sources (no network, no NPU)
/// \author OpenFlowLM Team
/// \note Covers resolve_file_source / uses_pinned_sources: the table
///       validation, URL formation, percent-encoding, and the legacy
///       fallback. The strict download/verify behavior keyed off the pinning
///       needs a model tree and is exercised by oflm-add's installer tests.
#include "model_downloader.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#define TEST_REQUIRE(condition)                                                   \
    do {                                                                          \
        if (!(condition)) {                                                        \
            throw std::runtime_error(std::string("requirement failed: ") +       \
                                     #condition);                                  \
        }                                                                         \
    } while (false)

inline void RequireContains(std::string_view text, std::string_view expected) {
    if (text.find(expected) == std::string_view::npos) {
        throw std::runtime_error("expected '" + std::string(text) +
                                 "' to contain '" + std::string(expected) + "'");
    }
}

template <typename Exception = std::exception, typename Callable>
std::string RequireThrows(Callable&& callable) {
    try {
        callable();
    } catch (const Exception& error) {
        return error.what();
    }
    throw std::runtime_error("expected exception was not thrown");
}

inline void RunTest(void (*test)(), const char* name) {
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        std::exit(1);
    }
}

namespace {

constexpr const char* kRevA = "0123456789abcdef0123456789abcdef01234567";
constexpr const char* kRevB = "fedcba9876543210fedcba9876543210fedcba98";

nlohmann::json BaseInfo() {
    return {
        {"url", "https://huggingface.co/SomeOrg/SomeModel"},
        {"ms_url", "https://modelscope.cn/models/some/SomeModel"},
        {"files", {"config.json", "model.gguf"}},
    };
}

nlohmann::json PinnedInfo() {
    auto info = BaseInfo();
    info["file_sources"] = {
        {"model.gguf", {{"url", "https://huggingface.co/SomeOrg/SomeModel-GGUF"},
                        {"revision", kRevA}}},
    };
    return info;
}

void test_gate_fires_only_on_non_empty_table() {
    TEST_REQUIRE(!uses_pinned_sources(BaseInfo()));
    auto empty = BaseInfo();
    empty["file_sources"] = nlohmann::json::object();
    TEST_REQUIRE(!uses_pinned_sources(empty));
    auto wrong_type = BaseInfo();
    wrong_type["file_sources"] = nlohmann::json::array();
    TEST_REQUIRE(!uses_pinned_sources(wrong_type));
    TEST_REQUIRE(uses_pinned_sources(PinnedInfo()));
}

void test_legacy_fallback_urls() {
    const auto info = BaseInfo();
    const auto main = resolve_file_source(info, "config.json", false);
    TEST_REQUIRE(main.url ==
                 "https://huggingface.co/SomeOrg/SomeModel/resolve/main/config.json?download=true");
    TEST_REQUIRE(main.revision.empty());

    const auto ms = resolve_file_source(info, "config.json", true);
    TEST_REQUIRE(ms.url ==
                 "https://modelscope.cn/models/some/SomeModel/resolve/main/config.json?download=true");

    auto branched = BaseInfo();
    branched["url"] = "https://huggingface.co/SomeOrg/SomeModel/resolve/refs%2Fpr%2F3";
    const auto branch = resolve_file_source(branched, "config.json", false);
    TEST_REQUIRE(branch.url ==
                 "https://huggingface.co/SomeOrg/SomeModel/resolve/refs%2Fpr%2F3/config.json?download=true");
}

void test_pin_forms_revision_url() {
    const auto info = PinnedInfo();
    const auto pinned = resolve_file_source(info, "model.gguf", false);
    TEST_REQUIRE(pinned.url == std::string("https://huggingface.co/SomeOrg/SomeModel-GGUF/resolve/") +
                               kRevA + "/model.gguf?download=true");
    TEST_REQUIRE(pinned.revision == kRevA);

    // A file with no pin falls back to the entry URL even when the table exists.
    const auto plain = resolve_file_source(info, "config.json", false);
    TEST_REQUIRE(plain.revision.empty());
    RequireContains(plain.url, "/resolve/main/config.json");
}

void test_pin_percent_encodes_filenames() {
    auto info = BaseInfo();
    info["files"] = {"my model.gguf"};
    info["file_sources"] = {
        {"my model.gguf", {{"url", "https://huggingface.co/SomeOrg/M"}, {"revision", kRevB}}},
    };
    const auto pinned = resolve_file_source(info, "my model.gguf", false);
    RequireContains(pinned.url, "my%20model.gguf");
    TEST_REQUIRE(pinned.url.find(' ') == std::string::npos);
}

void test_modelscope_with_pins_is_rejected() {
    const auto info = PinnedInfo();
    const std::string message =
        RequireThrows([&] { resolve_file_source(info, "model.gguf", true); });
    RequireContains(message, "--modelscope");

    // ...but an empty table pins nothing, so mirrors stay usable.
    auto empty = BaseInfo();
    empty["file_sources"] = nlohmann::json::object();
    TEST_REQUIRE(!resolve_file_source(empty, "config.json", true).url.empty());
}

void test_malformed_tables_throw() {
    auto wrong_type = BaseInfo();
    wrong_type["file_sources"] = nlohmann::json::array();
    RequireThrows([&] { resolve_file_source(wrong_type, "config.json", false); });

    auto unknown_key = PinnedInfo();
    unknown_key["file_sources"]["nope.bin"] = {{"url", "https://huggingface.co/X"},
                                               {"revision", kRevA}};
    RequireThrows([&] { resolve_file_source(unknown_key, "model.gguf", false); });

    auto missing_url = PinnedInfo();
    missing_url["file_sources"]["model.gguf"] = {{"revision", kRevA}};
    RequireThrows([&] { resolve_file_source(missing_url, "model.gguf", false); });

    auto empty_url = PinnedInfo();
    empty_url["file_sources"]["model.gguf"] = {{"url", ""}, {"revision", kRevA}};
    RequireThrows([&] { resolve_file_source(empty_url, "model.gguf", false); });

    auto extra_key = PinnedInfo();
    extra_key["file_sources"]["model.gguf"] = {{"url", "https://huggingface.co/X"},
                                               {"revision", kRevA},
                                               {"mirror", "y"}};
    RequireThrows([&] { resolve_file_source(extra_key, "model.gguf", false); });

    auto short_rev = PinnedInfo();
    short_rev["file_sources"]["model.gguf"] = {{"url", "https://huggingface.co/X"},
                                               {"revision", "abc123"}};
    const std::string short_msg =
        RequireThrows([&] { resolve_file_source(short_rev, "model.gguf", false); });
    RequireContains(short_msg, "40-character");

    auto non_hex_rev = PinnedInfo();
    non_hex_rev["file_sources"]["model.gguf"] = {
        {"url", "https://huggingface.co/X"},
        {"revision", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"}};
    RequireThrows([&] { resolve_file_source(non_hex_rev, "model.gguf", false); });

    // The whole table is validated even when the requested file has a pin:
    // a broken sibling must not ship silently.
    auto broken_sibling = BaseInfo();
    broken_sibling["file_sources"] = {
        {"config.json", {{"url", "https://huggingface.co/X"}, {"revision", kRevA}}},
        {"model.gguf", {{"url", "https://huggingface.co/X"}, {"revision", "nope"}}},
    };
    RequireThrows([&] { resolve_file_source(broken_sibling, "config.json", false); });
}

}  // namespace

int main() {
    RunTest(test_gate_fires_only_on_non_empty_table, "gate fires only on non-empty table");
    RunTest(test_legacy_fallback_urls, "legacy fallback urls");
    RunTest(test_pin_forms_revision_url, "pin forms revision url");
    RunTest(test_pin_percent_encodes_filenames, "pin percent-encodes filenames");
    RunTest(test_modelscope_with_pins_is_rejected, "modelscope with pins is rejected");
    RunTest(test_malformed_tables_throw, "malformed tables throw");
    std::cout << "All downloader pinning tests passed\n";
    return 0;
}
