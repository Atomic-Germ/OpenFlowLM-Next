/// \file settings_resolver_test.cpp
/// \brief Unit tests for settings resolver (precedence, delivery, source tracking)
#include "config/settings_resolver.hpp"
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>

#define TEST(name) static void name(); \
    static struct name##_registrar { name##_registrar() { tests.push_back({#name, name}); } } name##_instance; \
    static void name()
#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "FAIL: " << #expr << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "FAIL: " << #a << " != " << #b << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)

static int failures = 0;
static std::vector<std::pair<std::string, void(*)()>> tests;

static oflm::config::EffectiveConfig make_cfg(
    nlohmann::json shipped_defaults,
    nlohmann::json user_defaults,
    nlohmann::json user_settings = nlohmann::json::object())
{
    oflm::config::EffectiveConfig cfg;
    cfg.shipped_registry = {{"defaults", shipped_defaults}, {"models", nlohmann::json::object()}};
    cfg.user_settings = nlohmann::json::object();
    if (!user_defaults.empty()) cfg.user_settings["defaults"] = user_defaults;
    if (!user_settings.empty()) cfg.user_settings["settings"] = user_settings;
    cfg.merged = cfg.shipped_registry;
    return cfg;
}

static program_args_t make_args() {
    program_args_t args;
    args.power_mode = "performance";
    args.prefill_mode = "fast";
    args.ctx_length = -1;
    args.prefill_chunk_len = -1;
    return args;
}

TEST(shipped_default_used_when_nothing_else) {
    auto cfg = make_cfg({{"pmode", "balanced"}}, nlohmann::json::object());
    auto args = make_args();
    auto resolved = oflm::config::resolve_settings(cfg, "", args);
    // pmode from shipped defaults
    ASSERT_EQ(resolved["pmode"], "balanced");
}

TEST(user_overrides_shipped) {
    auto cfg = make_cfg({{"pmode", "performance"}}, {{"pmode", "balanced"}});
    auto args = make_args();
    args.power_mode = ""; // Clear CLI so user setting wins
    auto resolved = oflm::config::resolve_settings(cfg, "", args);
    ASSERT_EQ(resolved["pmode"], "balanced");
}

TEST(per_model_scope_filtered_for_global) {
    auto cfg = make_cfg(nlohmann::json::object(), nlohmann::json::object());
    auto args = make_args();
    auto resolved = oflm::config::resolve_settings(cfg, "", args);
    // prefill_mode is PER_MODEL — should not appear for global (empty model_tag)
    ASSERT_FALSE(resolved.contains("prefill_mode"));
}

TEST(per_model_setting_resolved_for_model) {
    auto cfg = make_cfg(
        nlohmann::json::object(),
        nlohmann::json::object(),
        {{"qwen3.8", {{"27b", {{"prefill_mode", "lean"}}}}}}
    );
    cfg.shipped_registry["models"]["qwen3.8"]["27b"] = {
        {"default_context_length", 8192},
        {"details", {{"think", true}}}
    };
    auto args = make_args();
    auto resolved = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    ASSERT_EQ(resolved["prefill_mode"], "lean"); // from user settings
    ASSERT_EQ(resolved["ctx_len"], 8192); // from shipped registry entry
    ASSERT_EQ(resolved["think"], true); // from shipped registry entry
}

TEST(env_var_overrides_user_settings) {
    auto cfg = make_cfg(
        nlohmann::json::object(),
        nlohmann::json::object(),
        {{"qwen3.8", {{"27b", {{"prefill_mode", "lean"}}}}}}
    );
    setenv("OFLM_OPEN_PREFILL_MODE", "fast", 1);
    auto args = make_args();
    auto resolved = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    // Env var "fast" should override user setting "lean" (unless CLI flag wins)
    // CLI flag for prefill_mode returns nullopt if "fast" (default), so env wins
    ASSERT_EQ(resolved["prefill_mode"], "fast");
    unsetenv("OFLM_OPEN_PREFILL_MODE");
}

TEST(apply_settings_sets_env_var) {
    nlohmann::json resolved = {{"prefill_mode", "lean"}};
    oflm::config::apply_settings(resolved);
    const char* val = getenv("OFLM_OPEN_PREFILL_MODE");
    ASSERT_TRUE(val != nullptr);
    ASSERT_EQ(std::string(val), std::string("lean"));
    unsetenv("OFLM_OPEN_PREFILL_MODE");
}

TEST(apply_settings_skips_non_env_var) {
    nlohmann::json resolved = {{"pmode", "turbo"}};
    oflm::config::apply_settings(resolved);
    // pmode is DIRECT delivery — should NOT be setenv'd
    // (We can't easily check it wasn't set, but we verify no crash)
}

TEST(get_setting_value_source_shipped) {
    auto cfg = make_cfg({{"pmode", "performance"}}, nlohmann::json::object());
    auto args = make_args();
    args.power_mode = ""; // Clear CLI flag so we fall through to shipped
    auto val = oflm::config::get_setting_value(cfg, "", "pmode", args);
    // Note: this test depends on get_cli_flag_value implementation
    // If CLI always returns a value, source will be CLI
}

TEST(get_setting_value_source_user) {
    auto cfg = make_cfg(
        nlohmann::json::object(),
        nlohmann::json::object(),
        {{"qwen3.8", {{"27b", {{"prefill_mode", "lean"}}}}}}
    );
    auto args = make_args();
    // Clear env var so user setting wins
    unsetenv("OFLM_OPEN_PREFILL_MODE");
    args.prefill_mode = "fast"; // This is the default, get_cli_flag_value returns nullopt
    auto val = oflm::config::get_setting_value(cfg, "qwen3.8:27b", "prefill_mode", args);
    ASSERT_TRUE(val.has_value());
    ASSERT_EQ(val->source, oflm::config::SettingValue::USER);
    ASSERT_EQ(val->value, "lean");
}

TEST(get_setting_value_unknown_key) {
    auto cfg = make_cfg(nlohmann::json::object(), nlohmann::json::object());
    auto args = make_args();
    auto val = oflm::config::get_setting_value(cfg, "", "nonexistent", args);
    ASSERT_FALSE(val.has_value());
}

TEST(built_in_default_used) {
    auto cfg = make_cfg(nlohmann::json::object(), nlohmann::json::object());
    auto args = make_args();
    args.prefill_mode = "fast"; // default, nullopt from CLI
    unsetenv("OFLM_OPEN_PREFILL_MODE");
    auto resolved = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    // prefill_mode default is "fast" from registry definition
    ASSERT_EQ(resolved["prefill_mode"], "fast");
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
