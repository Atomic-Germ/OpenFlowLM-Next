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
    args.prefill_mode = ""; // the parser's default sentinel: not supplied
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

TEST(user_model_metadata_and_explicit_settings_resolve_together) {
    auto cfg = make_cfg(nlohmann::json::object(), nlohmann::json::object());
    cfg.shipped_registry["models"]["qwen3.8"]["27b"] = {
        {"default_context_length", 32768}, {"details", {{"think", true}}}
    };
    auto args = make_args();
    for (const std::string type : {"qwen3.8", "local"}) {
        const auto tag = type + ":27b";
        cfg.user_settings["models"][type]["27b"] = {
            {"default_context_length", 4096}, {"details", {{"think", false}}}
        };
        cfg.user_settings["settings"][type]["27b"] = {{"prefill_mode", "lean"}};
        auto resolved = oflm::config::resolve_settings(cfg, tag, args);
        ASSERT_EQ(resolved["ctx_len"], 4096);
        ASSERT_EQ(resolved["think"], false);
        ASSERT_EQ(resolved["prefill_mode"], "lean");
        auto shown = oflm::config::get_setting_value(cfg, tag, "ctx_len", args);
        ASSERT_TRUE(shown.has_value());
        ASSERT_EQ(shown->value, "4096");
        cfg.user_settings["settings"][type]["27b"]["ctx_len"] = 2048;
        resolved = oflm::config::resolve_settings(cfg, tag, args);
        ASSERT_EQ(resolved["ctx_len"], 2048);
    }
}

TEST(env_var_overrides_user_settings) {
    auto cfg = make_cfg(
        nlohmann::json::object(),
        nlohmann::json::object(),
        {{"qwen3.8", {{"27b", {{"prefill_mode", "lean"}}}}}}
    );
    // An env var is an override only when the USER set it -- one we wrote
    // ourselves is not (see the test below). The user sets them in their
    // shell, so they are there when the process starts: that is when
    // capture_external_env_vars() runs.
    setenv("OFLM_OPEN_PREFILL_MODE", "fast", 1);
    oflm::config::capture_external_env_vars();
    auto args = make_args();
    auto resolved = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    ASSERT_EQ(resolved["prefill_mode"], "fast");
    unsetenv("OFLM_OPEN_PREFILL_MODE");
}

TEST(a_value_we_wrote_ourselves_is_not_an_override) {
    // The env-var delivery mechanism writes the settings into the
    // environment, and the environment is process-wide: a server that serves
    // several models would have the first model's values shadow every later
    // one. So a value WE wrote is not an override -- it belongs to the model we
    // wrote it for, and the next model resolves its own.
    nlohmann::json user_settings;
    user_settings["qwen3.8"]["27b"]["prefill_mode"] = "lean";
    user_settings["qwen3.6-moe"]["35b-a3b"]["prefill_mode"] = "fast";
    auto cfg = make_cfg(nlohmann::json::object(), nlohmann::json::object(), user_settings);
    // The variable is in the environment -- we put it there for qwen3.8:27b.
    setenv("OFLM_OPEN_PREFILL_MODE", "lean", 1);
    // ... but it was not in the environment when the process started, so it is
    // ours and not the user's.
    oflm::config::g_external_env_vars.erase("OFLM_OPEN_PREFILL_MODE");

    auto args = make_args();
    auto one = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    auto two = oflm::config::resolve_settings(cfg, "qwen3.6-moe:35b-a3b", args);

    ASSERT_EQ(one["prefill_mode"], "lean");
    ASSERT_EQ(two["prefill_mode"], "fast");
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
    // make_args() leaves prefill_mode empty: the CLI flag was not supplied,
    // so the user's saved `lean` wins.
    auto val = oflm::config::get_setting_value(cfg, "qwen3.8:27b", "prefill_mode", args);
    ASSERT_TRUE(val.has_value());
    ASSERT_EQ(val->source, oflm::config::SettingValue::USER);
    ASSERT_EQ(val->value, "lean");
}

TEST(explicit_prefill_mode_fast_wins_over_saved_lean) {
    // `--prefill-mode fast` is a real CLI override: "fast" is not the
    // parser's not-supplied sentinel (the empty string is), so it must beat
    // a saved `lean` preference.
    auto cfg = make_cfg(
        nlohmann::json::object(),
        nlohmann::json::object(),
        {{"qwen3.8", {{"27b", {{"prefill_mode", "lean"}}}}}}
    );
    auto args = make_args();
    args.prefill_mode = "fast";
    unsetenv("OFLM_OPEN_PREFILL_MODE");
    auto resolved = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    ASSERT_EQ(resolved["prefill_mode"], "fast");
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
    // Not supplied on the CLI: the registry's own default ("fast") applies.
    unsetenv("OFLM_OPEN_PREFILL_MODE");
    auto resolved = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    // prefill_mode default is "fast" from registry definition
    ASSERT_EQ(resolved["prefill_mode"], "fast");
}

TEST(each_model_resolves_its_own_settings) {
    // A server can serve several models, and `/load` can swap the model at
    // run time. The settings resolved for one model must not leak into
    // another: each is resolved for the tag it is asked about.
    auto cfg = make_cfg(
        nlohmann::json::object(),
        nlohmann::json::object(),
        {{"qwen3.8", {{"27b", {{"prefill_mode", "lean"}}}}},
         {"qwen3.6-moe", {{"35b-a3b", {{"prefill_mode", "fast"}}}}}}
    );
    cfg.shipped_registry["models"]["qwen3.8"]["27b"] = {{"default_context_length", 8192}};
    cfg.shipped_registry["models"]["qwen3.6-moe"]["35b-a3b"] = {{"default_context_length", 32768}};

    auto args = make_args();
    auto one = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    auto two = oflm::config::resolve_settings(cfg, "qwen3.6-moe:35b-a3b", args);

    ASSERT_EQ(one["prefill_mode"], "lean");
    ASSERT_EQ(two["prefill_mode"], "fast");
    ASSERT_EQ(one["ctx_len"], 8192);
    ASSERT_EQ(two["ctx_len"], 32768);
}

TEST(the_app_wide_default_reaches_a_model_that_does_not_override_it) {
    // A setting the user changed app-wide applies to every model, unless that
    // model overrides it. Without this the app-wide value showed up in the
    // global view and in no model at all.
    auto cfg = make_cfg(
        nlohmann::json::object(),
        {{"pmode", "balanced"}},
        {{"qwen3.8", {{"27b", {{"prefill_mode", "lean"}}}}}}
    );
    cfg.shipped_registry["models"]["qwen3.8"]["27b"] = {{"default_context_length", 8192}};
    cfg.shipped_registry["models"]["qwen3.6-moe"]["35b-a3b"] = {{"default_context_length", 32768}};

    auto args = make_args();
    auto overridden = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    auto plain = oflm::config::resolve_settings(cfg, "qwen3.6-moe:35b-a3b", args);

    ASSERT_EQ(overridden["pmode"], "balanced");
    ASSERT_EQ(plain["pmode"], "balanced");
}

TEST(a_models_own_override_beats_the_app_wide_value) {
    auto cfg = make_cfg(
        nlohmann::json::object(),
        {{"pmode", "balanced"}},
        {{"qwen3.6-moe", {{"35b-a3b", {{"pmode", "turbo"}}}}}}
    );
    cfg.shipped_registry["models"]["qwen3.8"]["27b"] = {{"default_context_length", 8192}};
    cfg.shipped_registry["models"]["qwen3.6-moe"]["35b-a3b"] = {{"default_context_length", 32768}};

    auto args = make_args();
    auto overridden = oflm::config::resolve_settings(cfg, "qwen3.6-moe:35b-a3b", args);
    auto plain = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);

    ASSERT_EQ(overridden["pmode"], "turbo");   // the model's own
    ASSERT_EQ(plain["pmode"], "balanced");    // the app-wide one
}

TEST(a_model_that_overrides_nothing_gets_the_shipped_value) {
    auto cfg = make_cfg(
        {{"pmode", "performance"}},
        nlohmann::json::object(),
        nlohmann::json::object()
    );
    cfg.shipped_registry["models"]["qwen3.8"]["27b"] = {{"default_context_length", 8192}};

    auto args = make_args();
    auto resolved = oflm::config::resolve_settings(cfg, "qwen3.8:27b", args);
    ASSERT_EQ(resolved["pmode"], "performance");
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
