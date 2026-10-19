/// \file settings_registry_test.cpp
/// \brief Unit tests for the settings registry
#include "config/settings_registry.hpp"
#include <cassert>
#include <iostream>
#include <set>

#define TEST(name) static void name(); \
    static struct name##_registrar { name##_registrar() { tests.push_back({#name, name}); } } name##_instance; \
    static void name()

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "FAIL: " << #expr << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "FAIL: " << #a << " != " << #b << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)
#define ASSERT_NE(a, b) do { if ((a) == (b)) { std::cerr << "FAIL: " << #a << " == " << #b << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)

static int failures = 0;
static std::vector<std::pair<std::string, void(*)()>> tests;

TEST(registry_has_5_settings) {
    ASSERT_EQ(oflm::config::settings_registry().size(), 5u);
}

TEST(every_setting_has_required_fields) {
    for (const auto& def : oflm::config::settings_registry()) {
        ASSERT_FALSE(def.key.empty());
        ASSERT_FALSE(def.description.empty());
        ASSERT_FALSE(def.category.empty());
        // delivery is an enum, always set
    }
}

TEST(enum_settings_have_options) {
    for (const auto& def : oflm::config::settings_registry()) {
        if (def.type == oflm::config::SettingType::ENUM) {
            ASSERT_TRUE(def.options.size() >= 2);
            for (const auto& opt : def.options) {
                ASSERT_FALSE(opt.value.empty());
                ASSERT_FALSE(opt.description.empty());
            }
        }
    }
}

TEST(int_settings_have_min_value) {
    for (const auto& def : oflm::config::settings_registry()) {
        if (def.type == oflm::config::SettingType::INT) {
            ASSERT_TRUE(def.min_value.has_value());
        }
    }
}

TEST(env_var_delivery_has_env_var) {
    for (const auto& def : oflm::config::settings_registry()) {
        if (def.delivery == oflm::config::DeliveryMechanism::ENV_VAR) {
            ASSERT_FALSE(def.env_var.empty());
        }
    }
}

TEST(prefill_mode_is_env_var_delivery) {
    auto def = oflm::config::find_setting("prefill_mode");
    ASSERT_TRUE(def.has_value());
    ASSERT_EQ(def->delivery, oflm::config::DeliveryMechanism::ENV_VAR);
    ASSERT_EQ(def->env_var, std::string("OFLM_OPEN_PREFILL_MODE"));
    ASSERT_EQ(def->scope, oflm::config::SettingScope::PER_MODEL);
    ASSERT_TRUE(def->reload_required);
}

TEST(pmode_is_global) {
    auto def = oflm::config::find_setting("pmode");
    ASSERT_TRUE(def.has_value());
    ASSERT_EQ(def->scope, oflm::config::SettingScope::GLOBAL);
    ASSERT_EQ(def->delivery, oflm::config::DeliveryMechanism::DIRECT);
    ASSERT_EQ(def->options.size(), 5u); // default, powersaver, balanced, performance, turbo
}

TEST(ctx_len_has_registry_key) {
    auto def = oflm::config::find_setting("ctx_len");
    ASSERT_TRUE(def.has_value());
    ASSERT_EQ(def->registry_key, std::string("default_context_length"));
    ASSERT_EQ(def->type, oflm::config::SettingType::INT);
    ASSERT_EQ(def->min_value.value(), 512);
}

TEST(prefill_chunk_len_has_registry_key) {
    auto def = oflm::config::find_setting("prefill_chunk_len");
    ASSERT_TRUE(def.has_value());
    ASSERT_EQ(def->registry_key, std::string("max_prefill_len"));
}

TEST(think_is_api_field) {
    auto def = oflm::config::find_setting("think");
    ASSERT_TRUE(def.has_value());
    ASSERT_EQ(def->delivery, oflm::config::DeliveryMechanism::API_FIELD);
    ASSERT_EQ(def->type, oflm::config::SettingType::BOOL);
}

TEST(find_setting_returns_nullopt_for_unknown) {
    auto def = oflm::config::find_setting("nonexistent_setting");
    ASSERT_FALSE(def.has_value());
}

TEST(settings_for_scope_global) {
    auto globals = oflm::config::settings_for_scope(oflm::config::SettingScope::GLOBAL);
    ASSERT_EQ(globals.size(), 1u); // only pmode
    ASSERT_EQ(globals[0].key, std::string("pmode"));
}

TEST(settings_for_scope_per_model) {
    auto per_model = oflm::config::settings_for_scope(oflm::config::SettingScope::PER_MODEL);
    ASSERT_EQ(per_model.size(), 4u); // prefill_mode, ctx_len, prefill_chunk_len, think
}

TEST(generate_schema_json_structure) {
    auto schema = oflm::config::generate_schema_json();
    ASSERT_TRUE(schema.contains("version"));
    ASSERT_EQ(schema["version"].get<int>(), 1);
    ASSERT_TRUE(schema.contains("settings"));
    ASSERT_TRUE(schema["settings"].is_array());
    ASSERT_EQ(schema["settings"].size(), 5u);
    // Each setting should have key, type, description, scope, delivery
    for (const auto& s : schema["settings"]) {
        ASSERT_TRUE(s.contains("key"));
        ASSERT_TRUE(s.contains("type"));
        ASSERT_TRUE(s.contains("description"));
        ASSERT_TRUE(s.contains("scope"));
        ASSERT_TRUE(s.contains("delivery"));
    }
}

TEST(generate_schema_json_enum_has_options) {
    auto schema = oflm::config::generate_schema_json();
    for (const auto& s : schema["settings"]) {
        if (s["type"] == "enum") {
            ASSERT_TRUE(s.contains("options"));
            ASSERT_TRUE(s["options"].is_array());
            ASSERT_TRUE(s["options"].size() >= 2);
        }
    }
}

TEST(generate_schema_json_delivery_field) {
    auto schema = oflm::config::generate_schema_json();
    bool found_env_var = false;
    for (const auto& s : schema["settings"]) {
        if (s["delivery"] == "env_var") {
            found_env_var = true;
            ASSERT_TRUE(s.contains("env_var"));
        }
    }
    ASSERT_TRUE(found_env_var);
}

TEST(compute_defaults_version_stable) {
    nlohmann::json registry = nlohmann::json::object();
    registry["defaults"] = {{{"pmode", "performance"}}};
    registry["models"] = nlohmann::json::object();
    auto v1 = oflm::config::compute_defaults_version(registry);
    auto v2 = oflm::config::compute_defaults_version(registry);
    ASSERT_EQ(v1, v2);
    ASSERT_FALSE(v1.empty());
}

TEST(compute_defaults_version_changes_with_defaults) {
    nlohmann::json reg1 = nlohmann::json::object();
    reg1["defaults"] = {{"pmode", "performance"}};
    reg1["models"] = nlohmann::json::object();
    
    nlohmann::json reg2 = nlohmann::json::object();
    reg2["defaults"] = {{"pmode", "balanced"}};
    reg2["models"] = nlohmann::json::object();
    auto v1 = oflm::config::compute_defaults_version(reg1);
    auto v2 = oflm::config::compute_defaults_version(reg2);
    ASSERT_NE(v1, v2);
}

TEST(convert_type_int) {
    auto j = oflm::config::convert_type("8192", oflm::config::SettingType::INT);
    ASSERT_TRUE(j.is_number());
    ASSERT_EQ(j.get<int>(), 8192);
}

TEST(convert_type_bool) {
    ASSERT_TRUE(oflm::config::convert_type("true", oflm::config::SettingType::BOOL).get<bool>());
    ASSERT_TRUE(oflm::config::convert_type("1", oflm::config::SettingType::BOOL).get<bool>());
    ASSERT_TRUE(oflm::config::convert_type("yes", oflm::config::SettingType::BOOL).get<bool>());
    ASSERT_FALSE(oflm::config::convert_type("false", oflm::config::SettingType::BOOL).get<bool>());
    ASSERT_FALSE(oflm::config::convert_type("0", oflm::config::SettingType::BOOL).get<bool>());
}

TEST(convert_type_enum) {
    auto j = oflm::config::convert_type("lean", oflm::config::SettingType::ENUM);
    ASSERT_TRUE(j.is_string());
    ASSERT_EQ(j.get<std::string>(), "lean");
}

TEST(get_all_setting_keys) {
    auto keys = oflm::config::get_all_setting_keys();
    ASSERT_EQ(keys.size(), 5u);
    std::set<std::string> key_set(keys.begin(), keys.end());
    ASSERT_TRUE(key_set.count("prefill_mode") == 1);
    ASSERT_TRUE(key_set.count("pmode") == 1);
    ASSERT_TRUE(key_set.count("ctx_len") == 1);
    ASSERT_TRUE(key_set.count("prefill_chunk_len") == 1);
    ASSERT_TRUE(key_set.count("think") == 1);
}

TEST(generate_help_lines) {
    auto lines = oflm::config::generate_help_lines();
    ASSERT_FALSE(lines.empty());
    // Should contain all 5 setting keys
    std::string all_text;
    for (const auto& l : lines) all_text += l + "\n";
    ASSERT_TRUE(all_text.find("prefill_mode") != std::string::npos);
    ASSERT_TRUE(all_text.find("pmode") != std::string::npos);
    ASSERT_TRUE(all_text.find("ctx_len") != std::string::npos);
    ASSERT_TRUE(all_text.find("prefill_chunk_len") != std::string::npos);
    ASSERT_TRUE(all_text.find("think") != std::string::npos);
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
