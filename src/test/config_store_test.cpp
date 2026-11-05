/// \file config_store_test.cpp
/// \brief Unit tests for the config store (merge, migration, lock, corrupted JSON)
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#define TEST(name) static void name(); \
    static struct name##_registrar { name##_registrar() { tests.push_back({#name, name}); } } name##_instance; \
    static void name()

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "FAIL: " << #expr << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "FAIL: " << #a << " != " << #b << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)

static int failures = 0;
static std::vector<std::pair<std::string, void(*)()>> tests;

static std::string make_temp_dir() {
    auto dir = std::filesystem::temp_directory_path() / ("oflm_test_" + std::to_string(rand()));
    std::filesystem::create_directories(dir);
    return dir.string();
}

static void write_json(const std::string& path, const nlohmann::json& j) {
    std::ofstream f(path);
    f << j.dump(2);
}

// ============================================================================
// deep_merge tests
// ============================================================================

TEST(deep_merge_objects_recursive) {
    nlohmann::json base = {{"a", {{"b", 1}, {"c", 2}}}};
    nlohmann::json overlay = {{"a", {{"b", 10}}}};
    auto result = oflm::config::deep_merge(base, overlay);
    ASSERT_EQ(result["a"]["b"], 10);
    ASSERT_EQ(result["a"]["c"], 2);
}

TEST(deep_merge_arrays_replace) {
    nlohmann::json base = {{"files", nlohmann::json::array({"a.txt", "b.txt"})}};
    nlohmann::json overlay = {{"files", nlohmann::json::array({"c.txt"})}};
    auto result = oflm::config::deep_merge(base, overlay);
    ASSERT_EQ(result["files"].size(), 1u);
    ASSERT_EQ(result["files"][0], "c.txt");
}

TEST(deep_merge_scalars_replace) {
    nlohmann::json base = {{"key", "old"}};
    nlohmann::json overlay = {{"key", "new"}};
    auto result = oflm::config::deep_merge(base, overlay);
    ASSERT_EQ(result["key"], "new");
}

TEST(deep_merge_new_keys_added) {
    nlohmann::json base = {{"a", 1}};
    nlohmann::json overlay = {{"b", 2}};
    auto result = oflm::config::deep_merge(base, overlay);
    ASSERT_EQ(result["a"], 1);
    ASSERT_EQ(result["b"], 2);
}

TEST(deep_merge_null_does_not_overwrite) {
    nlohmann::json base = {{"key", "value"}};
    nlohmann::json overlay = {{"key", nullptr}};
    auto result = oflm::config::deep_merge(base, overlay);
    ASSERT_EQ(result["key"], "value");
}

TEST(deep_merge_nested_objects) {
    nlohmann::json base = {{"details", {{"think", true}, {"family", "qwen3"}}}};
    nlohmann::json overlay = {{"details", {{"think", false}}}};
    auto result = oflm::config::deep_merge(base, overlay);
    ASSERT_EQ(result["details"]["think"], false);
    ASSERT_EQ(result["details"]["family"], "qwen3");
}

// ============================================================================
// split_tag tests
// ============================================================================

TEST(split_tag_with_size) {
    auto [type, size] = oflm::config::split_tag("qwen3.8:27b");
    ASSERT_EQ(type, std::string("qwen3.8"));
    ASSERT_EQ(size, std::string("27b"));
}

TEST(split_tag_without_size) {
    auto [type, size] = oflm::config::split_tag("qwen3.8");
    ASSERT_EQ(type, std::string("qwen3.8"));
    ASSERT_EQ(size, std::string(""));
}

// ============================================================================
// SettingsLock tests
// ============================================================================

TEST(settings_lock_acquires) {
    auto dir = make_temp_dir();
    std::string path = dir + "/settings.json";
    write_json(path, {{"test", true}});
    {
        oflm::config::SettingsLock lock(path);
        ASSERT_TRUE(lock.locked());
    }
    std::filesystem::remove_all(dir);
}

TEST(settings_lock_releases) {
    auto dir = make_temp_dir();
    std::string path = dir + "/settings.json";
    write_json(path, {{"test", true}});
    {
        oflm::config::SettingsLock lock1(path);
        ASSERT_TRUE(lock1.locked());
    }
    {
        oflm::config::SettingsLock lock2(path);
        ASSERT_TRUE(lock2.locked());
    }
    std::filesystem::remove_all(dir);
}

// ============================================================================
// save_user_settings tests
// ============================================================================

TEST(save_user_settings_writes_valid_json) {
    auto dir = make_temp_dir();
    std::string path = dir + "/settings.json";
    nlohmann::json settings = {{"version", 1}, {"defaults", {{"pmode", "balanced"}}}};
    oflm::config::save_user_settings(settings, path);

    std::ifstream f(path);
    auto loaded = nlohmann::json::parse(f);
    ASSERT_EQ(loaded["version"], 1);
    ASSERT_EQ(loaded["defaults"]["pmode"], "balanced");

    std::filesystem::remove_all(dir);
}

TEST(save_user_settings_is_atomic_no_temp_left) {
    auto dir = make_temp_dir();
    std::string path = dir + "/settings.json";
    oflm::config::save_user_settings({{"a", 1}}, path);

    int file_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        file_count++;
    }
    ASSERT_EQ(file_count, 1);

    std::filesystem::remove_all(dir);
}

// ============================================================================
// migrate_legacy_user_registry tests
// ============================================================================

TEST(migration_user_added_model) {
    auto dir = make_temp_dir();
    std::string legacy_path = dir + "/model_list.json";
    std::string settings_path = dir + "/settings.json";

    nlohmann::json shipped = {{"models", {{"qwen3.8", {{"27b", {{"name", "qwen3.8-27b"}}}}}}}};
    nlohmann::json legacy = {{"models", {
        {"qwen3.8", {{"27b", {{"name", "qwen3.8-27b"}}}}},
        {"my-model", {{"1b", {{"name", "my-model-1b"}, {"url", "https://example.com"}}}}}
    }}};
    write_json(legacy_path, legacy);

    auto log = oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);

    ASSERT_TRUE(std::filesystem::exists(settings_path));
    std::ifstream f(settings_path);
    auto settings = nlohmann::json::parse(f);

    ASSERT_TRUE(settings["models"]["my-model"]["1b"].contains("url"));
    ASSERT_FALSE(std::filesystem::exists(legacy_path));
    ASSERT_TRUE(std::filesystem::exists(legacy_path + ".bak"));

    std::filesystem::remove_all(dir);
}

TEST(migration_modified_entry_preserved_as_override) {
    auto dir = make_temp_dir();
    std::string legacy_path = dir + "/model_list.json";
    std::string settings_path = dir + "/settings.json";

    nlohmann::json shipped = {{"models", {{"qwen3.8", {{"27b", {
        {"name", "qwen3.8-27b"},
        {"default_context_length", 8192}
    }}}}}}};
    nlohmann::json legacy = {{"models", {{"qwen3.8", {{"27b", {
        {"name", "qwen3.8-27b"},
        {"default_context_length", 16384}
    }}}}}}};
    write_json(legacy_path, legacy);

    auto log = oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);

    std::ifstream f(settings_path);
    auto settings = nlohmann::json::parse(f);

    ASSERT_TRUE(settings["settings"]["qwen3.8"]["27b"].contains("ctx_len"));
    ASSERT_EQ(settings["settings"]["qwen3.8"]["27b"]["ctx_len"], 16384);

    std::filesystem::remove_all(dir);
}

TEST(migration_identical_entries_dropped) {
    auto dir = make_temp_dir();
    std::string legacy_path = dir + "/model_list.json";
    std::string settings_path = dir + "/settings.json";

    nlohmann::json entry = {{"name", "test"}, {"default_context_length", 4096}};
    nlohmann::json shipped = {{"models", {{"test", {{"1b", entry}}}}}};
    nlohmann::json legacy = {{"models", {{"test", {{"1b", entry}}}}}};
    write_json(legacy_path, legacy);

    auto log = oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);

    std::ifstream f(settings_path);
    auto settings = nlohmann::json::parse(f);

    ASSERT_FALSE(settings["models"].contains("test"));
    ASSERT_FALSE(settings["settings"].contains("test"));

    std::filesystem::remove_all(dir);
}

TEST(migration_idempotent) {
    auto dir = make_temp_dir();
    std::string legacy_path = dir + "/model_list.json";
    std::string settings_path = dir + "/settings.json";

    nlohmann::json shipped = {{"models", nlohmann::json::object()}};
    nlohmann::json legacy = {{"models", {{"my-model", {{"1b", {{"name", "test"}}}}}}}};
    write_json(legacy_path, legacy);

    oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);
    ASSERT_TRUE(std::filesystem::exists(settings_path));
    ASSERT_FALSE(std::filesystem::exists(legacy_path));

    auto log2 = oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);
    ASSERT_TRUE(log2.size() >= 1);

    std::filesystem::remove_all(dir);
}

// ============================================================================
// Shipped-default change detection tests
//
// The shipped defaults can change between releases. A user who overrode one
// of them would otherwise sit on a stale preference without being told, so
// the settings file records the defaults_version it was written against and
// the loader compares it with the shipped one.
//
// These go through the real loader: OFLM_CONFIG_PATH and OFLM_SETTINGS_PATH
// point it at files on disk, so what is checked is the code that runs, not a
// copy of its logic.
// ============================================================================

namespace {

// Write a shipped registry and a settings file, and point the loader at them.
struct Fixture {
    std::string dir;
    std::string shipped_path;
    std::string settings_path;

    Fixture(nlohmann::json shipped, nlohmann::json settings) {
        dir = make_temp_dir();
        shipped_path = dir + "/model_list.json";
        settings_path = dir + "/settings.json";
        write_json(shipped_path, shipped);
        write_json(settings_path, settings);
        setenv("OFLM_CONFIG_PATH", shipped_path.c_str(), 1);
        setenv("OFLM_SETTINGS_PATH", settings_path.c_str(), 1);
    }
    ~Fixture() {
        unsetenv("OFLM_CONFIG_PATH");
        unsetenv("OFLM_SETTINGS_PATH");
        std::filesystem::remove_all(dir);
    }
};

nlohmann::json shipped_with_defaults(const std::string& version, const std::string& pmode) {
    return {
        {"model_path", "models"},
        {"defaults", {{"pmode", pmode}}},
        {"defaults_version", version},
        {"models", nlohmann::json::object()}
    };
}

}  // namespace

TEST(conflict_detected_when_the_shipped_defaults_changed) {
    Fixture f(shipped_with_defaults("new_hash", "performance"),
              {{"version", 1},
               {"defaults_version", "old_hash"},
               {"defaults", {{"pmode", "balanced"}}},
               {"settings", nlohmann::json::object()},
               {"models", nlohmann::json::object()}});

    auto cfg = oflm::config::load_effective_config();
    ASSERT_TRUE(cfg.defaults_changed);
    ASSERT_EQ(cfg.shipped_defaults_version, std::string("new_hash"));
    ASSERT_EQ(cfg.user_defaults_version, std::string("old_hash"));
    // The override the user made is named, so the warning says what to look at.
    ASSERT_TRUE(cfg.affected_overrides.size() >= 1);
    bool names_pmode = false;
    for (const auto& line : cfg.affected_overrides) {
        if (line.find("pmode") != std::string::npos) names_pmode = true;
    }
    ASSERT_TRUE(names_pmode);
}

TEST(no_conflict_when_the_defaults_version_matches) {
    Fixture f(shipped_with_defaults("same_hash", "performance"),
              {{"version", 1},
               {"defaults_version", "same_hash"},
               {"defaults", {{"pmode", "balanced"}}},
               {"settings", nlohmann::json::object()},
               {"models", nlohmann::json::object()}});

    auto cfg = oflm::config::load_effective_config();
    ASSERT_FALSE(cfg.defaults_changed);
}

TEST(no_conflict_on_a_file_written_before_the_version_existed) {
    // A settings file from before defaults_version must not be reported as
    // conflicting: the first run records the current version instead.
    Fixture f(shipped_with_defaults("new_hash", "performance"),
              {{"version", 1},
               {"defaults", {{"pmode", "balanced"}}},
               {"settings", nlohmann::json::object()},
               {"models", nlohmann::json::object()}});

    auto cfg = oflm::config::load_effective_config();
    ASSERT_FALSE(cfg.defaults_changed);
    // And the version is now recorded, so the next run is quiet.
    std::ifstream f2(cfg.user_settings_path);
    auto settings = nlohmann::json::parse(f2);
    ASSERT_TRUE(settings.contains("defaults_version"));
    ASSERT_EQ(settings["defaults_version"], std::string("new_hash"));
}

TEST(the_users_override_wins_over_a_changed_shipped_default) {
    // The conflict is only reported. The value in effect is still the user's.
    Fixture f(shipped_with_defaults("new_hash", "performance"),
              {{"version", 1},
               {"defaults_version", "old_hash"},
               {"defaults", {{"pmode", "balanced"}}},
               {"settings", nlohmann::json::object()},
               {"models", nlohmann::json::object()}});

    auto cfg = oflm::config::load_effective_config();
    ASSERT_EQ(cfg.merged["defaults"]["pmode"], "balanced");
}

TEST(a_per_model_override_beats_the_app_wide_default) {
    // One model can differ from the app-wide value: the model block wins for
    // that model, the app-wide value applies to the others.
    Fixture f({{"model_path", "models"},
               {"defaults", {{"pmode", "balanced"}}},
               {"defaults_version", "h"},
               {"models", {{"qwen3.6-moe", {{"35b-a3b", {{"name", "moe"}}}}}}}},
              {{"version", 1},
               {"defaults_version", "h"},
               {"defaults", {{"pmode", "balanced"}}},
               {"settings", {{"qwen3.6-moe", {{"35b-a3b", {{"pmode", "turbo"}}}}}}},
               {"models", nlohmann::json::object()}});

    auto cfg = oflm::config::load_effective_config();
    ASSERT_EQ(cfg.merged["defaults"]["pmode"], "balanced");
    ASSERT_EQ(cfg.merged["models"]["qwen3.6-moe"]["35b-a3b"]["pmode"], "turbo");
}

// ============================================================================
// get_merged_model_entry tests
// ============================================================================

TEST(get_merged_model_entry_shipped_only) {
    oflm::config::EffectiveConfig cfg;
    cfg.shipped_registry = {{"models", {{"qwen3.8", {{"27b", {{"name", "test"}, {"default_context_length", 8192}}}}}}}};
    cfg.user_settings = nlohmann::json::object();

    auto entry = oflm::config::get_merged_model_entry(cfg, "qwen3.8:27b");
    ASSERT_EQ(entry["name"], "test");
    ASSERT_EQ(entry["default_context_length"], 8192);
}

TEST(get_merged_model_entry_with_overlay) {
    oflm::config::EffectiveConfig cfg;
    cfg.shipped_registry = {{"models", {{"qwen3.8", {{"27b", {{"name", "test"}, {"default_context_length", 8192}}}}}}}};
    cfg.user_settings = {{"settings", {{"qwen3.8", {{"27b", {{"ctx_len", 16384}}}}}}}};

    auto entry = oflm::config::get_merged_model_entry(cfg, "qwen3.8:27b");
    ASSERT_EQ(entry["name"], "test");
    ASSERT_EQ(entry["default_context_length"], 8192);
    ASSERT_EQ(entry["ctx_len"], 16384);
}

TEST(get_merged_model_entry_user_added_model) {
    oflm::config::EffectiveConfig cfg;
    cfg.shipped_registry = {{"models", nlohmann::json::object()}};
    cfg.user_settings = {{"models", {{"my-model", {{"1b", {{"name", "my-model"}, {"url", "https://example.com"}}}}}}}};

    auto entry = oflm::config::get_merged_model_entry(cfg, "my-model:1b");
    ASSERT_EQ(entry["name"], "my-model");
    ASSERT_EQ(entry["url"], "https://example.com");
}

TEST(get_merged_model_entry_not_found) {
    oflm::config::EffectiveConfig cfg;
    cfg.shipped_registry = {{"models", nlohmann::json::object()}};
    cfg.user_settings = nlohmann::json::object();

    auto entry = oflm::config::get_merged_model_entry(cfg, "nonexistent:1b");
    ASSERT_TRUE(entry.empty() || entry.is_null());
}

TEST(user_model_collision_matches_the_loaded_registry) {
    nlohmann::json shipped = {{"models", {{"qwen3.8", {{"27b", {
        {"name", "shipped"}, {"default_context_length", 32768}, {"url", "old"}
    }}}}}}};
    nlohmann::json user = {{"models", {{"qwen3.8", {{"27b", {
        {"name", "local"}, {"default_context_length", 4096}
    }}}}}}};
    Fixture f(shipped, user);
    auto cfg = oflm::config::load_effective_config();
    auto entry = oflm::config::get_merged_model_entry(cfg, "qwen3.8:27b");
    ASSERT_EQ(entry, cfg.merged["models"]["qwen3.8"]["27b"]);
    ASSERT_EQ(entry["default_context_length"], 4096);
    ASSERT_FALSE(entry.contains("url"));
}

TEST(user_model_metadata_survives_a_settings_overlay) {
    nlohmann::json user;
    user["models"]["local"]["27b"] = {
        {"name", "local"}, {"default_context_length", 4096}
    };
    user["settings"]["local"]["27b"] = {{"prefill_mode", "lean"}};
    Fixture f({{"models", nlohmann::json::object()}}, user);
    auto cfg = oflm::config::load_effective_config();
    auto entry = oflm::config::get_merged_model_entry(cfg, "local:27b");
    ASSERT_EQ(entry, cfg.merged["models"]["local"]["27b"]);
    ASSERT_EQ(entry["name"], "local");
    ASSERT_EQ(entry["default_context_length"], 4096);
}

// ============================================================================
// Main
// ============================================================================

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
