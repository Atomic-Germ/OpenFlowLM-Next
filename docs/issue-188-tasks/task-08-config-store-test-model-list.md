# Task 08: Config Store Tests + model_list Constructor

**Phase:** 2 (Config Store)
**Depends on:** Task 07
**Files to create:** `src/test/config_store_test.cpp`
**Files to modify:** `src/include/model_list.hpp` (add JSON constructor), `src/test/CMakeLists.txt`
**Estimated complexity:** Medium (~200 lines)

## Context

Unit tests for the config store (deep merge, migration, file locking, corrupted JSON handling) plus adding a constructor to `model_list` that accepts pre-loaded JSON (needed because we merge before constructing).

## What to do

1. Add a constructor to `model_list` that accepts `nlohmann::json` directly
2. Create `src/test/config_store_test.cpp` with comprehensive tests
3. Register the test in CMake

## Part 1: model_list constructor

In `src/include/model_list.hpp`, add a new constructor after the existing one (around line 53):

```cpp
/// \brief constructor from pre-loaded JSON (for merged config)
/// \param config_json the pre-loaded (and possibly merged) registry JSON
/// \param exe_dir the executable directory for resolving relative paths
model_list(const nlohmann::json& config_json, std::string& exe_dir) {
    this->config = config_json;
    // Resolve model_root_path relative to executable directory
    std::string relative_model_path = this->config.value("model_path", "models");
    std::filesystem::path root_path = std::filesystem::path(exe_dir) / relative_model_path;
    this->model_root_path = root_path.string();
    this->resolve_hf_owner();

    // Populate all_tags set
    if (this->config.contains("models")) {
        for (const auto& [model_type, sizes] : this->config["models"].items()) {
            all_tags.insert(model_type);
            for (const auto& [size, model_info] : sizes.items()) {
                all_tags.insert(model_type + ":" + size);
            }
        }
    }
}
```

## Part 2: Test file

Create `src/test/config_store_test.cpp`:

```cpp
/// \file config_store_test.cpp
/// \brief Unit tests for the config store (merge, migration, lock, corrupted JSON)
#include "config/config_store.hpp"
#include "config/settings_registry.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#define TEST(name) static void name(); \
    static struct name##_registrar { name##_registrar() { tests.push_back({#name, name}); } } name##_instance; \
    static void name()

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "FAIL: " << #expr << " at line " << __LINE__ << std::endl; failures++; return; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "FAIL: " << #a << " != " << #b << " (" << (a) << " vs " << (b) << ") at line " << __LINE__ << std::endl; failures++; return; } } while(0)

static int failures = 0;
static std::vector<std::pair<std::string, void(*)()>> tests;

// Helper: create a temp directory for tests
static std::string make_temp_dir() {
    auto dir = std::filesystem::temp_directory_path() / ("oflm_test_" + std::to_string(rand()));
    std::filesystem::create_directories(dir);
    return dir.string();
}

// Helper: write JSON to file
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
    ASSERT_EQ(result["a"]["c"], 2); // preserved from base
}

TEST(deep_merge_arrays_replace) {
    nlohmann::json base = {{"files", {"a.txt", "b.txt"}}};
    nlohmann::json overlay = {{"files", {"c.txt"}}};
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
    // After lock1 is destroyed, lock2 should acquire immediately
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

    // Read back and verify
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

    // No temp files should remain
    int file_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        file_count++;
    }
    ASSERT_EQ(file_count, 1); // Only settings.json, no .tmp or .lock

    std::filesystem::remove_all(dir);
}

// ============================================================================
// migrate_legacy_user_registry tests
// ============================================================================

TEST(migration_user_added_model) {
    auto dir = make_temp_dir();
    std::string legacy_path = dir + "/model_list.json";
    std::string settings_path = dir + "/settings.json";

    nlohmann::json shipped = {
        {"models", {{"qwen3.8", {{"27b", {{"name", "qwen3.8-27b"}}}}}}}
    };
    nlohmann::json legacy = {
        {"models", {
            {"qwen3.8", {{"27b", {{"name", "qwen3.8-27b"}}}}},  // identical to shipped
            {"my-model", {{"1b", {{"name", "my-model-1b"}, {"url", "https://example.com"}}}}}  // user-added
        }}
    };
    write_json(legacy_path, legacy);

    auto log = oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);

    // Verify settings.json was created
    ASSERT_TRUE(std::filesystem::exists(settings_path));
    std::ifstream f(settings_path);
    auto settings = nlohmann::json::parse(f);

    // User-added model should be in models section
    ASSERT_TRUE(settings["models"]["my-model"]["1b"].contains("url"));

    // Legacy file should be renamed to .bak
    ASSERT_FALSE(std::filesystem::exists(legacy_path));
    ASSERT_TRUE(std::filesystem::exists(legacy_path + ".bak"));

    std::filesystem::remove_all(dir);
}

TEST(migration_modified_entry_preserved_as_override) {
    auto dir = make_temp_dir();
    std::string legacy_path = dir + "/model_list.json";
    std::string settings_path = dir + "/settings.json";

    nlohmann::json shipped = {
        {"models", {{"qwen3.8", {{"27b", {
            {"name", "qwen3.8-27b"},
            {"default_context_length", 8192}
        }}}}}}
    };
    nlohmann::json legacy = {
        {"models", {{"qwen3.8", {{"27b", {
            {"name", "qwen3.8-27b"},
            {"default_context_length", 16384}  // Modified!
        }}}}}}
    };
    write_json(legacy_path, legacy);

    auto log = oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);

    std::ifstream f(settings_path);
    auto settings = nlohmann::json::parse(f);

    // Modified entry should be preserved as override (ctx_len maps to default_context_length)
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
    nlohmann::json legacy = {{"models", {{"test", {{"1b", entry}}}}}};  // identical
    write_json(legacy_path, legacy);

    auto log = oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);

    std::ifstream f(settings_path);
    auto settings = nlohmann::json::parse(f);

    // Identical entry should NOT appear in settings or models
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

    // First migration
    oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);
    ASSERT_TRUE(std::filesystem::exists(settings_path));
    ASSERT_FALSE(std::filesystem::exists(legacy_path)); // renamed to .bak

    // Second call should not re-migrate (legacy file is gone)
    auto log2 = oflm::config::migrate_legacy_user_registry(legacy_path, shipped, settings_path);
    ASSERT_TRUE(log2.size() >= 1);
    ASSERT_TRUE(log2[0].find("not found") != std::string::npos || log2[0].find("Failed") != std::string::npos);

    std::filesystem::remove_all(dir);
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
    ASSERT_EQ(entry["ctx_len"], 16384); // from overlay
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
```

## Part 3: CMake registration

Add to `src/test/CMakeLists.txt`:

```cmake
# Config store tests (host-only, no NPU needed)
add_executable(config_store_test config_store_test.cpp
    ${CMAKE_SOURCE_DIR}/config/config_store.cpp
    ${CMAKE_SOURCE_DIR}/config/settings_registry.cpp
    ${CMAKE_SOURCE_DIR}/common/utils.cpp)
target_include_directories(config_store_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR})
target_compile_definitions(config_store_test PRIVATE CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
add_test(NAME config_store_test COMMAND config_store_test)
```

## Test

```bash
cmake --preset linux-default 2>&1 | tail -3
cmake --build --preset linux-default --target config_store_test 2>&1 | tail -10
ctest --preset linux-default -R config_store --output-on-failure
```

## Success criteria

- [ ] `model_list` has a new constructor accepting `nlohmann::json`
- [ ] File `src/test/config_store_test.cpp` exists with 18+ test cases
- [ ] Test registered in `src/test/CMakeLists.txt`
- [ ] All tests pass: `ctest -R config_store --output-on-failure`
- [ ] Tests cover: deep_merge (6 cases), split_tag (2), SettingsLock (2), save (2), migration (4), get_merged_model_entry (4)
