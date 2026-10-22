# Task 23: CMake Test Registration + Smoke Tests

**Phase:** 9 (Testing)
**Depends on:** Task 03, Task 08, Task 10
**Files to modify:** `src/CMakeLists.txt`, `src/test/CMakeLists.txt`
**Estimated complexity:** Small (~40 lines)

## Context

Register all new test targets in CMake and add smoke tests for the config commands. Ensure `ctest` runs everything.

## What to do

1. Verify all test targets are registered in `src/test/CMakeLists.txt` (Tasks 03, 08, 10 should have added them)
2. Add smoke tests to `src/CMakeLists.txt` (near the existing `oflm_smoke` test at line 1373)

## Code to add

In `src/CMakeLists.txt`, after the existing `oflm_smoke` test (around line 1375), add:

```cmake
# Config command smoke tests
add_test(NAME oflm_config_show COMMAND oflm config show)
set_tests_properties(oflm_config_show PROPERTIES
    ENVIRONMENT "OFLM_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/model_list.json")

add_test(NAME oflm_config_gen_schema COMMAND oflm config gen-schema)
set_tests_properties(oflm_config_gen_schema PROPERTIES
    ENVIRONMENT "OFLM_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/model_list.json")

# Test config set/unset with isolated settings file
add_test(NAME oflm_config_set_unset
    COMMAND ${CMAKE_COMMAND} -E env
        "OFLM_SETTINGS_PATH=${CMAKE_BINARY_DIR}/test_settings.json"
        "OFLM_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/model_list.json"
        sh -c "oflm config set --config-key pmode --config-value balanced && \
               oflm config show --json | grep -q '\"pmode\":\"balanced\"' && \
               oflm config unset --config-key pmode && \
               oflm config show --json | grep -q '\"pmode\":\"performance\"'")

# TUI smoke test (Tier 1 fallback via piped input)
add_test(NAME oflm_config_browse_smoke
    COMMAND sh -c "echo q | oflm config browse")
set_tests_properties(oflm_config_browse_smoke PROPERTIES
    ENVIRONMENT "OFLM_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/model_list.json")

# Guided flow smoke test (Tier 1 fallback)
add_test(NAME oflm_config_guided_smoke
    COMMAND sh -c "printf '0\\n' | oflm config")
set_tests_properties(oflm_config_guided_smoke PROPERTIES
    ENVIRONMENT "OFLM_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/model_list.json")
```

## Verify test registration

Check that `src/test/CMakeLists.txt` has these entries (added by Tasks 03, 08, 10):

```cmake
add_executable(settings_registry_test ...)
add_test(NAME settings_registry_test COMMAND settings_registry_test)

add_executable(config_store_test ...)
add_test(NAME config_store_test COMMAND config_store_test)

add_executable(settings_resolver_test ...)
add_test(NAME settings_resolver_test COMMAND settings_resolver_test)
```

If any are missing, add them.

## Test

```bash
# Full build and test
cmake --preset linux-default 2>&1 | tail -3
cmake --build --preset linux-default 2>&1 | tail -10

# Run all tests
ctest --preset linux-default --output-on-failure 2>&1 | tail -30

# Run just the new tests
ctest --preset linux-default -R "settings_registry|config_store|settings_resolver|oflm_config" --output-on-failure
```

## Success criteria

- [ ] All unit test targets registered in CMake
- [ ] Smoke tests added (config show, gen-schema, set/unset, browse, guided)
- [ ] `ctest --preset linux-default` runs all tests
- [ ] All new tests pass
- [ ] No regressions in existing tests
