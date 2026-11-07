# Issue #188: Task Decomposition for Delegation

**Parent plan:** `docs/issue-188-implementation-plan.md` (v3)
**Branch:** `issue-188-per-model-user-settings`

## Goal

25 small, testable tasks that can be delegated to smaller models. Each task:
- Touches 1-2 files
- Has exact code or skeleton to fill in
- Has a clear test command
- Has checkbox success criteria

## Dependency Graph

```
Phase 1 (Registry)          Phase 7 (oflm-add)         Phase 10 (Docs)
  Task 01 ─┬─► Task 02 ─┬─► Task 03    Task 19 ─► Task 20    Task 24
           │            │                (independent)        Task 25
           │            │                                     (independent)
           └────────────┴──┐
                           ▼
Phase 2 (Config Store)    Phase 6 (Schema Gen)
  Task 04 ──► Task 05 ──► Task 06 ──► Task 07 ──► Task 08    Task 18
  (utils.cpp)  (header)   (merge/save)  (migrate/load)  (tests)   (needs Task 12)
                           │
                           ▼
Phase 3 (Resolver)
  Task 09 ──► Task 10
                           │
                           ▼
Phase 4 (CLI)
  Task 11 ──► Task 12 ──► Task 13 ──► Task 14
  (options)   (commands)  (review)    (dispatch)
                           │
                           ▼
Phase 5 (UI)
  Task 15 ──► Task 16 ──► Task 17
  (tier detect) (picker)   (TUI + guided)
                           │
                           ▼
Phase 8 (Integration)
  Task 21 ──► Task 22
                           │
                           ▼
Phase 9 (Testing)
  Task 23
```

## Execution Order (respects dependencies)

| Wave | Tasks | Notes |
|------|-------|-------|
| 1 | 01, 04, 19, 24, 25 | No dependencies. Can run in parallel. |
| 2 | 02, 05, 20 | Depends on Wave 1. |
| 3 | 03, 06, 09, 11 | Depends on Wave 2. |
| 4 | 07, 10, 12 | Depends on Wave 3. |
| 5 | 08, 13, 18 | Depends on Wave 4. |
| 6 | 14, 15, 21 | Depends on Wave 5. |
| 7 | 16, 22 | Depends on Wave 6. |
| 8 | 17, 23 | Depends on Wave 7. |

## Task Index

| # | Title | Phase | Files | Depends on |
|---|-------|-------|-------|------------|
| 01 | Settings registry header | 1 | `src/config/settings_registry.hpp` | — |
| 02 | Settings registry implementation | 1 | `src/config/settings_registry.cpp` | 01 |
| 03 | Settings registry tests | 1 | `src/test/settings_registry_test.cpp` | 02 |
| 04 | Path discovery functions | 2 | `src/common/utils.cpp`, `src/include/utils/utils.hpp` | — |
| 05 | Config store header | 2 | `src/config/config_store.hpp` | 01 |
| 06 | Config store: merge, save, lock | 2 | `src/config/config_store.cpp` | 05 |
| 07 | Config store: migration, load, reconcile | 2 | `src/config/config_store.cpp` | 06 |
| 08 | Config store tests + model_list constructor | 2 | `src/test/config_store_test.cpp`, `src/include/model_list.hpp` | 07 |
| 09 | Settings resolver implementation | 3 | `src/config/settings_resolver.hpp`, `.cpp` | 02, 05 |
| 10 | Settings resolver tests + get_setting_value | 3 | `src/test/settings_resolver_test.cpp` | 09 |
| 11 | CLI options in vm_args.hpp | 4 | `src/include/utils/vm_args.hpp`, `src/include/program_args.hpp` | — |
| 12 | Config CLI commands (show/set/unset/edit/gen-schema/migrate) | 4 | `src/config/config_cmd.hpp`, `.cpp` | 09, 11 |
| 13 | Config review command (non-interactive) | 4 | `src/config/config_cmd.cpp` | 12 |
| 14 | Wire config dispatch in main.cpp | 4 | `src/src/main.cpp` | 12 |
| 15 | UI tier detection + numbered menu | 5 | `src/config/config_ui.hpp`, `.cpp` | 11 |
| 16 | Arrow-key picker (POSIX + Windows) | 5 | `src/config/config_ui.cpp` | 15 |
| 17 | Guided flow + TUI + config review interactive | 5 | `src/config/config_ui.cpp`, `src/config/config_cmd.cpp` | 16, 13 |
| 18 | Schema injection script + release workflow | 6 | `utilities/release/inject-schema.py`, `.github/workflows/release.yml` | 12 |
| 19 | oflm-add: settings.json path + register() rewrite | 7 | `utilities/oflm-add/oflm_add/__init__.py` | — |
| 20 | oflm-add: --name/--force flags + tests | 7 | `utilities/oflm-add/oflm_add/__init__.py`, `tests/test_settings_json.py` | 19 |
| 21 | Engine integration: main.cpp + automodel.cpp | 8 | `src/src/main.cpp`, `src/common/AutoModel/automodel.cpp` | 09, 14 |
| 22 | Engine integration: runner + rest_handler | 8 | `src/runner/runner.cpp`, `src/server/rest_handler.cpp` | 21 |
| 23 | CMake test registration + smoke tests | 9 | `src/CMakeLists.txt`, `src/test/CMakeLists.txt` | 03, 08, 10 |
| 24 | User config documentation | 10 | `docs/config.md` | — |
| 25 | Update create_new_model.md | 10 | `src/create_new_model.md` | 01 |
| 26 | Settings registry CI lint | Enforcement | `utilities/check-settings-registry.py`, `src/CMakeLists.txt`, CI | 01-02, 12 |
| 27 | Auto-generate settings registry | Enforcement | `utilities/scan-settings.py`, `utilities/generate-registry.py`, `src/config/settings.json`, `src/config/settings_registry_generated.hpp` | 01-02, 11-12 |

## How to Work on a Task

1. Read the task file
2. Read the parent plan section referenced in the task
3. Create/modify the files listed
4. Run the test command
5. Check off the success criteria
6. Commit with message: `task-NN: <title> (issue #188)`

## Build & Test Commands

```bash
# Build (from repo root)
cmake --preset linux-default
cmake --build --preset linux-default

# Run all tests
ctest --preset linux-default

# Run specific test
ctest --preset linux-default -R settings_registry

# Build just the oflm binary
cmake --build --preset linux-default --target oflm
```
