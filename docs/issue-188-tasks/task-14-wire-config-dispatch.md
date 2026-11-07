# Task 14: Wire Config Dispatch in main.cpp

**Phase:** 4 (CLI Commands)
**Depends on:** Task 12
**Files to modify:** `src/src/main.cpp`
**Estimated complexity:** Small (~15 lines)

## Context

Wire the `config` command dispatch in `main.cpp`. The `config` command is handled inside the Boost parser (C3 fix), so we just need to add a dispatch case.

## What to do

Add a dispatch case for `config` in `main.cpp`, after the `list` dispatch (around line 765).

## Code to add

In `src/src/main.cpp`, after the `list` command handling (around line 814), add:

```cpp
if (parsed_args.command == "config") {
    return oflm::config::config_command_main(parsed_args);
}
```

Also add the include at the top of the file:

```cpp
#include "config/config_cmd.hpp"
```

## Important: config command should NOT load the model registry

The `config` command should work even if `model_list.json` is not found (e.g., user just wants to edit settings). Currently, `main.cpp` loads the registry at line 565-577 before dispatching. For the `config` command, we should skip registry loading.

Move the `config` dispatch **before** the registry loading (around line 565), or add an early check:

```cpp
// Early dispatch for config — doesn't need the model registry
if (parsed_args.command == "config") {
    return oflm::config::config_command_main(parsed_args);
}
```

Place this right after `arg_utils::parse_options()` (line 556-558), before `find_model_list()` (line 565).

Wait — but the `config` command needs the shipped registry for `show` and `gen-schema`. So it does need `find_shipped_registry()` but not the full `model_list` construction.

The `config_command_main()` function calls `load_effective_config()` which calls `find_shipped_registry()` internally. So we can dispatch early — `config_command_main()` handles its own loading.

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -5

# Test config commands work
./build/src/oflm config show
./build/src/oflm config gen-schema | head -5

# Test config works even without model_list.json nearby
cd /tmp
/work/build/src/oflm config show 2>&1 | head -5
```

## Success criteria

- [ ] `#include "config/config_cmd.hpp"` added to main.cpp
- [ ] `config` dispatch added before registry loading (around line 560)
- [ ] `oflm config show` works
- [ ] `oflm config gen-schema` works
- [ ] `oflm config` commands work even when CWD has no model_list.json
- [ ] Other commands (run, serve, list) still work
- [ ] Project compiles
