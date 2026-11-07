# Task 22: Engine Integration — Runner + RestHandler

**Phase:** 8 (Integration with Engine Settings)
**Depends on:** Task 21
**Files to modify:** `src/runner/runner.cpp`, `src/server/rest_handler.cpp`
**Estimated complexity:** Small (~40 lines)

## Context

Pass resolved settings to the Runner and RestHandler so they can use them for per-model defaults (prefill_chunk_len, think, etc.).

## What to do

1. In `runner.cpp`: use resolved settings for `prefill_chunk_len`
2. In `rest_handler.cpp`: use resolved settings as per-request defaults

## Code changes

### In `src/runner/runner.cpp`

Find where `prefill_chunk_len` is set (around line 88-98):

```cpp
// OLD:
// prefill_chunk_len = (parsed_args.prefill_chunk_len > 0) ? parsed_args.prefill_chunk_len : model_info["max_prefill_len"].get<int>();

// NEW:
int prefill_chunk_len = -1;
if (oflm::config::g_resolved_settings && oflm::config::g_resolved_settings->contains("prefill_chunk_len")) {
    prefill_chunk_len = (*oflm::config::g_resolved_settings)["prefill_chunk_len"].get<int>();
} else if (parsed_args.prefill_chunk_len > 0) {
    prefill_chunk_len = parsed_args.prefill_chunk_len;
} else {
    prefill_chunk_len = model_info.value("max_prefill_len", 4096);
}
```

Add include at top:
```cpp
#include "config/settings_resolver.hpp"
```

### In `src/server/rest_handler.cpp`

Find `configure_chat_engine_parameters` (around line 664-708). Update the think default:

```cpp
// OLD:
// (think is handled per-request via request["think"])

// NEW:
// Get default think from resolved settings
bool default_think = true;
if (oflm::config::g_resolved_settings && oflm::config::g_resolved_settings->contains("think")) {
    default_think = (*oflm::config::g_resolved_settings)["think"].get<bool>();
}

// In the think handling (around line 695-698):
if (request.contains("think")) {
    configure_parameter("enable_think", request["think"].get<bool>());
} else {
    // Use resolved setting as default
    configure_parameter("enable_think", default_think);
}
```

Also update the prefill_chunk_len handling (around line 569-571):

```cpp
// OLD:
// prefill_chunk_len = model_info["max_prefill_len"];

// NEW:
if (oflm::config::g_resolved_settings && oflm::config::g_resolved_settings->contains("prefill_chunk_len")) {
    prefill_chunk_len = (*oflm::config::g_resolved_settings)["prefill_chunk_len"].get<int>();
} else if (parsed_args.prefill_chunk_len > 0) {
    prefill_chunk_len = parsed_args.prefill_chunk_len;
} else {
    prefill_chunk_len = model_info.value("max_prefill_len", 4096);
}
```

Add include at top:
```cpp
#include "config/settings_resolver.hpp"
```

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -10

# Verify runner uses resolved settings
export OFLM_SETTINGS_PATH=/tmp/test_settings.json
echo '{"version":1,"settings":{"qwen3.8":{"27b":{"prefill_chunk_len":2048}}}}' > /tmp/test_settings.json

# Run and check logs for the applied setting
./build/src/oflm run qwen3.8:27b 2>&1 | head -10 &
sleep 3
kill %1 2>/dev/null
```

## Success criteria

- [ ] `runner.cpp` uses resolved settings for `prefill_chunk_len`
- [ ] `rest_handler.cpp` uses resolved settings as default for `think`
- [ ] `rest_handler.cpp` uses resolved settings for `prefill_chunk_len`
- [ ] Per-request API fields still override resolved settings
- [ ] Includes added (`config/settings_resolver.hpp`)
- [ ] Project compiles without errors
- [ ] `oflm run` and `oflm serve` still work
