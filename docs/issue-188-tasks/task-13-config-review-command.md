# Task 13: Config Review Command (Non-Interactive)

**Phase:** 4 (CLI Commands)
**Depends on:** Task 12
**Files to modify:** `src/config/config_cmd.cpp` (add review action)
**Estimated complexity:** Small (~60 lines)

## Context

Add the `config review` action to the config command. This task implements only the **non-interactive** modes (`--accept-shipped`, `--keep-all`). The interactive mode comes in Task 17.

## What to do

Add a `review` action handler to `config_command_main()` in `src/config/config_cmd.cpp`.

## Code to add

In the action dispatch switch in `config_command_main()`, add:

```cpp
else if (action == "review") {
    std::string mode = "interactive";
    if (args.config_accept_shipped) mode = "accept-shipped";
    else if (args.config_keep_all) mode = "keep-all";

    if (mode == "interactive") {
        // Interactive mode not yet available (Task 17)
        std::cerr << "Interactive review not yet available." << std::endl;
        std::cerr << "Use 'oflm config review --accept-shipped' to remove redundant overrides," << std::endl;
        std::cerr << "or 'oflm config review --keep-all' to keep all and update version." << std::endl;
        return 1;
    }

    std::string settings_path = utils::find_user_settings();
    int removed = reconcile_settings(settings_path, mode);
    return removed >= 0 ? 0 : 1;
}
```

Also add `"review"` to the list of valid actions (wherever the action validation happens).

## Test

```bash
cmake --build --preset linux-default --target oflm 2>&1 | tail -5

export OFLM_SETTINGS_PATH=/tmp/test_settings.json

# Set up a test scenario
echo '{"version":1,"defaults_version":"old_hash","defaults":{"pmode":"balanced"}}' > /tmp/test_settings.json

# Test keep-all
./build/src/oflm config review --keep-all

# Test accept-shipped
./build/src/oflm config review --accept-shipped

# Test interactive (should print "not yet available")
./build/src/oflm config review
```

## Success criteria

- [ ] `config review` action added to dispatch
- [ ] `--accept-shipped` mode works (removes redundant overrides)
- [ ] `--keep-all` mode works (updates version, no changes)
- [ ] Interactive mode prints "not yet available" message
- [ ] Project compiles
