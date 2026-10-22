# Task 26: Settings Registry CI Lint + CMake Target

**Phase:** Enforcement (post-implementation)
**Depends on:** Tasks 01-02 (registry), 12 (gen-schema command), 18 (inject-schema.py)
**Files to create:** `utilities/check-settings-registry.py`
**Files to modify:** `src/CMakeLists.txt`, `.github/workflows/cmake-build-test.yml`
**Estimated complexity:** Medium (~200 lines Python + ~20 lines CMake/YAML)

## Context

This task adds **enforcement** — CI checks that catch when someone adds a new `OFLM_*` env var or `--flag` but forgets to register it in the settings registry. Without this, the registry silently drifts out of sync with the code.

The script checks four things:

**a) Unregistered env vars** — Scan `src/` for `getenv_oflm("OFLM_...")` and `std::getenv("OFLM_...")` calls. Each must be either:
- In the settings registry (as `env_var` field of a SettingDef), OR
- In a known-debug allowlist (OFLM_OPEN_*, OW_*, etc.)

**b) Unregistered CLI flags** — Scan `src/include/utils/vm_args.hpp` for `("--flag-name", ...)` patterns. Each must be either:
- Mapped to a registry entry (as `cli_flag` field), OR
- In a known-non-setting allowlist (--json, --quiet, --force, etc.)

**c) Schema out of date** — Run `oflm config gen-schema` and compare the `settings_schema` block against what's in `src/model_list.json`. Fail if different (means someone added a registry entry but didn't run inject-schema.py).

**d) Registry completeness** — Run `oflm config gen-schema` and verify every setting has: key, type, description, scope, delivery, category.

## What to do

### 1. Create `utilities/check-settings-registry.py`

```python
#!/usr/bin/env python3
"""Check settings registry consistency.

Verifies that:
a) Every user-facing OFLM_* env var is in the settings registry (or debug allowlist)
b) Every --flag in vm_args.hpp maps to a registry entry (or non-setting allowlist)
c) The settings_schema in model_list.json matches `oflm config gen-schema` output
d) Every registry entry has required fields

Exit 0 = all checks pass. Exit 1 = failures (prints details).

Usage:
    check-settings-registry.py [--oflm PATH] [--source-dir PATH]

    --oflm PATH       Path to oflm binary (default: build/src/oflm)
    --source-dir PATH Path to src/ directory (default: src)
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Known debug env vars (not user-facing, don't need registry entries)
DEBUG_ENV_ALLOWLIST = {
    # Open engine debug knobs
    "OFLM_OPEN_OMP_THREADS", "OFLM_OPEN_OMP_PASSIVE", "OFLM_OPEN_GEMM_BLOCK",
    "OFLM_OPEN_GEMM_BLOCK_MIN", "OFLM_OPEN_MOE_BATCH", "OFLM_OPEN_ATTN_BLOCK",
    "OFLM_OPEN_LAYER_MAJOR", "OFLM_OPEN_STAGE_MAJOR", "OFLM_OPEN_HOST_ATTN",
    "OFLM_OPEN_HOST_ATTN_DECODE", "OFLM_DECODE_AUTO_THRESHOLD",
    "OFLM_OPEN_MOE_REDISPATCH", "OFLM_ROUTE_CHECK", "OFLM_DET_ROUTE",
    "OFLM_OPEN_ROUTE_SENTINEL", "OFLM_OPEN_READ_FENCE", "OFLM_OPEN_STEP_TRACE",
    "OFLM_OPEN_SPIN_US", "OFLM_OPEN_SUBMIT_AHEAD", "OFLM_OPEN_DISPATCH_LOG",
    "OFLM_OPEN_DXB_LOG", "OFLM_OPEN_MOE_BATCH_LOG", "OFLM_OPEN_TIMEOUT_MS",
    "OFLM_OPEN_TIMEOUT_RETRY_MS", "OFLM_OPEN_QUIET",
    # Whisper debug
    "OFLM_WHISPER_DUMP_MEL", "OFLM_WHISPER_PROTOCOL",
    # Embedding debug
    "OFLM_NPU_DISABLE", "OFLM_NPU_DEVICE_ID",
    # Path/plumbing (not settings)
    "OFLM_CONFIG_PATH", "OFLM_SETTINGS_PATH", "OFLM_MODELINFO_PATH",
    "OFLM_XCLBIN_PATH", "OFLM_MODEL_PATH", "OFLM_SERVE_PORT",
    "OFLM_HF_OWNER", "OFLM_PLATFORM", "OFLM_DISABLE_UPDATE_CHECK",
    "OFLM_ADD_SCRIPT", "OFLM_PACK_SCRIPT", "OFLM_PYTHON", "OFLM_EXECUTABLE",
    "OFLM_MODEL_DIR", "OFLM_BACKEND", "OFLM_OPEN_KERNELS_DIR",
    "OFLM_WHISPER_KERNELS_DIR", "OFLM_WHISPER_ENGINE",
    # Per-family engine selectors (13 vars)
    "OFLM_QWEN3_ENGINE", "OFLM_QWEN3VL_ENGINE", "OFLM_QWEN35_ENGINE",
    "OFLM_QWEN36_ENGINE", "OFLM_QWEN2_ENGINE", "OFLM_QWEN2VL_ENGINE",
    "OFLM_GEMMA_ENGINE", "OFLM_GRANITE_ENGINE", "OFLM_LLAMA_ENGINE",
    "OFLM_PHI4_ENGINE", "OFLM_LFM2_ENGINE", "OFLM_GPTOSS_ENGINE",
}

# Known non-setting CLI flags (not user-facing settings)
NON_SETTING_FLAGS = {
    "--json", "-j", "--quiet", "-q", "--force", "--host", "--port", "-p",
    "--filter", "--socket", "--q-len", "--cors", "--preemption",
    "--asr", "--embed", "--asrmodel", "--embeddingmodel",
    "--imagegen", "--imagemodel", "--img-pre-resize",
    "--bench-iterations", "--max-batch", "--prompt-name",
    "--out", "-o", "--size", "--seed", "--image",
    "--prompt", "-i", "--modelscope", "--config-key", "--config-value",
    "--config-model", "--config-accept-shipped", "--config-keep-all",
    "--help", "-h", "--version", "-v",
}

# Required fields for every registry setting
REQUIRED_FIELDS = ["key", "type", "description", "scope", "delivery", "category"]

failures = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL: {msg}")


def ok(msg):
    print(f"  OK: {msg}")


def get_registry_settings(oflm_path):
    """Run `oflm config gen-schema` and return the settings list."""
    result = subprocess.run(
        [oflm_path, "config", "gen-schema"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: `oflm config gen-schema` failed: {result.stderr}")
        sys.exit(1)
    data = json.loads(result.stdout)
    return data.get("settings_schema", {}).get("settings", []), data


def check_env_vars(source_dir, registry_env_vars):
    """Check that all OFLM_* env vars in source are registered or in allowlist."""
    print("\n=== Check a) Environment variables ===")
    src_path = Path(source_dir)
    env_vars_found = set()

    # Scan for getenv_oflm("OFLM_...") and std::getenv("OFLM_...")
    patterns = [
        re.compile(r'getenv_oflm\(\s*"(OFLM_[A-Z0-9_]+)"\s*\)'),
        re.compile(r'std::getenv\(\s*"(OFLM_[A-Z0-9_]+)"\s*\)'),
    ]

    for cpp_file in src_path.rglob("*.cpp"):
        if "test" in str(cpp_file) or "build" in str(cpp_file):
            continue
        content = cpp_file.read_text(errors="ignore")
        for pat in patterns:
            for match in pat.finditer(content):
                env_vars_found.add(match.group(1))
    for hpp_file in src_path.rglob("*.hpp"):
        if "test" in str(hpp_file) or "build" in str(hpp_file):
            continue
        content = hpp_file.read_text(errors="ignore")
        for pat in patterns:
            for match in pat.finditer(content):
                env_vars_found.add(match.group(1))

    unregistered = env_vars_found - registry_env_vars - DEBUG_ENV_ALLOWLIST
    if unregistered:
        for var in sorted(unregistered):
            fail(f"Env var {var} used in code but not in settings registry or debug allowlist")
    else:
        ok(f"All {len(env_vars_found)} env vars are registered or in allowlist")

    # Also check for env vars in registry that don't exist in code (dead entries)
    unused = registry_env_vars - env_vars_found
    if unused:
        for var in sorted(unused):
            fail(f"Env var {var} in registry but not used in any source file (dead entry?)")


def check_cli_flags(source_dir, registry_flags):
    """Check that all --flags in vm_args.hpp are registered or in allowlist."""
    print("\n=== Check b) CLI flags ===")
    vm_args = Path(source_dir) / "include" / "utils" / "vm_args.hpp"
    if not vm_args.exists():
        fail(f"vm_args.hpp not found at {vm_args}")
        return

    content = vm_args.read_text()
    # Find all ("--flag-name", ...) patterns
    flags_found = set(re.findall(r'\("(--[a-z0-9-]+)"', content))
    # Also find short flags like ("-j", ...)
    flags_found |= set(re.findall(r'\("(-[a-zA-Z])"', content))

    unregistered = flags_found - registry_flags - NON_SETTING_FLAGS
    if unregistered:
        for flag in sorted(unregistered):
            fail(f"CLI flag {flag} defined in vm_args.hpp but not in settings registry or non-setting allowlist")
    else:
        ok(f"All {len(flags_found)} CLI flags are registered or in allowlist")


def check_schema_sync(oflm_path, source_dir, gen_schema_output):
    """Check that model_list.json's settings_schema matches gen-schema output."""
    print("\n=== Check c) Schema sync ===")
    model_list = Path(source_dir) / "model_list.json"
    if not model_list.exists():
        fail(f"model_list.json not found at {model_list}")
        return

    with open(model_list) as f:
        shipped = json.load(f)

    shipped_schema = shipped.get("settings_schema", {})
    gen_schema = gen_schema_output.get("settings_schema", {})

    if not shipped_schema:
        fail("model_list.json has no settings_schema block — run inject-schema.py")
        return

    if shipped_schema.get("version") != gen_schema.get("version"):
        fail(f"Schema version mismatch: shipped={shipped_schema.get('version')}, generated={gen_schema.get('version')}")
        return

    shipped_keys = {s["key"] for s in shipped_schema.get("settings", [])}
    gen_keys = {s["key"] for s in gen_schema.get("settings", [])}

    if shipped_keys != gen_keys:
        missing = gen_keys - shipped_keys
        extra = shipped_keys - gen_keys
        if missing:
            fail(f"Settings in registry but not in model_list.json schema: {missing} — run inject-schema.py")
        if extra:
            fail(f"Settings in model_list.json schema but not in registry: {extra}")
        return

    ok(f"Schema sync OK ({len(gen_keys)} settings)")


def check_registry_completeness(gen_schema_output):
    """Check that every registry setting has required fields."""
    print("\n=== Check d) Registry completeness ===")
    settings = gen_schema_output.get("settings_schema", {}).get("settings", [])
    for s in settings:
        for field in REQUIRED_FIELDS:
            if field not in s or not s[field]:
                fail(f"Setting '{s.get('key', '?')}' missing required field: {field}")
    if not failures:
        ok(f"All {len(settings)} settings have required fields")


def main():
    parser = argparse.ArgumentParser(description="Check settings registry consistency")
    parser.add_argument("--oflm", default="build/src/oflm", help="Path to oflm binary")
    parser.add_argument("--source-dir", default="src", help="Path to src/ directory")
    args = parser.parse_args()

    oflm_path = args.oflm
    source_dir = args.source_dir

    if not Path(oflm_path).exists():
        print(f"ERROR: oflm binary not found at {oflm_path}")
        print("Build first: cmake --build --preset linux-default --target oflm")
        sys.exit(1)

    print(f"Checking settings registry consistency...")
    print(f"  oflm binary: {oflm_path}")
    print(f"  source dir: {source_dir}")

    # Get registry settings
    settings, gen_output = get_registry_settings(oflm_path)
    registry_env_vars = {s["env_var"] for s in settings if s.get("env_var")}
    registry_flags = {s["cli_flag"] for s in settings if s.get("cli_flag")}

    print(f"\nRegistry has {len(settings)} settings")
    print(f"  Env vars: {sorted(registry_env_vars)}")
    print(f"  CLI flags: {sorted(registry_flags)}")

    # Run checks
    check_env_vars(source_dir, registry_env_vars)
    check_cli_flags(source_dir, registry_flags)
    check_schema_sync(oflm_path, source_dir, gen_output)
    check_registry_completeness(gen_output)

    # Summary
    print(f"\n{'='*50}")
    if failures:
        print(f"FAILED: {len(failures)} issue(s) found")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("PASSED: All settings registry checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

### 2. Add CMake target in `src/CMakeLists.txt`

Add near the end of the file (after the test registrations, around line 1400+):

```cmake
# Settings registry consistency check
add_custom_target(check-settings
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../utilities/check-settings-registry.py
            --oflm $<TARGET_FILE:oflm>
            --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS oflm
    COMMENT "Checking settings registry consistency"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
)
```

### 3. Add CI step in `.github/workflows/cmake-build-test.yml`

After the build step (around line 85, after `ctest`), add:

```yaml
      - name: Check settings registry
        run: |
          cmake --build --preset linux-default --target check-settings
```

### 4. Make the script executable

```bash
chmod +x utilities/check-settings-registry.py
```

## Test

```bash
# Build oflm first
cmake --build --preset linux-default --target oflm

# Run the check
python3 utilities/check-settings-registry.py --oflm build/src/oflm --source-dir src

# Or via CMake target
cmake --build --preset linux-default --target check-settings

# Verify it catches issues: temporarily add a fake env var to a source file
# echo 'getenv_oflm("OFLM_FAKE_VAR");' >> src/common/utils.cpp
# python3 utilities/check-settings-registry.py ...  # Should FAIL
# git checkout src/common/utils.cpp  # Revert
```

## Success criteria

- [ ] `utilities/check-settings-registry.py` created and executable
- [ ] Script checks: env vars, CLI flags, schema sync, registry completeness
- [ ] Script exits 0 when all checks pass
- [ ] Script exits 1 with helpful messages when checks fail
- [ ] CMake target `check-settings` added to `src/CMakeLists.txt`
- [ ] `cmake --build --preset linux-default --target check-settings` works
- [ ] CI step added to `.github/workflows/cmake-build-test.yml`
- [ ] Script catches a fake unregistered env var (manual test)
- [ ] Script catches schema drift (manual test: modify registry, don't run inject-schema)
- [ ] All existing tests still pass
