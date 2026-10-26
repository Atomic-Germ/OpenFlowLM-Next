# Task 18: Schema Injection Script + Release Workflow

**Phase:** 6 (Schema Generation at Package Time)
**Depends on:** Task 12 (needs `config gen-schema` command)
**Files to create:** `utilities/release/inject-schema.py`
**Files to modify:** `.github/workflows/release.yml`
**Estimated complexity:** Small (~80 lines)

## Context

At package time, inject the `settings_schema` and `defaults_version` blocks into `model_list.json`. The `oflm config gen-schema` command outputs both as JSON; this script merges them into the shipped registry.

## What to do

1. Create `utilities/release/inject-schema.py`
2. Add a step to `.github/workflows/release.yml` to run it before cpack

## Code

### `utilities/release/inject-schema.py`

```python
#!/usr/bin/env python3
"""Inject settings_schema and defaults_version into model_list.json.

Usage:
    inject-schema.py --input model_list.json --schema schema.json --output model_list.json

The schema.json file is the output of `oflm config gen-schema`:
    {
      "settings_schema": { "version": 1, "settings": [...] },
      "defaults_version": "a3f2b8c1"
    }
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Inject settings schema into model_list.json")
    parser.add_argument("--input", required=True, help="Input model_list.json path")
    parser.add_argument("--schema", required=True, help="Schema JSON from oflm config gen-schema")
    parser.add_argument("--output", required=True, help="Output model_list.json path")
    args = parser.parse_args()

    # Load input registry
    with open(args.input, "r") as f:
        registry = json.load(f)

    # Load schema
    with open(args.schema, "r") as f:
        schema = json.load(f)

    # Inject settings_schema
    if "settings_schema" in schema:
        registry["settings_schema"] = schema["settings_schema"]
        print(f"Injected settings_schema (version {schema['settings_schema'].get('version', '?')}, "
              f"{len(schema['settings_schema'].get('settings', []))} settings)")

    # Inject defaults_version
    if "defaults_version" in schema:
        registry["defaults_version"] = schema["defaults_version"]
        print(f"Injected defaults_version: {schema['defaults_version']}")

    # Ensure top-level version exists
    if "version" not in registry:
        registry["version"] = 1

    # Ensure defaults block exists
    if "defaults" not in registry:
        registry["defaults"] = {}

    # Write output
    with open(args.output, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")

    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
```

### `.github/workflows/release.yml`

Find the Linux packaging job (around line 211-260) and add a step before `cpack`:

```yaml
- name: Inject settings schema
  run: |
    ./build/src/oflm config gen-schema > /tmp/schema.json
    python3 utilities/release/inject-schema.py \
      --input src/model_list.json \
      --schema /tmp/schema.json \
      --output src/model_list.json
    echo "Schema injected. Verifying..."
    python3 -c "import json; d=json.load(open('src/model_list.json')); assert 'settings_schema' in d; assert 'defaults_version' in d; print('OK:', len(d['settings_schema']['settings']), 'settings')"
```

## Test

```bash
# Build oflm first
cmake --build --preset linux-default --target oflm 2>&1 | tail -5

# Generate schema
./build/src/oflm config gen-schema > /tmp/schema.json

# Test injection on a copy
cp src/model_list.json /tmp/test_model_list.json
python3 utilities/release/inject-schema.py \
  --input /tmp/test_model_list.json \
  --schema /tmp/schema.json \
  --output /tmp/test_model_list_out.json

# Verify
python3 -c "import json; d=json.load(open('/tmp/test_model_list_out.json')); print('settings_schema:', 'settings_schema' in d); print('defaults_version:', d.get('defaults_version', 'MISSING')); print('settings count:', len(d.get('settings_schema',{}).get('settings',[])))"
```

## Success criteria

- [ ] `utilities/release/inject-schema.py` created and executable
- [ ] Script correctly injects `settings_schema` and `defaults_version`
- [ ] Script preserves existing model_list.json content
- [ ] Release workflow updated with injection step
- [ ] Manual test passes (schema injected into copy of model_list.json)
