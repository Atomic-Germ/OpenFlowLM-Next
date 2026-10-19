#!/usr/bin/env python3
"""Inject the settings schema, the app-wide defaults and the defaults
version into model_list.json.

Usage:
    inject-schema.py --input model_list.json --schema schema.json --output model_list.json

The schema.json file is the output of `oflm config gen-schema`:
    {
      "settings_schema": { "version": 1, "settings": [...] },
      "defaults": { "pmode": "performance", "prefill_mode": "fast" },
      "defaults_version": "a3f2b8c1"
    }

The defaults block is what makes the shipped registry carry every config key:
a user who has not set one still has a value to show, tagged (global).
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

    # Inject the app-wide defaults. What the file already says wins, so a
    # release can pin a default without a code change.
    if "defaults" in schema:
        existing = registry.get("defaults", {})
        merged = dict(schema["defaults"])
        merged.update(existing)
        registry["defaults"] = merged
        print(f"Injected defaults: {', '.join(sorted(merged))}")

    # Inject defaults_version
    if "defaults_version" in schema:
        registry["defaults_version"] = schema["defaults_version"]
        print(f"Injected defaults_version: {schema['defaults_version']}")

    # Ensure top-level version exists
    if "version" not in registry:
        registry["version"] = 1

    # Write output
    with open(args.output, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")

    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
