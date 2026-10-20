#!/usr/bin/env python3
"""Scan source code for CLI flags and env vars. Outputs JSON to stdout.

Usage:
    scan-settings.py --source-dir src
"""
import argparse, json, re, sys
from pathlib import Path

def scan_cli_flags(source_dir):
    """Extract CLI flags from vm_args.hpp."""
    vm_args = Path(source_dir) / "include" / "utils" / "vm_args.hpp"
    if not vm_args.exists():
        return {}
    content = vm_args.read_text()
    flags = {}
    # Pattern: ("name[,alias]", po::value<T>(&parsed_args.field)->default_value(X), "description")
    # or: ("name[,alias]", po::bool_switch(&parsed_args.field), "description")
    # or: ("name[,alias]", po::value<T>(&parsed_args.field), "description")
    #
    # Boost declares option names WITHOUT the leading "--" (e.g. "pmode",
    # "ctx-len,c", "prefill-mode"), so matching a "--"-prefixed name found
    # nothing. The long name is the first comma-separated element; aliases
    # (",c", ",pcl") are dropped. The result is normalized to "--name".
    pattern = re.compile(
        r'\("([a-z0-9-]+(?:,[a-z0-9-]+)*)"\s*,\s*'
        r'po::(value<([^>]+)>|bool_switch)\(&parsed_args\.(\w+)\)'
        r'(?:->default_value\(([^)]+)\))?'
        r'\s*,\s*"([^"]*)"(?:\s*"[^"]*")*\s*\)'  # desc; concatenated string literals allowed
    )
    for m in pattern.finditer(content):
        names, boost_type, value_type, field, default, desc = m.groups()
        long_name = names.split(",")[0]
        flag = "--" + long_name
        type_hint = "bool" if boost_type == "bool_switch" else (
            "int" if value_type and "int" in value_type else "string"
        )
        key = long_name.replace("-", "_")
        flags[flag] = {
            "key": key,
            "description": desc,
            "type_hint": type_hint,
            "field": field,
            "default": default.strip('"') if default else None,
        }
    return flags

def scan_env_vars(source_dir):
    """Extract OFLM_* env var usage from source files."""
    src = Path(source_dir)
    env_vars = {}
    read_pattern = re.compile(r'getenv_oflm\(\s*"(OFLM_[A-Z0-9_]+)"\s*\)')
    set_pattern = re.compile(r'setenv\(\s*"(OFLM_[A-Z0-9_]+)"')
    getenv_pattern = re.compile(r'std::getenv\(\s*"(OFLM_[A-Z0-9_]+)"\s*\)')

    for pattern, kind in [(read_pattern, "read"), (set_pattern, "set"), (getenv_pattern, "read")]:
        for f in list(src.rglob("*.cpp")) + list(src.rglob("*.hpp")):
            if "test" in str(f) or "build" in str(f) or "config" in str(f):
                continue  # Skip tests, build dir, and config dir (avoid self-reference)
            try:
                content = f.read_text(errors="ignore")
            except:
                continue
            for m in pattern.finditer(content):
                var = m.group(1)
                line_num = content[:m.start()].count("\n") + 1
                if var not in env_vars:
                    env_vars[var] = {"read_locations": [], "set_locations": []}
                if kind == "read":
                    env_vars[var]["read_locations"].append(f"{f.relative_to(src)}:{line_num}")
                else:
                    env_vars[var]["set_locations"].append(f"{f.relative_to(src)}:{line_num}")
    return env_vars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="src")
    args = parser.parse_args()

    result = {
        "cli_flags": scan_cli_flags(args.source_dir),
        "env_vars": scan_env_vars(args.source_dir),
    }
    json.dump(result, sys.stdout, indent=2)

if __name__ == "__main__":
    main()
