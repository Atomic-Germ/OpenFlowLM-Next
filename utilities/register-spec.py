#!/usr/bin/env python3
"""Register a shipped spec in the two frozen lists the tests enforce.

    python utilities/register-spec.py qwen3-0.6b [more-specs...]

A new spec in open_kernels/recipes/specs/ breaks two tests until it is listed:

  * specs/open-engine/tests/test_lfm2.py, test_no_shipped_models_hash_moved:
    the frozen {spec file: spec_hash} map -- it exists so a ModelSpec field
    change that silently rehashes every shipped model is caught. A new spec
    joins it at the hash it ships with.
  * specs/open-engine/tests/test_spec_model_names.py, EXPECTED: only when the
    name the test derives from the file stem (<family>-<size> -> Family-SIZE-NPU2)
    is not the model name the spec exports as (extra.model).

Both edits are mechanical, so they are scripted, not hand-written: this is the
last step of utilities/build-new-spec.py, and it is idempotent -- running it on
an already-registered spec changes nothing. It edits text, preserving the
files' formatting, and refuses a spec it cannot load.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "open_kernels" / "recipes" / "specs"
TEST_LFM2 = ROOT / "specs" / "open-engine" / "tests" / "test_lfm2.py"
TEST_NAMES = ROOT / "specs" / "open-engine" / "tests" / "test_spec_model_names.py"

HASH_ENTRY = re.compile(r'^(?P<indent>\s*)"(?P<name>[^"]+\.json)": "(?P<hash>sha256:[0-9a-f]+)",\s*$')
NAME_ENTRY = re.compile(r'^(?P<indent>\s*)"(?P<stem>[^"]+)": "(?P<model>[^"]+)",\s*$')


def derived_model(stem: str) -> str:
    """test_spec_model_names.expected_model, for stems not in EXPECTED."""
    family, _, size = stem.rpartition("-")
    return f"{family}-{size.upper()}-NPU2".replace("gemma3", "Gemma3")


def update_frozen_hash(path: Path, filename: str, spec_hash: str) -> str:
    lines = path.read_text().splitlines(keepends=True)
    entries = [i for i, ln in enumerate(lines) if HASH_ENTRY.match(ln)]
    if not entries:
        raise SystemExit(f"register-spec: no frozen hash entries found in {path}")
    for i in entries:
        m = HASH_ENTRY.match(lines[i])
        if m["name"] == filename:
            if m["hash"] == spec_hash:
                return "already registered"
            lines[i] = f'{m["indent"]}"{filename}": "{spec_hash}",\n'
            path.write_text("".join(lines))
            return f"hash updated (was {m['hash'][:19]}...)"
    last = entries[-1]
    indent = HASH_ENTRY.match(lines[last])["indent"]
    lines.insert(last + 1, f'{indent}"{filename}": "{spec_hash}",\n')
    path.write_text("".join(lines))
    return "registered"


def update_expected_name(path: Path, stem: str, model: str) -> str:
    if derived_model(stem) == model:
        return "derived name matches; no EXPECTED entry needed"
    lines = path.read_text().splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if ln.startswith("EXPECTED = {")), None)
    if start is None:
        raise SystemExit(f"register-spec: no EXPECTED dict found in {path}")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip() == "}")
    for i in range(start + 1, end):
        m = NAME_ENTRY.match(lines[i])
        if m and m["stem"] == stem:
            if m["model"] == model:
                return "already registered"
            lines[i] = f'{m["indent"]}"{stem}": "{model}",\n'
            path.write_text("".join(lines))
            return f"name updated (was {m['model']})"
    indent = NAME_ENTRY.match(lines[start + 1])["indent"] if end > start + 1 else "    "
    lines.insert(end, f'{indent}"{stem}": "{model}",\n')
    path.write_text("".join(lines))
    return "registered"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="spec stems (qwen3-0.6b) or paths to spec JSON files")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes.load import load_spec

    for item in a.specs:
        p = Path(item)
        spec_path = p if p.is_file() else SPECS / (item if item.endswith(".json") else f"{item}.json")
        if not spec_path.is_file():
            print(f"register-spec: no such spec: {item}", file=sys.stderr)
            return 1
        spec = load_spec(spec_path)
        model = (spec.extra or {}).get("model", "")
        if not model:
            print(f"register-spec: {spec_path.name}: extra.model is missing", file=sys.stderr)
            return 1
        h = update_frozen_hash(TEST_LFM2, spec_path.name, spec.spec_hash())
        n = update_expected_name(TEST_NAMES, spec_path.stem, model)
        print(f"{spec_path.stem}: hash {h}; name {n}")
    print("commit: specs/open-engine/tests/test_lfm2.py specs/open-engine/tests/test_spec_model_names.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
