#!/usr/bin/env python3
"""Validate a generated spec's geometry before it ships.

    python utilities/register-spec.py qwen3-0.6b [more-specs...]

This used to append the spec's hash and export name to two frozen lists in the
test files, which made every new base/size a manual, committed edit. A model is
now linked by its GEOMETRY signature (ModelSpec.kernel_hash / serves), not by a
stored digest or a name table, so there is nothing to freeze:

  * the export label is checked by specs/open-engine/tests/test_spec_model_names.py
    (unique, safe, not a GGUF upload suffix);
  * the geometry signature is checked by
    specs/open-engine/tests/test_lfm2.py::test_shipped_specs_have_distinct_geometry_signatures.

What this tool still does is fail early, at generation time, on a spec that
would break either invariant -- a missing geometry, a duplicate kernel hash, or
an export name already claimed by another spec. It edits no files: the dev
commits only the generated spec JSON and the validated catalogue points.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "open_kernels" / "recipes" / "specs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="spec stems (qwen3-0.6b) or paths to spec JSON files")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes.load import load_spec

    # Index the shipped specs once: a new spec must not duplicate a geometry or
    # an export directory that is already there.
    existing_shapes: dict[str, str] = {}
    existing_models: dict[str, str] = {}
    for p in SPECS.glob("*.json"):
        try:
            s = load_spec(p)
        except Exception:
            continue
        existing_shapes.setdefault(s.kernel_hash(), p.name)
        model = (s.extra or {}).get("model")
        if model:
            existing_models.setdefault(model, p.name)

    bad = False
    for item in a.specs:
        p = Path(item)
        sp = p if p.is_file() else SPECS / (item if item.endswith(".json") else f"{item}.json")
        if not sp.is_file():
            print(f"register-spec: no such spec: {item}", file=sys.stderr)
            bad = True
            continue
        try:
            s = load_spec(sp)
        except Exception as exc:
            print(f"register-spec: {sp.name}: could not load spec: {exc}", file=sys.stderr)
            bad = True
            continue
        shape = s.kernel_hash()
        model = (s.extra or {}).get("model", "")
        if not model:
            print(f"register-spec: {sp.name}: extra.model is missing", file=sys.stderr)
            bad = True
            continue
        if existing_shapes.get(shape, sp.name) != sp.name:
            print(f"register-spec: {sp.name}: duplicate kernel geometry of "
                  f"{existing_shapes[shape]} -- link to it instead", file=sys.stderr)
            bad = True
            continue
        if existing_models.get(model, sp.name) != sp.name:
            print(f"register-spec: {sp.name}: export name {model!r} already claimed by "
                  f"{existing_models[model]}", file=sys.stderr)
            bad = True
            continue
        print(f"{sp.stem}: geometry {shape[:19]}, export {model}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
