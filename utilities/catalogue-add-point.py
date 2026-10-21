#!/usr/bin/env python3
"""Record a VALIDATED point in the op catalogue.

    python utilities/catalogue-add-point.py --op attn \
        --point "64, 16, 8, 64, True, False, False, False" --note "Qwen3-0.6B"

The catalogue is the list of points a kernel has actually been run at, so
adding one is a claim about the hardware. This script makes that claim
mechanical: it checks the point against the template's keys, refuses a
duplicate, edits the source in place with a comment that says where the point
came from, and verifies afterwards that the catalogue really contains it.

Run it AFTER the point's fixture has passed -- the gate that does that is
utilities/build-new-spec.py (the attention geometry) or the design's own
make_test.py / compare.py.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "open_kernels" / "recipes" / "catalogue.py"


def load_catalogue():
    spec = importlib.util.spec_from_file_location("_catalogue_probe", CAT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def template_node(tree: ast.Module, op: str) -> ast.Call:
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "Template" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == op):
            return node
    raise SystemExit(f"catalogue-add-point: no template {op!r} in {CAT}")


def combination_call(tpl: ast.Call) -> ast.Call:
    for kw in tpl.keywords:
        if kw.arg != "combos":
            continue
        value = kw.value
        if isinstance(value, ast.Tuple) and value.elts:   # combos=(combination(...),)
            value = value.elts[0]
        if isinstance(value, ast.Call):
            return value
    raise SystemExit("catalogue-add-point: that template has no combination() to add to")


def parse_point(text: str) -> tuple:
    try:
        value = ast.literal_eval("(" + text + ",)" if not text.strip().startswith("(") else text)
    except (ValueError, SyntaxError) as exc:
        raise SystemExit(f"catalogue-add-point: cannot read the point: {exc}")
    if not isinstance(value, tuple):
        raise SystemExit("catalogue-add-point: the point must be a tuple")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True, help="the template's name (attn, gemv_q4, ln, ...)")
    ap.add_argument("--point", required=True, help="the validated point, e.g. \"64, 16, 8, 64, True, False\"")
    ap.add_argument("--note", default="", help="what the point is (a model, a family, a date)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    point = parse_point(a.point)
    text = CAT.read_text()
    tree = ast.parse(text)
    combo = combination_call(template_node(tree, a.op))

    keys = None
    for kw in combo.keywords:
        if kw.arg == "keys":
            keys = ast.literal_eval(kw.value)
    if keys is None:
        raise SystemExit(f"catalogue-add-point: the {a.op} combination has no keys")
    if len(point) != len(keys):
        raise SystemExit(f"catalogue-add-point: the {a.op} point has {len(point)} values, "
                         f"the combination has {len(keys)} ({', '.join(keys)})")

    mod = load_catalogue()
    tpl = mod.CATALOGUE.get(a.op)
    if tpl is None:
        raise SystemExit(f"catalogue-add-point: no template {a.op!r}")
    for c in tpl.combos:
        if point in c.points:
            print(f"catalogue-add-point: {a.op} {point} is already validated")
            return 0
        if len(c.keys) == len(keys) and tuple(keys) == tuple(c.keys):
            target = c

    # the source line to insert after: the LAST point argument of the call
    points = [a for a in combo.args if isinstance(a, ast.Tuple)]
    if not points:
        raise SystemExit("catalogue-add-point: the combination has no points")
    last = max(points, key=lambda e: e.end_lineno)
    lines = text.splitlines(keepends=True)
    # the last point's comment can run over several lines; the new point goes
    # after the comment, not inside it
    at = last.end_lineno
    while at < len(lines) and lines[at].lstrip().startswith("#"):
        at += 1
    # the code's own indentation (the comment lines under a tuple are indented
    # to the comment column, not to the code)
    code_line = lines[last.lineno - 1]
    indent = code_line[: len(code_line) - len(code_line.lstrip())]
    # the column the trailing comments line up in
    col = code_line.find("#")
    literal = "(" + ", ".join(repr(v) for v in point) + "),"
    if a.note:
        gap = max(1, col - len(indent) - len(literal)) if col > 0 else 2
        insert = f"{indent}{literal}{' ' * gap}# {a.note}\n"
    else:
        insert = f"{indent}{literal}\n"
    if a.dry_run:
        print(f"would insert after line {at}:\n{insert}")
        return 0

    lines.insert(at, insert)
    new_text = "".join(lines)
    ast.parse(new_text)                      # the edit must leave a parseable file
    CAT.write_text(new_text)

    mod = load_catalogue()
    tpl = mod.CATALOGUE[a.op]
    for c in tpl.combos:
        if point in c.points:
            print(f"catalogue-add-point: {a.op} {point} added"
                  + (f" ({a.note})" if a.note else ""))
            return 0
    print("catalogue-add-point: FAILED to add the point (the catalogue does not contain it)",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
