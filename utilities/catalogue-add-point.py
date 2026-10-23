#!/usr/bin/env python3
"""Record a VALIDATED point in the op catalogue.

    python utilities/catalogue-add-point.py --op attn \
        --point "64, 16, 8, 64, True, False, False, False" --note "Qwen3-0.6B"

Whole-tuple templates get --point; a parametric template's values() set gets
--param/--value:

    python utilities/catalogue-add-point.py --op lm_head_q4 \
        --param K --value 5120 --note "Qwen3-14B (the lm_head_q4 fixture)"

The catalogue is the list of points a kernel has actually been run at, so
adding one is a claim about the hardware. This script makes that claim
mechanical: it checks the point against the template's keys, refuses a
duplicate, edits the source in place with a comment that says where the point
came from, and verifies afterwards that the catalogue really contains it.

Run it AFTER the point's fixture has passed -- the gate that does that is
utilities/build-new-spec.py (the attention geometry), a values() template's
own designs/<op>/make_test.py + compare.py, or the design's whole-layer run.
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


def values_call(tpl: ast.Call, param: str) -> tuple[ast.Call, ast.Dict]:
    """The values(...) Call behind the template's `param`, and the params dict
    it sits in (the dict can be a positional arg: Template(name, source, {..}))."""
    dicts = []
    for i, arg in enumerate(tpl.args):
        if i >= 2 and isinstance(arg, ast.Dict):        # args[2] = params
            dicts.append(arg)
    for kw in tpl.keywords:
        if kw.arg == "params" and isinstance(kw.value, ast.Dict):
            dicts.append(kw.value)
    for d in dicts:
        for key, v in zip(d.keys, d.values):
            if isinstance(key, ast.Constant) and key.value == param:
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "values":
                    return v, d
                raise SystemExit(f"catalogue-add-point: {param!r} is not a values() Param "
                                 f"(a rule-based Param has no enumerable set to extend)")
    raise SystemExit(f"catalogue-add-point: the template has no param {param!r}")


def append_value(text: str, call: ast.Call, value, note: str, params: ast.Dict | None) -> str:
    """`, value` right after the values() call's last positional argument, and
    the note as a comment at the end of the line the params dict closes on
    (a mid-line comment is illegal inside most values() layouts, so the note
    does not follow the value itself)."""
    if not call.args:
        raise SystemExit("catalogue-add-point: the values() has no argument to insert after")
    last = call.args[-1]
    lines = text.splitlines(keepends=True)
    pos = sum(len(ln) for ln in lines[:last.end_lineno - 1]) + last.end_col_offset
    frag = f" {value!r}" if text[pos] == "," else f", {value!r}"
    line = lines[last.end_lineno - 1]
    lines[last.end_lineno - 1] = line[:last.end_col_offset] + frag + line[last.end_col_offset:]
    if note and params is not None:
        li = params.end_lineno - 1
        lines[li] = lines[li].rstrip("\n") + f"  # {note}"
        if text.splitlines(keepends=True)[li].endswith("\n"):
            lines[li] += "\n"
    return "".join(lines)


def parse_point(text: str) -> tuple:
    try:
        value = ast.literal_eval("(" + text + ",)" if not text.strip().startswith("(") else text)
    except (ValueError, SyntaxError) as exc:
        raise SystemExit(f"catalogue-add-point: cannot read the point: {exc}")
    if not isinstance(value, tuple):
        raise SystemExit("catalogue-add-point: the point must be a tuple")
    return value


def insert_at(text: str, call: ast.Call, literal: str, note: str) -> str:
    """`literal` with its comment, after call's last argument (its comment may
    run over several lines; the new line goes AFTER it, not inside it), with
    the code's own indent and the column the trailing comments line up in."""
    args = [a for a in call.args if isinstance(a, (ast.Tuple, ast.Constant))]
    if not args:
        raise SystemExit("catalogue-add-point: the call has no points to insert after")
    last = max(args, key=lambda e: e.end_lineno)
    lines = text.splitlines(keepends=True)
    at = last.end_lineno
    while at < len(lines) and lines[at].lstrip().startswith("#"):
        at += 1
    # the code's own indentation (the comment lines under a tuple are indented
    # to the comment column, not to the code)
    code_line = lines[last.lineno - 1]
    indent = code_line[: len(code_line) - len(code_line.lstrip())]
    col = code_line.find("#")
    if note:
        gap = max(1, col - len(indent) - len(literal)) if col > 0 else 2
        line = f"{indent}{literal}{' ' * gap}# {note}\n"
    else:
        line = f"{indent}{literal}\n"
    lines.insert(at, line)
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True, help="the template's name (attn, gemv_q4, ln, ...)")
    ap.add_argument("--point", default="", help="the validated point, e.g. \"64, 16, 8, 64, True, False\"")
    ap.add_argument("--param", default="", help="values() write: the param's name (e.g. K)")
    ap.add_argument("--value", default="", help="values() write: the validated value (e.g. 5120)")
    ap.add_argument("--note", default="", help="what the point is (a model, a family, a date)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.param:
        if not a.value or a.point:
            raise SystemExit("catalogue-add-point: --param takes --value (= the validated number), "
                             "not --point")
        try:
            value = ast.literal_eval(a.value)
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"catalogue-add-point: cannot read the value: {exc}")
        text = CAT.read_text()
        tree = ast.parse(text)
        call, params = values_call(template_node(tree, a.op), a.param)

        mod = load_catalogue()
        tpl = mod.CATALOGUE[a.op]
        param = tpl.params.get(a.param)
        if param is None:
            raise SystemExit(f"catalogue-add-point: the template has no param {a.param!r}")
        if value in param.values:
            print(f"catalogue-add-point: {a.op} {a.param}={value!r} is already validated")
            return 0
        if a.dry_run:
            print(f"would append {value!r} to the {a.op} {a.param} values()"
                  + (f", with the note {a.note!r}" if a.note else ""))
            return 0
        new_text = append_value(text, call, value, a.note, params)
        ast.parse(new_text)                      # the edit must leave a parseable file
        CAT.write_text(new_text)

        mod = load_catalogue()
        if value in mod.CATALOGUE[a.op].params[a.param].values:
            print(f"catalogue-add-point: {a.op} {a.param}={value!r} added"
                  + (f" ({a.note})" if a.note else ""))
            return 0
        print("catalogue-add-point: FAILED to add the value (the catalogue does not contain it)",
              file=sys.stderr)
        return 1

    if not a.point:
        raise SystemExit("catalogue-add-point: give --point (a combination) or --param + --value")
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

    literal = "(" + ", ".join(repr(v) for v in point) + "),"
    if a.dry_run:
        print(f"would append {literal} to the {a.op} combination"
              + (f" with the note {a.note!r}" if a.note else ""))
        return 0
    new_text = insert_at(text, combo, literal, a.note)
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
