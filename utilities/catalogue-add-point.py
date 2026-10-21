#!/usr/bin/env python3
"""Append a validated point to open_kernels/recipes/catalogue.py.

Usage:
    python utilities/catalogue-add-point.py --op attn --point "(128, 16, 8, 128, True, False, False, False)"

This edits the catalogue source so the next build no longer needs
OPEN_KERNELS_UNVALIDATED=1 for that geometry. Run only after the fixture/compare
step has passed.
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "open_kernels" / "recipes" / "catalogue.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--point", required=True, help="Python tuple literal, e.g. (128, 16, 8, 128, True, False, False, False)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = CAT.read_text()
    # Find the combination block for the op by locating its Template entry.
    # We append the point before the closing of that combination's points set.
    # Simplest: insert the point line right before "keys=(" for that op's combo.
    # Locate op section.
    idx = text.find(f'Template("{args.op}"')
    if idx == -1:
        print(f"op {args.op} not found", file=__import__("sys").stderr)
        raise SystemExit(1)
    # find "keys=" after idx
    ki = text.find("keys=(", idx)
    if ki == -1:
        print("no combination keys found", file=__import__("sys").stderr)
        raise SystemExit(1)
    # find the last tuple line before keys=
    # insert point before the line containing keys=
    lines = text.splitlines(keepends=True)
    # find line index of keys= occurrence
    pos = text[:ki].count("\n")
    # insert before that line, with same indentation as previous non-empty line
    indent = ""
    for j in range(pos - 1, -1, -1):
        if lines[j].strip():
            indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            break
    insertion = f"{indent}{args.point},  # auto-added\n"
    if args.dry_run:
        print("would insert:", insertion)
        return
    lines.insert(pos, insertion)
    CAT.write_text("".join(lines))
    print(f"Added {args.point} to {args.op} catalogue.")


if __name__ == "__main__":
    main()
