#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check that README.md's build commands describe the design sets on disk.

WHY THIS EXISTS. The design sets are built, not checked in, so README.md is not
documentation about the artifacts -- it IS the artifacts, one step removed. A
wrong flag there does not produce a build error; it produces a different, valid,
silently-substituted design.

Both defects this was written after were exactly that shape, and both went
unnoticed until someone deleted the binaries and rebuilt from an empty tree:

  * `BERT-h384-bf16` was documented with `--c-bf16`, but the set that shipped
    (and that passed the accuracy gates) has `c_dtype: f32`. bge-small is the
    one model held back from the aggressive datapath -- it failed the bfp16
    MTEB gate -- so following the README put precisely that model on a narrower
    C than it was validated for. The runtime reads `c_dtype` from design.json
    and adapts, so nothing crashes and nothing warns: the numbers just change.
  * `BERT-h1024-bfp16` was documented with four batch tiers and shipped with
    one.

Neither is detectable by building. Both are detectable by comparing the command
to the design.json it produces, which is all this script does.

Every flag that reaches design.json is compared. A flag that does NOT reach it
cannot be checked here -- see UNCHECKED below, and keep that list honest.

Usage:
    python check_readme.py [--xclbins <dir>]

Exits non-zero on any mismatch, and on a family the README names but the tree
does not have (which, with the sets ignored by git, means "not built yet").
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Flags that do not appear in design.json and so cannot be verified from it.
# --qkv-n IS recorded (as "qkv_n"); --out is the destination. If a flag is added
# to export_gemm_rtp.py that changes the design without reaching design.json,
# that is a bug in the exporter, not an entry for this list.
UNCHECKED = ("--out",)


def parse_commands(readme: Path) -> dict[str, list[str]]:
    """Every `python export_gemm_rtp.py ...` invocation, keyed by output name.

    PowerShell continues lines with a trailing backtick; comment lines start
    with '#' and are dropped.
    """
    text = readme.read_text(encoding="utf-8")
    # Join backtick continuations first so a command is one line.
    text = re.sub(r"`\s*\n\s*", " ", text)
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("python export_gemm_rtp.py"):
            continue
        toks = line.split()[2:]
        if "--out" not in toks:
            raise SystemExit(f"command with no --out: {line}")
        dest = toks[toks.index("--out") + 1]
        name = dest.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        out[name] = toks
    return out


def expected(toks: list[str]) -> dict:
    """What design.json must say, given the flags."""
    def val(flag, cast=str, default=None):
        return cast(toks[toks.index(flag) + 1]) if flag in toks else default

    return {
        "hidden": val("--hidden", int),
        "intermediate": val("--intermediate", int),
        "qkv_n": val("--qkv-n", int),
        "gated_ffn": "--gated-ffn" in toks,
        "emulate_bfp16": "--emulate-bfp16" in toks,
        # --c-bf16 narrows C on the core; without it C stays fp32.
        "c_dtype": "bf16" if "--c-bf16" in toks else "f32",
        "tile_n": val("-n", int) or val("--tile-n", int),
        "tiers": sorted(int(b) for b in (val("--batches") or "").split(",") if b),
        "tg_depth": val("--tg-depth", int),
        "tb_max_n_rows": val("--tb-rows", int),
        "a_dtype": "int8" if "--int8" in toks else "bf16",
    }


def actual(design: dict) -> dict:
    return {
        "hidden": design.get("hidden"),
        "intermediate": design.get("intermediate"),
        "qkv_n": design.get("qkv_n"),
        "gated_ffn": bool(design.get("gated_ffn")),
        "emulate_bfp16": bool(design.get("emulate_bfp16")),
        "c_dtype": design.get("c_dtype", "f32"),
        "tile_n": (design.get("tile") or {}).get("n"),
        "tiers": sorted(design.get("tiers") or []),
        "tg_depth": design.get("tg_depth"),
        "tb_max_n_rows": design.get("tb_max_n_rows"),
        "a_dtype": design.get("a_dtype", "bf16"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xclbins", default=str(HERE.parent.parent / "src" / "xclbins"),
                    help="directory holding the built design families")
    args = ap.parse_args()

    root = Path(args.xclbins)
    cmds = parse_commands(HERE / "README.md")
    if not cmds:
        print("no build commands found in README.md", file=sys.stderr)
        return 2

    bad = 0
    for name, toks in sorted(cmds.items()):
        dj = root / name / "gemm_rtp" / "design.json"
        if not dj.is_file():
            print(f"MISSING  {name}: not built ({dj})")
            bad += 1
            continue
        want, got = expected(toks), actual(json.loads(dj.read_text(encoding="utf-8")))
        diff = {k: (want[k], got[k]) for k in want
                if want[k] is not None and want[k] != got[k]}
        if diff:
            bad += 1
            print(f"MISMATCH {name}:")
            for k, (w, g) in sorted(diff.items()):
                print(f"           {k}: README says {w!r}, design.json says {g!r}")
        else:
            print(f"ok       {name}")

    if bad:
        print(f"\n{bad} famil{'y' if bad == 1 else 'ies'} disagree with README.md.")
        print("The README is what users build from, so it is the artifact: fix "
              "whichever side is wrong, and say in the README why the odd one "
              "out is odd.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
