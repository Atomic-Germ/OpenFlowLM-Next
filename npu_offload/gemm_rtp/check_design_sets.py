#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the built design sets against families.json.

WHY THIS EXISTS. A design set is selected at load time by geometry AND
datapath, and the runtime refuses a mismatched pair -- so a set built with the
wrong flags is not a slower design, it is one the wrong model loads or none
does. None of that is visible from a successful build: every wrong flag here
produces a perfectly valid design set.

Two such defects shipped, and both were found by accident from outside:

  * `BERT-h384-bf16` was documented with `--c-bf16` against a validated set
    whose `c_dtype` is `f32`. bge-small is the one model held back from the
    aggressive datapath for failing the MTEB gate, so following the wrong
    command put precisely the conservative model on a narrower accumulator than
    it was validated for. The runtime reads `c_dtype` and adapts: nothing
    crashes, nothing warns, the numbers change.
  * `BERT-h1024-bfp16` shipped with one batch tier because `--batches` was
    omitted and defaults to "just --batch". Every request is then padded to
    batch 128 -- 18.3x on a single text, with the four-tier design
    bit-identical.

This used to parse README.md, which was a transcript of the commands rather
than the commands themselves -- the same prose-versus-artifact gap one level
up. It reads families.json now, which is what build.ps1 actually builds from.

Usage:
    python check_design_sets.py [--xclbins DIR]

Exits non-zero on any disagreement, and on a family that is not built.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def expected(argv: list[str]) -> dict:
    """What design.json must say, given a family's flags."""
    def val(flag, cast=str, default=None):
        return cast(argv[argv.index(flag) + 1]) if flag in argv else default

    hidden = val("--hidden", int, 384)
    batch = val("--batch", int, 128)
    batches = val("--batches")
    return {
        "hidden": hidden,
        # export_gemm_rtp.py: --intermediate defaults to 4*hidden ...
        "intermediate": val("--intermediate", int, 4 * hidden),
        # ... and --qkv-n to 3*hidden.
        "qkv_n": val("--qkv-n", int, 3 * hidden),
        "gated_ffn": "--gated-ffn" in argv,
        "emulate_bfp16": "--emulate-bfp16" in argv,
        # --c-bf16 narrows C on the core; without it C stays fp32.
        "c_dtype": "bf16" if "--c-bf16" in argv else "f32",
        "tile_n": val("-n", int, 48),
        "cols": val("--cols", int, 8),
        # --batches defaults to "just --batch". THAT DEFAULT is the one-tier
        # bug above: omit the flag and you silently get a single tier.
        "tiers": sorted(int(b) for b in batches.split(",")) if batches else [batch],
        "tg_depth": val("--tg-depth", int),
        "a_dtype": "int8" if "--int8" in argv else "bf16",
    }


def actual(d: dict) -> dict:
    return {
        "hidden": d.get("hidden"),
        "intermediate": d.get("intermediate"),
        "qkv_n": d.get("qkv_n"),
        "gated_ffn": bool(d.get("gated_ffn")),
        "emulate_bfp16": bool(d.get("emulate_bfp16")),
        "c_dtype": d.get("c_dtype", "f32"),
        "tile_n": (d.get("tile") or {}).get("n"),
        "cols": d.get("cols"),
        "tiers": sorted(d.get("tiers") or []),
        "tg_depth": d.get("tg_depth"),
        "a_dtype": d.get("a_dtype", "bf16"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xclbins", default=str(HERE.parent.parent / "src" / "xclbins"),
                    help="directory holding the built design families")
    ap.add_argument("--spec", default=str(HERE / "families.json"))
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    common = spec["common"]
    root = Path(args.xclbins)

    bad = 0
    for fam in spec["families"]:
        name = fam["name"]
        dj = root / name / "gemm_rtp" / "design.json"
        if not dj.is_file():
            print(f"MISSING  {name}: not built ({dj})")
            bad += 1
            continue
        want = expected(list(fam["args"]) + list(common))
        got = actual(json.loads(dj.read_text(encoding="utf-8")))
        diff = {k: (want[k], got[k]) for k in want
                if want[k] is not None and want[k] != got[k]}
        if diff:
            bad += 1
            print(f"MISMATCH {name}:")
            for k, (w, g) in sorted(diff.items()):
                print(f"           {k}: families.json says {w!r}, "
                      f"design.json says {g!r}")
        else:
            print(f"ok       {name}")

    if bad:
        print(f"\n{bad} famil{'y' if bad == 1 else 'ies'} disagree with "
              f"{Path(args.spec).name}.")
        print("Either the set is stale -- rebuild it with build.ps1 -Force -- or "
              "families.json is wrong, in which case fix it there and nowhere "
              "else: it is the only place these flags are written down.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
