#!/usr/bin/env python3
"""One-shot: generate spec -> build kernel (unvalidated) -> run fixture -> add to catalogue.

Chains gen-kernel-spec.py, export-kernels.py (with OPEN_KERNELS_UNVALIDATED=1),
the family's make_test/compare, and catalogue-add-point.py. Intended to make
adding a new size a single command instead of manual steps.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, env=None, check=True):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, env=env or os.environ, check=check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--spec-name", required=True, help="spec file stem, e.g. qwen3-0.6b")
    args = ap.parse_args()

    # 1. generate candidate spec
    spec_path = ROOT / "open_kernels" / "recipes" / "specs" / f"{args.spec_name}.json"
    run([sys.executable, str(ROOT / "utilities" / "gen-kernel-spec.py"),
         "--repo", args.repo, "--out", str(spec_path)])

    # 2. build unvalidated
    env = dict(os.environ, OPEN_KERNELS_UNVALIDATED="1")
    run([sys.executable, str(ROOT / "utilities" / "export-kernels.py"),
         "--specs", args.spec_name], env=env)

    print("Build complete. Run the family fixture/compare manually, then:")
    print(f"  python utilities/catalogue-add-point.py --op attn --point '<tuple>'")
    print("Automation of fixture+catalogue append is TODO.")


if __name__ == "__main__":
    main()
