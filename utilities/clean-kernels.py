#!/usr/bin/env python3
"""Remove the scratch a kernel export leaves behind.

    python utilities/clean-kernels.py            # scratch only
    python utilities/clean-kernels.py --sets     # ... and the exported sets in src/xclbins

`cmake --build build --target clean-kernels` runs the first form, `distclean-kernels` the
second. CMake owns the name `clean` (build outputs), so these are separate targets.

What counts as scratch: `open_kernels/.staging_*` (a per-run copy of designs/), the
diffusion's `build/dit-kernels`, and `build/attn-gate` (the attention gate's artifacts).
`--sets` adds `src/xclbins/*`, the exported sets -- that is the "rebuild every kernel from
the cached design builds" case, and it is why it is a separate target.

A path this user cannot remove -- a staging dir from a root-run build -- is REPORTED with
the command that would remove it, and the run continues. That is deliberate: a leftover is
scratch, the export gives every run a unique staging name so a stale one cannot poison a
build, and a `clean` that fails on it would be the same deceptive failure it exists to end.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "open_kernels"
SCRATCH = [OPEN, ROOT / "build" / "dit-kernels", ROOT / "build" / "attn-gate"]


def _remove(path: Path, stuck: list[Path]) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except OSError:
        stuck.append(path)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", action="store_true",
                    help="also remove the exported sets in src/xclbins (a full kernel re-export)")
    a = ap.parse_args()

    stuck: list[Path] = []
    removed = 0
    for d in sorted(OPEN.glob(".staging_*")):
        removed += _remove(d, stuck)
    for d in SCRATCH[1:]:
        removed += _remove(d, stuck)
    if a.sets:
        for d in sorted((ROOT / "src" / "xclbins").iterdir()):
            if d.name.startswith("."):
                continue
            removed += _remove(d, stuck)

    print(f"clean-kernels: removed {removed} path(s)"
          + (" (including src/xclbins/*)" if a.sets else ""))
    if stuck:
        print("\nnot removable by this user -- a build ran as root. Remove with:", file=sys.stderr)
        for p in stuck[:10]:
            print(f"  sudo rm -rf {p}", file=sys.stderr)
        if len(stuck) > 10:
            print(f"  ... and {len(stuck) - 10} more", file=sys.stderr)
        print("(harmless to leave: every export stages under a unique name, so a stale one is "
              "never reused)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
