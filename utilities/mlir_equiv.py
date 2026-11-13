"""Emit the MLIR of every kernel set the specs build, to diff a `git archive` of the base against the tree (2026-10-16)."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="an open_kernels directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default="dense/dx.py,dense/dx_attn.py,ln/ln.py,lm_head_q4/lm_head_q4.py",
                    help="comma-separated designs, or `all` for every set each spec builds")
    ap.add_argument("--specs", default="", help="spec stems (default: every recipes/specs/*.json)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name("npu2", n_cols=None))
    from recipes.families import for_spec
    from recipes.load import load_spec
    wanted = None if a.designs == "all" else set(a.designs.split(","))
    stems = [s for s in a.specs.split(",") if s] or sorted(p.stem for p in (root / "recipes" / "specs").glob("*.json"))
    bad = 0
    for stem in stems:
        sf = root / "recipes" / "specs" / f"{stem}.json"
        spec = load_spec(sf)
        try:
            builds = for_spec(spec).builds(spec)
        except Exception as e:
            bad += 1
            print(f"FAIL {stem}: builds() raised {type(e).__name__}: {e}")
            continue
        for name, b in sorted(builds.items()):
            if wanted is not None and b["design"] not in wanted:
                continue
            saved = dict(os.environ)
            os.environ.update(b["env"])
            os.environ["OPEN_KERNELS_SPEC"] = str(sf)
            src = root / "designs" / b["design"]
            try:
                ms = importlib.util.spec_from_file_location(f"d_{stem}_{name}".replace("-", "_").replace(".", "_"), src)
                mod = importlib.util.module_from_spec(ms)
                sys.path.insert(0, str(src.parent))
                ms.loader.exec_module(mod)
                text = str(mod.DESIGN.specialize(**mod.SPECIALIZE).as_mlir())
                (out / f"{stem}.{name}.mlir").write_text(text, encoding="utf-8", newline="\n")
                print(f"ok   {stem} {name} ({len(text)} chars)")
            except Exception:
                bad += 1
                print(f"FAIL {stem} {name}")
                traceback.print_exc(limit=3)
            finally:
                sys.path.remove(str(src.parent))
                os.environ.clear()
                os.environ.update(saved)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
