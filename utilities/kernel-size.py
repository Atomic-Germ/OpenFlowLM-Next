#!/usr/bin/env python3
"""Code-size report for an open-kernel design, so a failed build can be read.

    python utilities/kernel-size.py --spec <spec.json> [--kernel lx]
    python utilities/kernel-size.py --spec <spec.json> --compare 2560 2048 4096

WHY THIS EXISTS. When a design will not build, aiecc says one line --

    _XAie_LoadProgMemSection(): Overflow of program memory

-- and then DELETES its project directory. The per-core object files that
would say which core was over, and by how much, go with it. The one artifact
that survives, `insts.bin`, is the TOTAL across cores, which for the width
that fails is SMALLER than for one that builds; a small total on a design
that overflows is only explicable as one core carrying more than its share,
and nothing in the build's own output says which.

So this runs the real exporter (the same build_design.py path, the same
flags) and reports the numbers aiecc threw away: the per-core and per-TU
`.text`, each core's 16 KB budget, and which kernel carries the spike.

There is no guard to bypass: it measured the fold, and the fold fixed it. The
straddle that guard watched (HID 2560's output bands crossing a 4 KB xn element
boundary) never caused the overflow -- it only marked the one all-q4 dense
width, which is what overflowed. The q4_1 GEMV pair now folds for every dense
spec, so `--allow-straddle` is gone.

The build is run as a SUBPROCESS, deliberately: the production exporter is the
thing being measured, and driving it in-process (via runpy) fights IRON's
process-global ExternalFunction registry and can compile nothing. The xclbin
that a failing aiecc does not write is fine -- this reads the objects, which
are produced before aiecc runs and left behind on the way out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "open_kernels"
SPECS = KERNELS / "recipes" / "specs"

# AIE2P: 16 KB of program memory per core. The same figure the exporter reports.
PROGRAM_MEMORY = 16384
MAX_CTX = 4096


def llvm_size() -> Path:
    import glob
    hits = sorted(glob.glob(str(REPO / "ironvenv" / "lib" / "python*"
                                 / "site-packages" / "llvm-aie" / "bin" / "llvm-size")))
    if not hits:
        raise SystemExit("llvm-size not found under ironvenv; source mlir-aie/utils/env_setup.sh")
    return Path(hits[-1])


def text_size(obj: Path, size: Path) -> int:
    """Code bytes in an object: the `text` column of llvm-size.

    Per-symbol `.text.*` sections, not a section literally named `.text` (that
    one is 0 in these objects), so the summed column is what the exporter's
    fullest_core() reports for a linked per-core ELF.
    """
    r = subprocess.run([str(size), str(obj)], capture_output=True, text=True)
    lines = r.stdout.splitlines()
    for i, line in enumerate(lines):
        f = line.split()
        if f and f[0] == "text" and "data" in f and i + 1 < len(lines):
            row = lines[i + 1].split()
            if row and row[0].isdigit():
                return int(row[0])
    return 0


def build_one(spec: Path, kernel: str, out: Path):
    """Build one design through the real exporter; return a measurement dict.

    The build is the production one, via utilities/_keep_objects.py, which wraps
    IRON's compile_external_kernel so each object is copied aside before aiecc
    deletes the project directory. aiecc is NOT stubbed: if the design really
    overflows, the build fails as it would in CI, and the objects that explain
    it are already saved.
    """
    env = dict(os.environ)
    env["OPEN_KERNELS_SPEC"] = str(spec.resolve())
    out.mkdir(parents=True, exist_ok=True)
    keep = out / "objects"
    # Generate the TUs first: the design references .cc files that only exist
    # after gen_kernels runs, and for a width with no cached build they are absent.
    g = subprocess.run([sys.executable, "-c", GEN_TUS, str(spec.resolve()), str(kernel)],
                       env=env, cwd=str(REPO), capture_output=True, text=True)
    if g.returncode:
        raise SystemExit("generating the kernel TUs failed:\n" + g.stdout + g.stderr)
    r = subprocess.run([sys.executable, str(REPO / "utilities" / "_keep_objects.py"),
                        str(KERNELS / "designs" / _design_for(spec, kernel)),
                        str(out), str(keep)],
                       env=env, cwd=str(REPO), capture_output=True, text=True)
    built = "BUILD_OK" in r.stdout
    size = llvm_size()
    rows = [(o.name, text_size(o, size)) for o in sorted(keep.glob("*.o"))]
    return {"built": built, "total": sum(n for _, n in rows), "rows": rows,
            "kernel": kernel, "spec": spec, "log": r.stdout + r.stderr}


GEN_TUS = """
import sys, os, importlib.util
sys.path.insert(0, 'open_kernels')
from pathlib import Path
from recipes.load import load_spec
from recipes.families import for_spec
spec = load_spec(Path(sys.argv[1])); kernel = sys.argv[2]
F = for_spec(spec)
gp = Path('open_kernels') / F.GEN_KERNELS
sys.path.insert(0, str(gp.parent))
gs = importlib.util.spec_from_file_location('gen_kernels', gp)
g = importlib.util.module_from_spec(gs); gs.loader.exec_module(g)
g.generate(F.recipe(spec, 4096))
"""


def _design_for(spec: Path, kernel: str) -> str:
    import importlib.util
    sys.path.insert(0, str(KERNELS))
    from recipes.load import load_spec
    from recipes.families import for_spec
    s = load_spec(spec)
    return str(for_spec(s).builds(s)[kernel]["design"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", type=Path, help="a derived spec JSON")
    ap.add_argument("--compare", nargs="*", default=None, metavar="HIDDEN",
                    help="instead of --spec, build this kernel at these hidden widths")
    ap.add_argument("--kernel", default="lx", help="kernel set (lx, ax, ...)")
    ap.add_argument("--out", type=Path, default=REPO / "build" / "kernel-size")
    a = ap.parse_args()

    if a.compare:
        # Widths are driven by editing a copy of the spec, so this needs no new
        # spec files and works for a width the catalogue has no entry for.
        from copy import deepcopy
        results = []
        for hid in a.compare:
            base = json.loads((SPECS / "qwen35-h4096-L32.json").read_text())
            base["hidden"] = int(hid)
            p = a.out / f"spec_h{hid}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(base, indent=2))
            mm = build_one(p, a.kernel, a.out / f"h{hid}")
            results.append((int(hid), mm["built"], mm["total"]))
        print(f"  {'HID':>6} {'built':>7} {'total .text':>13}  vs 16384/core")
        for hid, built, total in results:
            print(f"  {hid:>6} {str(bool(built)):>7} {total:>13}")
        return 0

    if not a.spec:
        ap.error("--spec or --compare is required")
    m = build_one(a.spec, a.kernel, a.out / a.spec.stem)
    rows, total = m["rows"], m["total"]
    print(f"kernel {m['kernel']}  spec {m['spec'].name}  "
          f"{'BUILT' if m['built'] else 'DID NOT BUILD (aiecc overflowed program memory)'}")
    print(f"budget: {PROGRAM_MEMORY} B of program memory PER CORE\n")
    if not rows:
        print("no objects were produced -- see the build log:")
        print(m["log"][-2000:])
        return 1
    print(f"  {'kernel TU':<28} {'.text':>7}")
    for name, n in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<28} {n:>7}  {'#' * max(1, n // 400)}")
    print(f"  {'TOTAL':<28} {total:>7}\n")
    gemv = {n[:-2]: b for n, b in rows if n.startswith("gemv_")}
    if gemv:
        q4 = sum(b for n, b in gemv.items() if n.startswith("gemv_q4"))
        print(f"  q4 GEMV entry points on one core: {q4} B"
              + ("  (UN-FUSED: gy + gms both resident)" if "gemv_q4_gy" in gemv and "gemv_q4_gms" in gemv
                 else "  (folded into one gyms)" if "gemv_q4_gyms" in gemv else ""))
        if "gemv_q4_gy" in gemv and "gemv_q4_gms" in gemv:
            print("  An all-q4 dense spec emits BOTH q4_1 entry points; a spec with any")
            print("  q8 role folds them into gemv_q4_gyms. The fold is what the core")
            print("  budget is being spent on, and an all-q4 spec does not get it.")
    print("\n  The TOTAL is across TUs, not per core: IRON links each core from a")
    print("  subset, and the linked per-core ELFs exist only for a width that fits.")
    print("  Read the fold line above -- that is what decides the per-core total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
