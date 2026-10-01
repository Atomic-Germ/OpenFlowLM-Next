#!/usr/bin/env python3
"""Name the IRON fifo behind a dataflow error, instead of guessing which one.

    python utilities/why-no-fifo.py --spec open_kernels/recipes/specs/qwen35-h5120-L64.json \
        --design lx [--kernel-size]

WHY. IRON raises, from ObjectFifo.acquire:

    ValueError: Number of elements to acquire 3 must be smaller than depth 2

with no indication of WHICH fifo. A whole-layer design has a dozen, and
finding the one that fired means reading the traceback into the design and
then working out which acquire it was -- which is how you end up confidently
naming the wrong fifo, as this one did twice: `of_x` looked right, the
arithmetic even said 3 > 2, and yet the 9B has the same OG_ELEMS and builds.

So this patches the raise to include the fifo's name and the call site, as a
wrapper that imports IRON first and then rebinds one method. Nothing in the
installed tree is edited: the original is kept and restored, so a run that
dies partway leaves no trace behind.

Use --kernel-size to run the whole measurement afterwards, which reports the
per-TU `.text` aiecc throws away; the two together answer "which fifo" and
"does it still overflow" in one pass.
"""
from __future__ import annotations

import argparse
import importlib
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "open_kernels"

# The frame lines worth showing: a design or a helper, skipping iron's own
# plumbing, which is a fixed ladder that tells you nothing about the caller.
KEEP = ("open_kernels/", "utilities/")


def patch_acquire() -> tuple[object, str]:
    """Wrap ObjectFifo.acquire/release to name the fifo. Returns (original, class)."""
    mod = importlib.import_module("aie.iron.dataflow.objectfifo")
    # The acquire/release a design calls lives on the HANDLE (ObjectFifo is the
    # device-side object being declared), so patching ObjectFifo silently does
    # nothing -- which is what the first version did.
    cls = mod.ObjectFifoHandle
    orig_acquire, orig_release = cls.acquire, cls.release

    def _name(self) -> str:
        # IRON's wrapper has no name attribute of its own; the underlying
        # objectfifo does. Either is better than "unknown".
        for attr in ("_name", "name"):
            v = getattr(self, attr, None)
            if isinstance(v, str):
                return v
        inner = getattr(self, "_object_fifo", None)
        v = getattr(inner, "_name", None)
        return v if isinstance(v, str) else type(self).__name__

    def _site() -> str:
        out = []
        for fr in traceback.extract_stack()[:-2]:
            f = fr.filename.replace(str(REPO) + "/", "")
            if any(k in f for k in KEEP):
                out.append(f"      {f}:{fr.lineno}  {fr.line}")
        return "\n".join(reversed(out[-4:]))

    def acquire(self, num_elem: int):
        if self._depth < num_elem:
            raise ValueError(
                f"fifo {_name(self)!r}: cannot acquire {num_elem} with depth "
                f"{self._depth} ({self._depth} buffers allocated)\n{_site()}")
        return orig_acquire(self, num_elem)

    def release(self, num_elem: int):
        if self._depth < num_elem:
            raise ValueError(
                f"fifo {_name(self)!r}: cannot release {num_elem} with depth "
                f"{self._depth}\n{_site()}")
        return orig_release(self, num_elem)

    cls.acquire, cls.release = acquire, release
    return cls, "patched"


def restore(cls) -> None:
    """Put IRON back exactly as it was."""
    importlib.reload(sys.modules[cls.__module__])


def run_design(design: str, out: Path) -> int:
    """Specialize and compile a design, in THIS process, with the patch live.

    In-process on purpose: the patch rebinds a method on an imported class, and
    a child would not see it -- the first version of this ran the design as a
    subprocess, printed an UNPATCHED error, and looked like the patch had
    silently failed. Nothing in the installed tree is edited either way, since
    the rebinding is undone in main()'s finally.
    """
    os.environ.setdefault("MLIR_AIE_ROOT", str(REPO / "third_party" / "mlir-aie"))
    out.mkdir(parents=True, exist_ok=True)
    keep = out / "objects"
    keep.mkdir(parents=True, exist_ok=True)
    import shutil
    from aie.utils.compile import utils as au
    real = au.compile_external_kernel

    def keep_one(func, kernel_dir, target_arch):
        before = set(Path(kernel_dir).glob("*.o")) if Path(kernel_dir).is_dir() else set()
        real(func, kernel_dir, target_arch)
        for o in Path(kernel_dir).glob("*.o"):
            if o not in before:
                shutil.copy2(o, keep / o.name)
    au.compile_external_kernel = keep_one
    import aie.utils.compile.jit.compilabledesign as cd
    cd.compile_external_kernel = keep_one

    import shutil
    src = KERNELS / "designs" / design
    sys.argv = ["build_design.py", str(src), str(out)]
    cwd = os.getcwd()
    try:
        os.chdir(src.parent)
        runpy.run_path(str(KERNELS / "build_design.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"build exited {e.code}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - the error IS the measurement
        msg = str(e)
        if "expected output file" not in msg:
            print(f"{type(e).__name__}: {msg}", file=sys.stderr)
    finally:
        os.chdir(cwd)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", type=Path, required=True, help="a derived spec JSON")
    ap.add_argument("--design", default="layer_x/lx.py", help="design, relative to designs/")
    ap.add_argument("--out", type=Path, default=REPO / "build" / "why-no-fifo")
    ap.add_argument("--kernel-size", action="store_true",
                    help="afterwards, report per-TU .text via utilities/kernel-size.py")
    a = ap.parse_args()

    os.environ["OPEN_KERNELS_SPEC"] = str(a.spec.resolve())
    cls, _ = patch_acquire()
    try:
        rc = run_design(a.design, a.out)
    finally:
        restore(cls)

    if a.kernel_size:
        k = subprocess.run(
            [sys.executable, str(REPO / "utilities" / "kernel-size.py"),
             "--spec", str(a.spec), "--kernel", Path(a.design).stem],
            cwd=str(REPO))
        rc = rc or k.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
