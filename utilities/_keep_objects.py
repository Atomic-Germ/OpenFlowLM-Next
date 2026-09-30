#!/usr/bin/env python3
"""Run build_design.py and keep the kernel objects aiecc would delete.

    python utilities/_keep_objects.py <design.py> <out_dir> <keep_dir>

aiecc wipes its project directory on the way out -- including the per-core
objects that show which core exceeded program memory, which is the one thing
worth reading when a build fails with "Overflow of program memory".

This wraps IRON's compile_external_kernel so each object is copied to
<keep_dir> the moment it is compiled (i.e. before aiecc runs), then lets the
build proceed. aiecc is left alone: if the design genuinely overflows the build
fails as usual, but the objects that explain why are already saved.

utilities/kernel-size.py drives this; it is not a build path.
"""
import os
import runpy
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "open_kernels"


def main() -> int:
    design, out, keep = (Path(a).resolve() for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    keep.mkdir(parents=True, exist_ok=True)

    import aie.utils.compile.utils as au

    real = au.compile_external_kernel

    def keep_one(func, kernel_dir, target_arch):
        before = set(Path(kernel_dir).glob("*.o")) if Path(kernel_dir).is_dir() else set()
        real(func, kernel_dir, target_arch)
        # The object is named func.object_file_name; copy whatever appeared.
        for o in Path(kernel_dir).glob("*.o"):
            if o not in before:
                shutil.copy2(o, keep / o.name)

    au.compile_external_kernel = keep_one
    # compilabledesign imported the symbol directly, so patch it there too.
    import aie.utils.compile.jit.compilabledesign as cd
    cd.compile_external_kernel = keep_one

    sys.argv = ["build_design.py", str(design), str(out)]
    os.chdir(design.parent)
    try:
        runpy.run_path(str(KERNELS / "build_design.py"), run_name="__main__")
    except SystemExit as e:
        print(f"exit: {e.code}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - a failed build is a valid outcome here
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
    n = len(list(keep.glob("*.o")))
    print(f"kept {n} objects in {keep}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
