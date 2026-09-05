r"""Build an IRON design into xclbin + insts.bin + insts.elf (no NPU needed).

Runs in the WSL `ironenv` (mlir-aie wheel + Peano); the artifacts are then
loaded on Windows by phlegm's XRT shim exactly like FLM's kernels are:
xrt::elf(insts.elf) -> xrt::module -> xrt::ext::kernel(ctx, "MLIR_AIE"),
run(opcode=3, 0, 0, bo...).

    source ~/ironenv/bin/activate
    python build_design.py designs/rot13/rot13.py [out_dir]

The design module must expose DESIGN (an @iron.jit callable) and SPECIALIZE
(dict of CompileTime kwargs).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from pathlib import Path

import aie.iron as iron
from aie.iron.device import from_name


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.parent / "build"
    # Always a fresh project: aiecc keeps copies of the kernel sources and their
    # objects in final.prj and skips recompiling them (header edits in the
    # design dir were silently ignored: identical results build after build).
    shutil.rmtree(out / "final.prj", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # Trap 1 (LLMNpuTest): without this IRON silently targets aie2 / NPU1.
    iron.set_current_device(from_name("npu2", n_cols=None))

    spec = importlib.util.spec_from_file_location(src.stem, src)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(src.parent))
    spec.loader.exec_module(mod)

    t0 = time.time()
    xclbin, insts = mod.DESIGN.specialize(**mod.SPECIALIZE).compile(
        xclbin_path=str(out / "final.xclbin"),
        inst_path=str(out / "insts.bin"),
        elf_path=str(out / "insts.elf"),
    )
    dt = time.time() - t0
    for p in sorted(out.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size:>10d} B")
    print(f"BUILD_OK {src.name} -> {out}  ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
