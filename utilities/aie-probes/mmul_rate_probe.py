r"""mmul_rate_probe: how fast one AIE2P core issues mmuls of each operand type, operands resident in L1.

    $env:MM_VARIANT = 0|1|2|3|4; $env:MM_ITERS = 1000000
    python open_kernels\build_design.py utilities\aie-probes\mmul_rate_probe.py <out>
    python utilities\aie-probes\mmul_rate_probe.py --run <out>      # wall ms per dispatch, five runs

VARIANT: 0 bf16 4x8x8 (native, what mm.cc runs), 1 int8 x int8 8x8x8, 2 bfp16 x bfp16 8x8x8, 3 int8 x uint4 8x8x8,
4 bf16 8x8x8 with AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16. Each iteration is one 2x2 expansion: four mmuls on four loads.
Time two ITERS and difference them; the dispatch overhead cancels.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VARIANT = int(os.environ.get("MM_VARIANT", 0))
ITERS = int(os.environ.get("MM_ITERS", 1000000))


def design():
    import aie.iron as iron
    from aie.iron import CompileTime, ExternalFunction, ObjectFifo, Out, Program, Runtime, Worker

    flags = [f"-DVARIANT={VARIANT}"]
    if VARIANT == 4:
        flags.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")

    @iron.jit
    def mmul_rate(Y: Out, *, iters: CompileTime[int] = ITERS):
        y_ty = np.ndarray[(256,), np.dtype[np.int32]]
        k = ExternalFunction("mmul_rate", source_file=str(HERE / "mmul_rate_probe.cc"),
                             arg_types=[y_ty, np.int32], compile_flags=flags)
        fy = ObjectFifo(y_ty, name="Y", depth=1)

        def core(yo, kern):
            y = yo.acquire(1)
            kern(y, iters)
            yo.release(1)

        w = Worker(core, [fy.prod(), k])
        rt = Runtime(lambda b, cy: cy.drain(b, wait=True), [y_ty, fy.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    return mmul_rate


if __name__ != "__main__":
    DESIGN = design()
    SPECIALIZE = {"iters": ITERS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.run).resolve()
    cfg = ["device", f"xclbin G {out / 'final.xclbin'}", f"kernelx k G {out / 'insts.bin'}", "buf y 1024"]
    cfg += ["run k y"] * a.repeat + [""]
    (out / "run.cfg").write_text("\n".join(cfg))
    exe = HERE.parents[1] / "open_kernels" / "harness" / "out" / "run_kernel.exe"
    r = subprocess.run([str(exe), "run.cfg"], cwd=out, capture_output=True, text=True)
    print(r.stdout + r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
