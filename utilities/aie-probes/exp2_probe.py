r"""exp2_probe: dump the aie2p hardware exp2 (aie::exp2<bfloat16> on an fp32 input).

    python open_kernels\build_design.py utilities\aie-probes\exp2_probe.py <out>
    python utilities\aie-probes\exp2_probe.py --run <out>      # writes <out>\exp2_table.npz

One core, one shot: N fp32 inputs in, N bf16 results out. --run makes the inputs (a
dense grid on [-40, 0.5]), runs them through open_kernels/harness's run_kernel and
prints the relative error against the exactly rounded bf16 result. The last 16 inputs
are extremes (large negative, -inf, NaN, large positive).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

N = 16384
E = 2048   # per fifo element; the core loops N // E times
HERE = Path(__file__).resolve().parent


def design():
    import aie.iron as iron
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
    from ml_dtypes import bfloat16

    @iron.jit
    def exp2_probe(X: In, Y: Out, *, n: CompileTime[int] = N):
        x_ty = np.ndarray[(E,), np.dtype[np.float32]]
        y_ty = np.ndarray[(E,), np.dtype[bfloat16]]
        k = ExternalFunction("exp2_hw", source_file=str(HERE / "exp2_probe.cc"),
                             arg_types=[x_ty, y_ty, np.int32])
        fx = ObjectFifo(x_ty, name="X", depth=1)
        fy = ObjectFifo(y_ty, name="Y", depth=1)

        def core(xi, yo, kern):
            for _ in range(N // E):
                x = xi.acquire(1)
                y = yo.acquire(1)
                kern(x, y, E)
                xi.release(1)
                yo.release(1)

        w = Worker(core, [fx.cons(), fy.prod(), k])
        rt = Runtime(lambda a, b, px, cy: (px.fill(a), cy.drain(b, wait=True)),
                     [np.ndarray[(N,), np.dtype[np.float32]], np.ndarray[(N,), np.dtype[bfloat16]],
                      fx.prod(), fy.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    return exp2_probe


if __name__ != "__main__":
    DESIGN = design()
    SPECIALIZE = {"n": N}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="build dir (final.xclbin, insts.bin)")
    a = ap.parse_args()
    from ml_dtypes import bfloat16
    out = Path(a.run).resolve()
    x = np.linspace(-40.0, 10.0, N, dtype=np.float32)
    x[-16:] = [-50, -100, -126, -127, -128, -130, -1e4, -4.3e37, -np.inf, np.nan,
               16, 64, 127, 128, 200, np.inf]
    x.tofile(out / "x.bin")
    cfg = ["device", f"xclbin G {out / 'final.xclbin'}", f"kernelx k G {out / 'insts.bin'}",
           f"buf x {N * 4} x.bin", f"buf y {N * 2}", "run k x y", f"dump y y.bin {N * 2}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    exe = HERE.parents[1] / "open_kernels" / "harness" / "out" / "run_kernel.exe"
    subprocess.run([str(exe), "run.cfg"], cwd=out, check=True)
    y = np.fromfile(out / "y.bin", dtype=bfloat16).astype(np.float64)
    exact = np.exp2(x.astype(np.float64))
    rounded = exact.astype(bfloat16).astype(np.float64)
    rel = (y - exact) / exact
    np.savez(out / "exp2_table.npz", x=x, y=y)
    for xv, yv in zip(x[-16:], y[-16:]):
        print(f"  exp2({xv:g}) -> {yv:g}")
    x, y, exact, rounded, rel = x[:-16], y[:-16], exact[:-16], rounded[:-16], rel[:-16]
    for lo, hi in ((-1, 0), (-4, -1), (-10, -4), (-40, -10), (0, 1), (1, 4), (4, 10)):
        m = (x >= lo) & (x < hi)
        print(f"x in [{lo:4d},{hi:4d}): rel err mean {rel[m].mean():+.2e}  rms {np.sqrt((rel[m]**2).mean()):.2e}"
              f"  max {np.abs(rel[m]).max():.2e}   (exact bf16 rounding rms "
              f"{np.sqrt((((rounded - exact) / exact)[m] ** 2).mean()):.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
