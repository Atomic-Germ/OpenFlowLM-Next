r"""vecmath_probe: dump aie2p vector tanh (fp32 -> bf16) and bf16 inv / invsqrt.

    python open_kernels\build_design.py utilities\aie-probes\vecmath_probe.py <out>
    python utilities\aie-probes\vecmath_probe.py --run <out>   # writes <out>\vecmath_table.npz

One core: N fp32 inputs in, 3N bf16 results out (tanh(x); inv and invsqrt of
bf16(|x| + 0.25)). --run makes a grid on [-10, 10], runs it through run_kernel and prints
each function's error against the exactly rounded bf16 result.
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
    def vecmath_probe(X: In, Y: Out, *, n: CompileTime[int] = N):
        x_ty = np.ndarray[(E,), np.dtype[np.float32]]
        y_ty = np.ndarray[(3 * E,), np.dtype[bfloat16]]
        k = ExternalFunction("vecmath_hw", source_file=str(HERE / "vecmath_probe.cc"),
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
                     [np.ndarray[(N,), np.dtype[np.float32]],
                      np.ndarray[(3 * N,), np.dtype[bfloat16]], fx.prod(), fy.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    return vecmath_probe


if __name__ != "__main__":
    DESIGN = design()
    SPECIALIZE = {"n": N}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="build dir (final.xclbin, insts.bin)")
    a = ap.parse_args()
    from ml_dtypes import bfloat16
    out = Path(a.run).resolve()
    x = np.linspace(-10.0, 10.0, N, dtype=np.float32)
    x.tofile(out / "x.bin")
    cfg = ["device", f"xclbin G {out / 'final.xclbin'}", f"kernelx k G {out / 'insts.bin'}",
           f"buf x {N * 4} x.bin", f"buf y {3 * N * 2}", "run k x y",
           f"dump y y.bin {3 * N * 2}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    exe = HERE.parents[1] / "open_kernels" / "harness" / "out" / "run_kernel.exe"
    subprocess.run([str(exe), "run.cfg"], cwd=out, check=True)
    # Each E-sized element holds [tanh | inv | invsqrt] for its E inputs.
    y = np.fromfile(out / "y.bin", dtype=bfloat16).astype(np.float64)
    y = y.reshape(N // E, 3, E).transpose(1, 0, 2).reshape(3, N)
    b = (np.abs(x) + 0.25).astype(bfloat16).astype(np.float64)
    refs = {"tanh": np.tanh(x.astype(np.float64)), "inv": 1 / b, "invsqrt": 1 / np.sqrt(b)}
    np.savez(out / "vecmath_table.npz", x=x, y=y)
    for k, (name, ref) in enumerate(refs.items()):
        got = y[k]
        rnd = ref.astype(np.float32).astype(bfloat16).astype(np.float64)
        den = np.where(ref == 0, 1, np.abs(ref))
        rel, rr = (got - ref) / den, (rnd - ref) / den
        for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, 10)):
            m = (np.abs(x) >= lo) & (np.abs(x) < hi)
            print(f"{name:8s} |x| in [{lo},{hi}): rel err mean {rel[m].mean():+.2e} rms "
                  f"{np.sqrt((rel[m] ** 2).mean()):.2e} max {np.abs(rel[m]).max():.2e}"
                  f"   (bf16 rounding rms {np.sqrt((rr[m] ** 2).mean()):.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
