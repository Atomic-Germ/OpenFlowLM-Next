r"""probe: how much faster dit_gemm runs when A arrives as bfp16ebs8 (a timing probe).

For each shape, builds open_kernels/designs/dit_gemm/dit_gemm.py (A bf16, converted in
the core) and dit_gemm_bfp16a.py (A pre-converted), feeds both random bytes and times
them with open_kernels/harness/bench.py. The outputs are not checked: the probe's A
arrangement is not a real one. Run it on a quiet machine, in turbo.

    . C:\dev\mlir-aie\iron_env.ps1
    xrt-smi configure --pmode turbo
    python utilities\dit-gemm-bench\bfp16a\probe.py --work C:\dev\gemm-probe [--shapes 4608x3072x27648,4608x12288x3072] [--runs 20]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = ROOT / "open_kernels" / "build_design.py"
BENCH = ROOT / "open_kernels" / "harness" / "bench.py"
DRIVER = ROOT / "open_kernels" / "harness" / "out" / "run_kernel.exe"
DESIGNS = {"bf16 A": ROOT / "open_kernels" / "designs" / "dit_gemm" / "dit_gemm.py",
           "bfp16 A": HERE / "dit_gemm_bfp16a.py"}


def build(design: Path, out: Path, M: int, K: int, N: int) -> None:
    if (out / "final.xclbin").is_file() and (out / "insts.bin").is_file():
        return
    env = dict(os.environ, DG_M=str(M), DG_K=str(K), DG_N=str(N), DG_LAYOUT="{}")
    r = subprocess.run([sys.executable, str(BUILD), str(design), str(out)], env=env,
                       cwd=str(BUILD.parent), capture_output=True, text=True)
    if "BUILD_OK" not in r.stdout:
        sys.stdout.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"build of {design.name} {M}x{K}x{N} failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--shapes", default="4608x3072x27648,4608x12288x3072")
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()
    work = Path(a.work)
    rng = np.random.default_rng(0)
    rows = []
    for shape in a.shapes.split(","):
        M, K, N = map(int, shape.split("x"))
        b_bytes = K * N * 9 // 8
        res = {}
        for tag, design in DESIGNS.items():
            d = work / f"{tag.split()[0]}_{shape}"
            build(design, d, M, K, N)
            a_bytes = M * K * 2 if tag == "bf16 A" else M * K * 9 // 8
            for name, nb in (("a.bin", a_bytes), ("b.bin", b_bytes)):
                if not (d / name).is_file():
                    x = rng.integers(0, 256, nb, dtype=np.uint8)
                    x[1::2] &= 0x3F              # keep bf16s (and bfp exponents) finite
                    x.tofile(d / name)
            cfg = ["device", f"xclbin G {d / 'final.xclbin'}", f"kernelx k G {d / 'insts.bin'}",
                   f"buf a {a_bytes} a.bin", f"buf b {b_bytes} b.bin", f"buf c {M * N * 2}"]
            cfg += ["run k a b c"] * a.runs + [""]
            (d / "run.cfg").write_text("\n".join(cfg))
            r = subprocess.run([sys.executable, str(BENCH), str(d / "run.cfg"), "--driver",
                                str(DRIVER), "--warm", "1"], capture_output=True, text=True)
            m = re.search(r"med=\s*([\d.]+)", r.stdout)
            if not m or "!!" in r.stdout or "driver exit" in r.stdout or r.returncode:
                sys.stdout.write(r.stdout[-2000:] + r.stderr[-2000:])
                raise SystemExit(f"bench of {d} failed")
            res[tag] = float(m.group(1))
        flop = 2 * M * K * N
        base, probe = res["bf16 A"], res["bfp16 A"]
        rows.append((shape, base, flop / base / 1e9, probe, flop / probe / 1e9, 1 - probe / base))
        print(f"{shape:18s} bf16 A {base:7.2f} ms ({flop / base / 1e9:4.1f} TFLOPS)   "
              f"bfp16 A {probe:7.2f} ms ({flop / probe / 1e9:4.1f} TFLOPS)   "
              f"{100 * (1 - probe / base):+5.1f}% time saved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
