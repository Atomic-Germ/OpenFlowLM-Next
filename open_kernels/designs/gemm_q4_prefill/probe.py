r"""Time and check gemm_q4_prefill builds of one shape on the NPU (Windows), interleaved.

    python probe.py n4096_k5120 base bfp base_nd bfp_nd bfp_nm nm_nd [--rounds 5] [--data tern]

Each variant is a build dir `probe_<variant>_<shape>` (see the GQP_* flags in
gemm_q4_prefill.py: bfp = GQP_BFP16, *_nd = GQP_NULL_DEQUANT, *_nm = GQP_NULL_MM). Every
round runs every variant back to back so machine drift hits them alike. For each: the min
over rounds of the per-round minimum, TFLOPS at that time, and y against the fp64 reference
(ablated builds give garbage by design, so only full builds are checked).
`--data tern` uses make_test.py --ternary's vectors (Bonsai-shaped weights), `--data q4` the
plain random q4_1 ones.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))
from bench import RUN  # noqa: E402

DEFAULT_DRIVER = Path(r"C:\code\openflowlm-next\open_kernels\harness\out\run_kernel.exe")
T = 256


def write_cfg(variant: str, shape: str, data: str, runs: int) -> Path:
    name = shape + ("_tern" if data == "tern" else "")
    w = HERE / (f"w_{name}_t2.bin" if "t2" in variant else f"w_{name}.bin")
    x = HERE / (f"x_{name}_t{T}_k128.bin" if "t2k" in variant else f"x_{name}_t{T}.bin")
    n, _k = (int(p[1:]) for p in shape.split("_"))
    ybytes = n * T * 4
    b = f"probe_{variant}_{shape}"
    cfg = ["device", f"xclbin G {b}/final.xclbin", f"kernelx k G {b}/insts.bin",
           f"buf w {w.stat().st_size} {w.name}", f"buf x {x.stat().st_size} {x.name}", f"buf y {ybytes}"]
    cfg += ["run k w x y"] * runs
    cfg += [f"dump y y_probe_{variant}_{name}.bin {ybytes}", ""]
    p = HERE / f"run_probe_{variant}_{name}.cfg"
    p.write_text("\n".join(cfg), newline="\n")
    return p


def run(driver: Path, cfg: Path) -> list[float]:
    p = subprocess.run([str(driver), cfg.name], cwd=HERE, capture_output=True, text=True)
    ts = []
    for line in p.stdout.splitlines():
        m = RUN.match(line)
        if m:
            if int(m.group(2)) != 4:
                print(f"  !! {cfg.name}: {line}")
            ts.append(float(m.group(3)))
    if p.returncode != 0 or not ts:
        print(f"  driver exit {p.returncode} on {cfg.name}\n{p.stdout[-600:]}\n{p.stderr[-600:]}")
    return ts[1:] or ts          # drop the cold run


def check(variant: str, shape: str, data: str) -> str:
    if "_n" in variant:
        return "(ablation)"
    name = shape + ("_tern" if data == "tern" else "")
    got = np.fromfile(HERE / f"y_probe_{variant}_{name}.bin", np.float32).astype(np.float64)
    ref = np.fromfile(HERE / f"ref_{name}_t{T}.bin", np.float32).astype(np.float64)
    if "yt" in variant:                     # GQP_YT builds write y as [T, N]
        got = got.reshape(T, -1).T.ravel()
    rel_fro = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
    maxrel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
    n = ref.size // T
    G, R = got.reshape(n, T), ref.reshape(n, T)
    cos_tok = np.einsum("it,it->t", G, R) / (np.linalg.norm(G, axis=0) * np.linalg.norm(R, axis=0) + 1e-30)
    return f"rel_fro={rel_fro:.3e} maxrel={maxrel:.3e} min tok cos={cos_tok.min():.9f} finite={np.isfinite(got).all()}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shape")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--data", default="tern", choices=("tern", "q4"))
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    a = ap.parse_args()
    cfgs = {v: write_cfg(v, a.shape, a.data, a.runs) for v in a.variants}
    times: dict[str, list[float]] = {v: [] for v in a.variants}
    meds: dict[str, list[float]] = {v: [] for v in a.variants}
    for r in range(a.rounds):
        line = []
        for v in a.variants:
            ts = run(a.driver, cfgs[v])
            if ts:
                times[v].append(min(ts))
                meds[v].append(statistics.median(ts))
                line.append(f"{v} {min(ts):.3f}")
        print(f"round {r}: " + "  ".join(line), flush=True)
    n, k = (int(p[1:]) for p in a.shape.split("_"))
    flop = 2.0 * n * k * T
    print(f"\n{a.shape} T={T} data={a.data}: {flop / 1e9:.2f} GFLOP per dispatch")
    for v in a.variants:
        if not times[v]:
            print(f"{v:10s} no runs")
            continue
        best = min(times[v])
        print(f"{v:10s} min {best:7.3f} ms  med {statistics.median(meds[v]):7.3f} ms  "
              f"{flop / best / 1e9:6.2f} TFLOPS  {check(v, a.shape, a.data)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
