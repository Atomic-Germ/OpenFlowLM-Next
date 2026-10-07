r"""Time and check probe 0a's four builds of one shape on the NPU (Windows).

    python probe.py gate [--rounds 5] [--driver ..\..\harness\out\run_kernel.exe]

Each round runs q4, q4 null, t2, t2 null back to back, so machine drift hits all
four alike (the peano-kernel-headroom lesson: the box can slow 2x mid-run). For
each build: the min over rounds of the per-round minimum, the weight bytes over
that time (GB/s) and the weights per second. Then each real build's y against the
fp64 reference (the null builds must give zeros).
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


def check(tag: str, ref: np.ndarray) -> str:
    if tag.startswith("t2h") and not tag.endswith("_null"):
        ref = np.fromfile(HERE / f"ref_{tag.split('_')[1]}_h.bin", np.float32)
    got = np.fromfile(HERE / f"y_{tag}.bin", np.float32).astype(np.float64)
    if tag.endswith("_null"):
        return f"zeros={not got.any()}"
    r = ref.astype(np.float64)
    rel = np.abs(got - r).max() / (np.abs(r).max() + 1e-30)
    cos = float(got @ r / (np.linalg.norm(got) * np.linalg.norm(r) + 1e-30))
    ok = cos > 0.9999999 and rel < 1e-4 and np.isfinite(got).all()
    return f"{'PASS' if ok else 'FAIL'} cos={cos:.9f} maxrel={rel:.2e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shape")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    ap.add_argument("--fmts", default="q4,t2", help="comma list of q4 / t2 / t2g")
    a = ap.parse_args()
    tags = [f"{f}_{a.shape}{n}" for f in a.fmts.split(",") for n in ("", "_null")]
    times: dict[str, list[float]] = {t: [] for t in tags}
    meds: dict[str, list[float]] = {t: [] for t in tags}
    for r in range(a.rounds):
        line = []
        for t in tags:
            ts = run(a.driver, HERE / f"run_{t}.cfg")
            if ts:
                times[t].append(min(ts))
                meds[t].append(statistics.median(ts))
                line.append(f"{t} {min(ts):.3f}")
        print(f"round {r}: " + "  ".join(line))
    ref = np.fromfile(HERE / f"ref_{a.shape}.bin", np.float32)
    n = ref.size
    k = (HERE / f"x_{a.shape}.bin").stat().st_size // 2
    print(f"\n{a.shape}: N={n} K={k}  ({n * k / 1e6:.1f} M weights)")
    for t in tags:
        if not times[t]:
            print(f"{t:16s} no runs")
            continue
        wbytes = (HERE / f"w_{'q4' if t.startswith('q4') else 't2'}_{a.shape}.bin").stat().st_size
        best = min(times[t])
        print(f"{t:16s} min {best:7.3f} ms  med {statistics.median(meds[t]):7.3f} ms  "
              f"{wbytes / best / 1e6:6.1f} GB/s  {n * k / best / 1e6:7.1f} Gweights/s  {check(t, ref)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
