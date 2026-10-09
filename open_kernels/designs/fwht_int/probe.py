r"""Check the fwht_int builds bit for bit against fwht_ref.py and time them (run make_test.py first).

    python probe.py [--rounds 5] [--kinds 0,1,2,3] [--driver ...\run_kernel.exe]
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))
from bench import RUN  # noqa: E402
from fwht_ref import BLOCK, errors  # noqa: E402

DEFAULT_DRIVER = Path(r"C:\code\openflowlm-next\open_kernels\harness\out\run_kernel.exe")
NAMES = {0: ("int", "bf16"), 1: ("int", "f32"), 2: ("fp", "bf16"), 3: ("fp", "f32")}


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
    return ts[1:] or ts


def parse(b: bytes, k: int):
    ng = k // 128
    xi = np.frombuffer(b, np.int16, k)
    sh = np.frombuffer(b, np.int32, ng, 2 * k)
    hi = np.frombuffer(b, bfloat16, ng, 2 * k + 4 * ng)
    lo = np.frombuffer(b, bfloat16, ng, 2 * k + 6 * ng)
    return xi, sh, hi, lo


def check(kind: int, k: int, sx: str = "") -> str:
    path, vt = NAMES[kind]
    got_b = (HERE / f"t_k{kind}{sx}_r1.bin").read_bytes()
    exp_b = (HERE / f"tab_{path}_{vt}.bin").read_bytes()
    x = np.fromfile(HERE / f"x_{vt}.bin", bfloat16 if vt == "bf16" else np.float32)
    got, exp = parse(got_b, k), parse(exp_b, k)
    same = [int((g.view(np.uint8) == e.view(np.uint8)).all()) for g, e in zip(got, exp)]
    nxi = int((got[0] != exp[0]).sum())
    e = errors(x, got)
    per = []
    for b in range(k // BLOCK):
        s = slice(b * BLOCK, (b + 1) * BLOCK)
        g8 = slice(b * 8, (b + 1) * 8)
        eb = errors(x[s], tuple(t[s] if i == 0 else t[g8] for i, t in enumerate(got)))
        per.append(f"{eb['rms']:.1e}")
    return (f"bit-exact xi/sh/hi/lo={same} ({nxi} xi differ)  rel rms {e['rms']:.2e}  max/max {e['max']:.2e}\n"
            f"      per-block rel rms: {' '.join(per)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--kinds", default="0,1,2,3")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--suffix", default="", help="build-tag suffix, e.g. s for the -Os builds")
    a = ap.parse_args()
    kinds = [int(s) for s in a.kinds.split(",")]
    sx = a.suffix
    tags = [f"k{k}{sx}_r{r}" for k in kinds for r in ((1,) if a.check_only else (1, 33))]
    best: dict[str, list[float]] = {t: [] for t in tags}
    for rnd in range(1 if a.check_only else a.rounds):
        line = []
        for t in tags:
            ts = run(a.driver, HERE / f"run_{t}.cfg")
            if ts:
                best[t].append(min(ts))
                line.append(f"{t} {min(ts):.3f}")
        print(f"round {rnd}: " + "  ".join(line), flush=True)
    nblk = a.k // BLOCK
    for k in kinds:
        path, vt = NAMES[k]
        msg = f"kind {k} ({path} {vt})"
        if not a.check_only and best[f"k{k}{sx}_r1"] and best[f"k{k}{sx}_r33"]:
            t1, t33 = min(best[f"k{k}{sx}_r1"]), min(best[f"k{k}{sx}_r33"])
            msg += f": r1 {t1:.3f} ms  r33 {t33:.3f} ms  -> {(t33 - t1) * 1e3 / (32 * nblk):.3f} us per block"
        print(msg)
        print("   ", check(k, a.k, sx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
