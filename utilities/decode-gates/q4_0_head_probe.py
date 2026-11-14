r"""Bytes probe: one random Q4_0 head packed as today's 5120 B chunks (m = -8 d) and as 4608 B Q4_0-native chunks,
with run_kernel cfgs for lm_head_q4 and its LMHEAD_Q4_0=1 build, then a correctness compare or an alternated timing.

    python q4_0_head_probe.py make    --out <dir> --q41 <lm_head_q4 dir> --q40 <LMHEAD_Q4_0 build dir> [--n 100352 --k 2560]
    python q4_0_head_probe.py compare --out <dir> --run-kernel <run_kernel.exe>
    python q4_0_head_probe.py bench   --out <dir> --run-kernel <run_kernel.exe> [--reps 20 --rounds 4]
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "open_kernels"))
from q4_1_pack import pack_q4_0_native_pool, pack_q4_1_pool, pool_reference, random_q4_0_blocks  # noqa: E402

NREF = 2048


def make(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    blocks = random_q4_0_blocks(a.n, a.k, rng)
    p41, p40 = pack_q4_1_pool(blocks, 2), pack_q4_0_native_pool(blocks, 2)
    x = (rng.standard_normal(a.k).astype(np.float32) * 0.5).astype(bfloat16)
    p41.tofile(out / "w_q41.bin")
    p40.tofile(out / "w_q40.bin")
    x.tofile(out / "x.bin")
    pool_reference(p41[:NREF * a.k // 8192 * 5120], x.astype(np.float32), NREF, a.k, 2).tofile(out / "ref.bin")
    for tag, d, w, n in (("q41", Path(a.q41), "w_q41.bin", p41.size), ("q40", Path(a.q40), "w_q40.bin", p40.size)):
        (out / f"run_{tag}.cfg").write_text(
            f"device\nxclbin G {d / 'final.xclbin'}\nkernelx k G {d / 'insts.bin'}\n"
            f"buf w {n} {w}\nbuf x {a.k * 2} x.bin\nbuf y {a.n * 4}\nrun k w x y\ndump y y_{tag}.bin {a.n * 4}\n")
    print("pools", p41.size, p40.size, "B; ratio", round(p40.size / p41.size, 4))
    return 0


def compare(a) -> int:
    out = Path(a.out)
    for tag in ("q41", "q40"):
        r = subprocess.run([a.run_kernel, str(out / f"run_{tag}.cfg")], cwd=out, capture_output=True, text=True)
        print(tag, "exit", r.returncode, flush=True)
    ref = np.fromfile(out / "ref.bin", np.float32).astype(np.float64)
    y41, y40 = (np.fromfile(out / f"y_{t}.bin", np.float32) for t in ("q41", "q40"))
    for name, y in (("q4_1 layout", y41), ("q4_0 native", y40)):
        v = y[:NREF].astype(np.float64)
        print(f"{name}: first {NREF} rows vs fp64 cos {v @ ref / np.linalg.norm(v) / np.linalg.norm(ref):.8f}, "
              f"maxrel {np.abs(v - ref).max() / np.abs(ref).max():.2e}")
    same = np.array_equal(y41.view(np.uint32), y40.view(np.uint32))
    print("q4_0 native vs q4_1 layout, every row:", "BIT-IDENTICAL" if same else f"{np.count_nonzero(y41 != y40)} differ")
    return 0 if same else 1


def bench(a) -> int:
    out = Path(a.out)
    for tag in ("q41", "q40"):
        lines = (out / f"run_{tag}.cfg").read_text().splitlines()
        run = next(l for l in lines if l.startswith("run "))
        (out / f"bench_{tag}.cfg").write_text("\n".join([l for l in lines if not l.startswith(("run ", "dump "))]
                                                        + [run] * (a.reps + 1)) + "\n")
    times = {"q41": [], "q40": []}
    for rnd in range(a.rounds):
        for tag in (("q41", "q40") if rnd % 2 == 0 else ("q40", "q41")):
            r = subprocess.run([a.run_kernel, str(out / f"bench_{tag}.cfg")], cwd=out, capture_output=True, text=True)
            ms = [float(m) for m in re.findall(r"\(([\d.]+) ms\)", r.stdout + r.stderr)][1:]   # the first run warms
            times[tag] += ms
            print(f"round {rnd} {tag}: min {min(ms):.3f} median {statistics.median(ms):.3f} ms ({len(ms)} runs)", flush=True)
    for tag in ("q41", "q40"):
        b = (out / f"w_{tag}.bin").stat().st_size
        v = times[tag]
        print(f"SUMMARY {tag}: min {min(v):.3f} median {statistics.median(v):.3f} ms; {b / 1e6:.1f} MB, "
              f"{b / min(v) / 1e6:.1f} GB/s at the min")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["make", "compare", "bench"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--q41")
    ap.add_argument("--q40")
    ap.add_argument("--run-kernel")
    ap.add_argument("--n", type=int, default=100352)
    ap.add_argument("--k", type=int, default=2560)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    return {"make": make, "compare": compare, "bench": bench}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
