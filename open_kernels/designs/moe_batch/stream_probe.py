r"""Time mb_s256 and its ablations against each other, with no model loaded: how
fast does the expert stream actually go, and what holds it there.

The question this answers: the weight stream reads a 128-row stripe as two bands
interleaved at k-tile granularity, so every band is a strided read of the pool.
That could be what keeps the kernel at ~31 GB/s when the same engine reads
contiguous q8 at 44. Build the variants (moe_batch.py, whose env flags they are)
and compare:

    build_s256_e256         the kernel as shipped
    build_s256_null         MB_NULL_MM=1: the streams without the core arithmetic
    build_s256_dq           MB_NULL_DQ=1: the product without the q4_1 scales
    build_s256_contig       MB_CONTIG=1: the same bytes read in order, not strided
    build_s256_contig_null  MB_CONTIG=1 MB_NULL_MM=1

The pool is random bytes and no reference is computed - every variant including
the shipped one produces garbage here, because only the wall time matters.

The box is not idle, so a variant measured in its own pass can lose several
percent to whatever else woke up. Run several passes with the build order
rotated and quote the minimum; contention only ever adds time.

    python stream_probe.py [--runs 8] [--passes 4] [--builds a,b,c]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent.parent / "harness" / "out" / "run_kernel.exe"

HID, FF, NT, SLOTS = 2048, 512, 8, 256
STRIPE = 128 * HID * 5 // 8
EXP_UG = 2 * (FF // 128) * STRIPE
DOWN_BYTES = HID * FF * 5 // 8
POOL_BYTES = 512 << 20
WEIGHT_BYTES = SLOTS * (EXP_UG + DOWN_BYTES)          # what one dispatch streams

DEFAULT = ["build_s256_e256", "build_s256_null", "build_s256_dq", "build_s256_contig", "build_s256_contig_null"]


def time_build(b: str, pool: Path, runs: int) -> list[float]:
    cfg = HERE / f"probe_{b}.cfg"
    lines = ["device", f"xclbin G {b}/final.xclbin", f"kernelx k G {b}/insts.bin",
             f"buf pool {POOL_BYTES} {pool}",
             f"buf x {SLOTS * HID * NT * 2}", f"buf h {SLOTS * FF * NT * 2}", f"buf y {SLOTS * HID * NT * 4}"]
    lines += ["run k pool x h y"] * (runs + 1)        # the first is the warm-up
    cfg.write_text("\n".join(lines) + "\n", newline="\n")
    r = subprocess.run([str(HARNESS), str(cfg)], capture_output=True, text=True)
    ms = [float(m) for m in re.findall(r"\(([0-9.]+) ms\)", r.stdout)][1:]
    if not ms or "DONE" not in r.stdout:
        print(f"{b}: FAILED\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        return []
    return ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--builds", default=",".join(DEFAULT))
    ap.add_argument("--pool", default=str(HERE / "pool_rand.bin"))
    a = ap.parse_args()

    pool = Path(a.pool)
    if not pool.is_file() or pool.stat().st_size != POOL_BYTES:
        print(f"writing {POOL_BYTES} B of random pool -> {pool}")
        pool.write_bytes(np.random.default_rng(0).integers(0, 256, POOL_BYTES, dtype=np.uint8).tobytes())

    order = [b for b in a.builds.split(",") if (HERE / b / "final.xclbin").is_file()]
    out: dict[str, list[float]] = {}
    for ps in range(a.passes):
        k = ps % len(order)
        for b in order[k:] + order[:k]:
            ms = time_build(b, pool, a.runs)
            if ms:
                out.setdefault(b, []).extend(ms)
                print(f"  pass {ps} {b:24s} min {min(ms):7.2f} median {float(np.median(ms)):7.2f} max {max(ms):7.2f}")

    print()
    for b in order:
        v = np.array(out.get(b, []))
        if not len(v):
            continue
        lo = float(v.min())
        print(f"{b:24s} min {lo:7.2f} ms  p25 {np.percentile(v, 25):7.2f}  median {np.median(v):7.2f}  "
              f"n={len(v)}  -> {WEIGHT_BYTES / (lo * 1e-3) / 1e9:5.1f} GB/s of weight at the min")
    if "build_s256_e256" in out:
        base = float(np.min(out["build_s256_e256"]))
        for b in order:
            if b != "build_s256_e256" and b in out:
                print(f"{b:24s} {base / float(np.min(out[b])):.3f}x the shipped kernel (min vs min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
