"""Run a .cfg through one or more drivers and summarise the per-run kernel times.

    python bench.py <run.cfg> [--driver run_kernel.exe] [--driver open-qwen-npu.exe npu] [--warm 1]

Each driver prints `run <kernel> [<n> bufs] -> state <st> (<ms> ms)` per run;
this parses those lines, drops the first `--warm` runs (cold context / first
touch), and prints min / median / max per kernel per driver. Drivers run
sequentially with the cfg's directory as cwd so relative paths resolve the same
way for both.
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

RUN = re.compile(r"^run (\S+) \[\d+ bufs\] -> state (\d+) \((\d+\.\d+) ms\)")


def run_driver(cmd: list[str], cfg: Path) -> dict[str, list[float]]:
    # The driver runs with the cfg's directory as cwd, so its own path must be absolute.
    exe = Path(cmd[0])
    cmd = [str(exe.resolve()) if exe.exists() else cmd[0]] + cmd[1:]
    p = subprocess.run(cmd + [cfg.name], cwd=cfg.parent, capture_output=True, text=True)
    times: dict[str, list[float]] = {}
    for line in p.stdout.splitlines():
        m = RUN.match(line)
        if m:
            k, st, ms = m.group(1), int(m.group(2)), float(m.group(3))
            if st != 4:
                print(f"  !! {line}")
            times.setdefault(k, []).append(ms)
    if p.returncode != 0:
        print(f"  driver exit {p.returncode}\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--driver", action="append", nargs="+", metavar="ARG",
                    help="driver command (repeatable); default: harness/out/run_kernel(.exe)")
    ap.add_argument("--warm", type=int, default=1, help="runs to drop as warm-up")
    a = ap.parse_args()
    cfg = Path(a.cfg).resolve()
    drivers = a.driver or [[str(Path(__file__).resolve().parent / "out" / "run_kernel")]]
    for d in drivers:
        name = Path(d[0]).stem + ("" if len(d) == 1 else " " + " ".join(d[1:]))
        times = run_driver(d, cfg)
        for k, ts in times.items():
            hot = ts[a.warm:] or ts
            print(f"{name:28s} {k:6s} runs={len(ts):3d} cold={ts[0]:8.3f}  "
                  f"min={min(hot):8.3f} med={statistics.median(hot):8.3f} max={max(hot):8.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
