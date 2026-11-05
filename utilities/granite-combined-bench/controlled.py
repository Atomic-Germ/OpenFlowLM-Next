r"""python controlled.py --cli <exe> --model <dir> --kernels <control set> --ref <ref.json> [--spread PCT] [-- <cmd ...>]
A 32-token context-0 decode before and after <cmd> plus an xrt-smi watch for foreign NPU work; no <cmd> calibrates ref.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
DECODE = re.compile(r"decode (\d+) tokens: ([\d.]+) ms/token \(([\d.]+) tok/s\)")
SYSTEM = {"WorkloadsSessionHost.exe", "svchost.exe"}


def smi() -> dict:
    try:
        out = subprocess.run([SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    lines, rows = out.splitlines(), {}
    for i, ln in enumerate(lines[:-1]):
        f = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(f) >= 3 and f[0].isdigit() and f[1].isdigit():
            name = lines[i + 1].strip().strip("|").split("|")[0].strip()
            rows[(int(f[0]), int(f[1]))] = (name, int(f[2]) if f[2].isdigit() else 0)
    return rows


class Watch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.first, self.last, self.mine, self.done = smi(), {}, set(), threading.Event()

    def run(self):
        while not self.done.wait(4):
            self.poll()

    def poll(self):
        self.mine |= {p.pid for p in psutil.Process().children(recursive=True)}
        self.last.update(smi())

    def report(self) -> tuple[list[str], int]:
        self.done.set()
        self.poll()
        foreign, system = [], 0
        for (pid, ctx), (name, subs) in self.last.items():
            grew = subs - self.first.get((pid, ctx), (name, 0))[1]
            if pid in self.mine or grew <= 0:
                continue
            if name in SYSTEM:
                system += grew
            else:
                foreign.append(f"pid {pid} {name} ctx {ctx} +{grew}")
        return foreign, system


def control(a, tag: str) -> float:
    t0 = time.time()
    r = subprocess.run([str(a.cli), "--model", str(a.model), "--kernels", str(a.kernels), "--ids", "100264",
                        "--prefill-mode", "fast", "--quiet", "--max-tokens", "33"], capture_output=True, text=True)
    m = DECODE.search(r.stdout + r.stderr)
    v = float(m[3]) if m else float("nan")
    print(f"control {tag}: {v:.2f} tok/s ({time.time() - t0:.0f} s)", flush=True)
    return v


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--kernels", required=True, type=Path, help="the control's kernel set")
    ap.add_argument("--ref", required=True, type=Path, help="the calibrated control, json")
    ap.add_argument("--tol", type=float, default=5.0, help="the control's allowed deviation, %%")
    ap.add_argument("--spread", type=float, help="max spread of the decode reps in <cmd>'s --out, %%")
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    watch = Watch()
    watch.start()
    off, rc = [], 0
    if not cmd:
        runs = [control(a, f"calib{i}") for i in range(3)]
        spread = (max(runs) - min(runs)) / statistics.median(runs)
        if not spread <= 0.02:
            off.append(f"calibration spread {spread * 100:.1f}%")
    else:
        ref = json.loads(a.ref.read_text())["tok_s"]
        before = control(a, "before")
        rc = subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT).returncode
        after = control(a, "after")
        devs = [(v - ref) / ref * 100 for v in (before, after)]
        print(f"CONTROL before {before:.2f} after {after:.2f} ref {ref:.2f} ({devs[0]:+.1f}% / {devs[1]:+.1f}%)", flush=True)
        if not all(abs(d) <= a.tol for d in devs):
            off.append(f"control off by more than {a.tol:g}%")
    foreign, system = watch.report()
    print(f"NPU WATCH: foreign {'; '.join(foreign) or 'none'}; system submissions +{system}", flush=True)
    if foreign:
        off.append("foreign NPU submissions")
    if a.spread is not None and "--out" in cmd:
        out = Path(cmd[cmd.index("--out") + 1])
        vals = [json.loads(ln)["tok_s"] for ln in out.read_text().splitlines() if ln.strip()] if out.exists() else []
        if vals:
            s = (max(vals) - min(vals)) / statistics.median(vals) * 100
            print(f"SPREAD {s:.1f}% over {len(vals)} reps", flush=True)
            if s > a.spread:
                off.append(f"spread {s:.1f}%")
    if not cmd and not off:
        a.ref.write_text(json.dumps({"tok_s": statistics.median(runs), "runs": runs, "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
        print(f"reference {statistics.median(runs):.2f} tok/s written to {a.ref}", flush=True)
    print("UNIT OFF (" + "; ".join(off) + ")" if off else "UNIT OK", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
