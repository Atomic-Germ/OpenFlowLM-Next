r"""Granite prefill, host attention against the NPU attention, on both GEMM routes, with the stage split.

    python attn_bench.py --cli <exe> --model <dir> --kernels <set> --lengths 512 --out <jsonl> \
        [--arms fast:host,fast:npu,bfp16:host,bfp16:npu] [--reps 3]

An arm is <prefill mode>:<attention>: `host` sets OFLM_OPEN_ATTN_FA=0, `npu` leaves it unset (the default). Each row is one
CLI process; the arm order alternates by rep. Wrap one length per call in `dibs run --measure`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench import body_ids, tokenizer  # noqa: E402

PREFILL = re.compile(r"prefill (\d+) tokens: ([\d.]+) ms")
STAGE = re.compile(r"gemm-prompt \[.*?: ([\d.]+) ms \(GEMM ([\d.]+), host ([\d.]+), per-token ([\d.]+), lm_head ([\d.]+)\)")
MID = re.compile(r"mid ([\d.]+) \(dn conv [\d.]+, dn rule [\d.]+, attn ([\d.]+)\), prenorm ([\d.]+), gemm tile ([\d.]+),"
                 r" gemm tr ([\d.]+), tail ([\d.]+), shared [\d.]+, state ([\d.]+), sync ([\d.]+)")
DISPATCH = re.compile(r"^\s+(\S+)\s+(\d+) calls\s+([\d.]+) ms total", re.M)


def run(a, mode: str, attn: str, ids_file: Path) -> dict:
    env = dict(os.environ)
    env.pop("OFLM_OPEN_ATTN_FA", None)
    if attn == "host":
        env["OFLM_OPEN_ATTN_FA"] = "0"
    if a.dispatch_log:
        env["OFLM_OPEN_DISPATCH_LOG"] = "1"
    cmd = [str(a.cli), "--model", str(a.model), "--kernels", str(a.kernels), "--ids-file", str(ids_file),
           "--prefill-mode", mode, "--gemm-block", "--max-tokens", "1"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    txt = p.stdout + p.stderr
    row: dict = {"mode": mode, "attn": attn, "exit": p.returncode}
    m = PREFILL.search(txt)
    if m:
        row["n"], row["ms"] = int(m[1]), float(m[2])
        row["tok_s"] = row["n"] / row["ms"] * 1000
    s = STAGE.search(txt)
    if s:
        row.update(total=float(s[1]), gemm=float(s[2]), host=float(s[3]))
    d = MID.search(txt)
    if d:
        row.update(mid=float(d[1]), attn_ms=float(d[2]), prenorm=float(d[3]), tile=float(d[4]), tr=float(d[5]),
                   tail=float(d[6]), state=float(d[7]), sync=float(d[8]))
    row["dispatch"] = {k: [int(c), float(ms)] for k, c, ms in DISPATCH.findall(txt)}
    if "ms" not in row:
        row["log"] = txt[-2000:]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--kernels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lengths", default="512")
    ap.add_argument("--arms", default="fast:host,fast:npu,bfp16:host,bfp16:npu")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dispatch-log", action="store_true", help="per-kernel dispatch times (OFLM_OPEN_DISPATCH_LOG=1)")
    a = ap.parse_args()
    tok = tokenizer(a.model)
    arms = [tuple(x.split(":")) for x in a.arms.split(",")]
    res: dict = {}
    with open(a.out, "w", encoding="utf-8") as out:
        for n in map(int, a.lengths.split(",")):
            f = a.out.parent / f"ids_{n}.txt"
            f.write_text(",".join(map(str, body_ids(tok, "prose", n))))
            for rep in range(a.reps):
                for mode, attn in (arms if rep % 2 == 0 else arms[::-1]):
                    row = run(a, mode, attn, f)
                    row["rep"] = rep
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(json.dumps({k: v for k, v in row.items() if k != "dispatch"}), flush=True)
                    if "tok_s" in row:
                        res.setdefault((mode, attn, n), []).append(row["tok_s"])
    for (mode, attn, n), v in res.items():
        print(f"SUMMARY pp{n} {mode}/{attn}: n={len(v)} min {min(v):.1f} median {statistics.median(v):.1f} "
              f"max {max(v):.1f} tok/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
