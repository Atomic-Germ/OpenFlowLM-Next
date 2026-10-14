r"""Run bench.py / uno_bench.py as small units, each under `dibs run --measure`, rerunning a unit until dibs says CLEAN.

    python matrix.py --cli <exe> --model <dir> --kernels <set> --tokenizer <dir> --out <dir> [--tries 5] [--only prefill,decode,uno,sweep]

A NOISY unit is kept as <unit>.noisyN.log and never read; the CLEAN one is <unit>.log with its jsonl beside it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def unit(a, name: str, cmd: list[str]) -> bool:
    for k in range(a.tries):
        log = a.out / f"{name}.log"
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run([shutil.which("dibs.cmd") or "dibs.cmd", "run", "--measure", "--label", f"granite combined {name}", "--eta", "30m",
                            "--tag", "combined-bench", "--", *cmd], stdout=f, stderr=subprocess.STDOUT)
        text = log.read_text(encoding="utf-8", errors="replace")
        if "dibs: CLEAN after" in text or ("dibs: CLEAN" in text and "NOISY after" not in text):
            print(f"{name}: CLEAN (try {k + 1})", flush=True)
            return True
        log.replace(a.out / f"{name}.noisy{k + 1}.log")
        print(f"{name}: NOISY (try {k + 1}) {text.strip().splitlines()[-1][:120]}", flush=True)
    print(f"{name}: GAVE UP", flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--only", default="prefill,decode,uno,sweep")
    ap.add_argument("--uno-lengths", default="1024,2048")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    only = a.only.split(",")
    base = [sys.executable, str(HERE / "bench.py")]
    common = ["--cli", a.cli, "--model", a.model, "--kernels", a.kernels]
    for n in (512, 1024, 2048):
        if "prefill" in only:
            unit(a, f"prefill_{n}", [*base, "prefill", *common, "--lengths", str(n), "--out", str(a.out / f"prefill_{n}.jsonl")])
        if "decode" in only:
            for r in ("fast", "bfp16"):
                unit(a, f"decode_{r}_{n}", [*base, "decode", *common, "--lengths", str(n), "--routes", r,
                                           "--out", str(a.out / f"decode_{r}_{n}.jsonl")])
    if "uno" in only:
        for n in map(int, a.uno_lengths.split(",")):
            for r in ("fast", "bfp16"):
                unit(a, f"uno_{r}_{n}", [*base, "uno", *common, "--lengths", str(n), "--routes", r,
                                        "--out", str(a.out / f"uno_{r}_{n}.jsonl")])
    if "sweep" in only:
        ub = [sys.executable, str(HERE.parent / "uno-bench" / "uno_bench.py"), "--cli", a.cli, "--model", a.model,
              "--kernels", a.kernels, "--tokenizer", a.tokenizer]
        unit(a, "sweep_default", [*ub, "--out", str(a.out / "sweep_default.jsonl")])
        unit(a, "sweep_bfp16", [*ub, "--out", str(a.out / "sweep_bfp16.jsonl"), "--gemm-block", "--prefill-mode", "bfp16"])
        unit(a, "sweep_fast_block", [*ub, "--out", str(a.out / "sweep_fast_block.jsonl"), "--gemm-block"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
