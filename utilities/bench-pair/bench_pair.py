#!/usr/bin/env python3
"""Measure the open engine and the closed one against each other, on one box, in one sitting.

    python utilities/bench-pair/bench_pair.py --model qwen3:4b \
        --model-dir "%USERPROFILE%\\.flm\\models\\Qwen3-4B-NPU2" \
        --kernels src/xclbins/Qwen3-4B-NPU2/open_kernels \
        --open-cli path/to/open_qwen36_cli.exe --closed "C:/Program Files/flm/flm.exe"

Every open-vs-closed ratio this project has recorded was taken from two runs minutes or
days apart, and on this hardware that is not good enough. Three things make such a pair
meaningless, and this script exists to remove all three:

1. **The NPU power mode is worth about 2x and neither engine reports which one it ran in.**
   The app sets it at startup (`src/src/main.cpp`), the standalone CLI historically did
   not, so a CLI run inherited whatever the last app run left behind. Measured here on
   2026-09-18: three identical closed `flm bench` runs minutes apart gave 9.05, 17.78 and
   16.48 tok/s of decode. This script sets the mode itself, before anything runs, and
   prints it.
2. **The first run after the NPU goes idle is slow** -- cold hardware contexts, cold page
   cache, and (see 1) possibly a mode transition still settling. The first round of each
   engine is run and thrown away.
3. **The box drifts over minutes.** Runs are INTERLEAVED (closed, open, closed, open, ...)
   so drift hits both sides equally, and the report gives min and spread per engine rather
   than one number. Quote the min; if the spread is wide, the run is not usable.

What is compared, and what is not:

  * `decode ms/token at position P` -- directly comparable. Both engines are asked for a
    step whose attention window spans P cached rows, which is the measurement the closed
    engine's own bench reports per context-length stage.
  * `TTFT for an N-token prompt` -- directly comparable ONLY when the open side has a block
    prefill route for the model (`--gemm-block`); without one it prefills a token at a time
    and the number is a floor on how bad that is, not a like-for-like.
  * Output QUALITY is not compared at all. The open side is driven with synthetic token ids
    (prefill and decode cost do not depend on WHICH ids, only on how many), so its text is
    meaningless by construction. Use `oflm-test --llm` for quality.

The closed engine is served from a scratch COPY of the container, never from the user's own
`.flm`: its registry check compares the file against a manifest and can delete and re-pull a
22 GB model when they disagree. `--base` reuses a copy made earlier.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# `flm bench` prints one of these per stage; the checkpoint line gives the prompt length it
# actually reached, which is what the open side must be given to match it.
CLOSED_RESULT = re.compile(r"TTFT:\s*([\d.]+)s,\s*Prefill Speed:\s*([\d.]+)\s*tokens/s,"
                           r"\s*Decoding Speed:\s*([\d.]+)\s*tokens/s")
CLOSED_CTX = re.compile(r"checkpoint at context length (\d+)")
OPEN_PREFILL = re.compile(r"prefill (\d+) tokens: ([\d.]+) ms")
OPEN_DECODE = re.compile(r"decode (\d+) tokens: ([\d.]+) ms/token \(([\d.]+) tok/s\)")
OPEN_STEP = re.compile(r"step @(\d+): ([\d.]+) ms \(part0 ([\d.]+), route ([\d.]+), part1 ([\d.]+), lm_head ([\d.]+)\)")


def set_power_mode(mode: str) -> None:
    """Both engines' numbers depend on this and neither states it, so state it here."""
    if mode == "none":
        print("NPU power mode: left as it was (--pmode none)")
        return
    exe = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "AMD" / "xrt-smi.exe"
    cmd = [str(exe), "configure", "--pmode", mode] if exe.exists() else ["xrt-smi", "configure", "--pmode", mode]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"could not set the NPU power mode ({' '.join(cmd)}): {r.stderr.strip() or r.stdout.strip()}\n"
                 "Pass --pmode none to measure anyway, and say so when quoting the result.")
    print(f"NPU power mode: {mode}")


def scratch_base(model_dir: Path, closed_exe: Path, base: Path | None) -> Path:
    """A private copy of the container plus the closed engine's own registry, so its
    verify-and-clean pass cannot touch the real one."""
    if base:
        return base
    root = Path(os.environ.get("TEMP", "/tmp")) / "oflm-bench-pair"
    dst = root / "models" / model_dir.name
    if not (dst / "model.q4nx").exists():
        print(f"copying {model_dir.name} -> {dst} (once; pass --base {root} to reuse)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, dst, dirs_exist_ok=True)
    for f in ("model_list.json", "model_info.json"):
        src = closed_exe.parent / f
        if src.exists():
            shutil.copy(src, root / f)
    return root


def run_closed(closed_exe: Path, base: Path, tag: str, cfg: Path) -> dict | None:
    env = dict(os.environ, FLM_MODEL_PATH=str(base), FLM_CONFIG_PATH=str(base / "model_list.json"))
    # It runs in the scratch base (that is where its CSV lands), so the config has to be an
    # absolute path -- a relative one silently resolves against the base, is not found, and
    # the run reports nothing rather than failing.
    r = subprocess.run([str(closed_exe), "bench", tag, "-i", str(cfg.resolve())], capture_output=True,
                       text=True, env=env, cwd=str(base))
    rows = CLOSED_RESULT.findall(r.stdout)
    ctx = CLOSED_CTX.findall(r.stdout)
    if not rows:
        print(f"  closed run produced no result line (exit {r.returncode}):\n"
              f"  stdout: {r.stdout[-500:]}\n  stderr: {r.stderr[-300:]}")
        return None
    # the LAST stage printed is the smallest / the one a one-stage config ran
    ttft, prefill, decode = (float(x) for x in rows[-1])
    return {"ttft_s": ttft, "prefill_tps": prefill, "decode_tps": decode,
            "decode_ms": 1000.0 / decode if decode else float("nan"),
            "prompt_tokens": int(ctx[-1]) if ctx else None}


def run_open(cli: Path, model_dir: Path, kernels: Path, ids: list[int], max_tokens: int,
             at_position: int, gemm_block: bool, pmode: str) -> dict | None:
    cmd = [str(cli), "--model", str(model_dir), "--kernels", str(kernels),
           "--ids", ",".join(str(i) for i in ids), "--max-tokens", str(max_tokens),
           "--pmode", "none"]     # the script already set it; do not let the CLI re-set it per run
    if at_position:
        cmd += ["--at-position", str(at_position)]
    if gemm_block:
        cmd.append("--gemm-block")
    r = subprocess.run(cmd, capture_output=True, text=True)
    pre, dec = OPEN_PREFILL.search(r.stderr), OPEN_DECODE.search(r.stderr)
    if not dec:
        print(f"  open run produced no decode line (exit {r.returncode}):\n{r.stderr[-800:]}")
        return None
    steps = [(int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(6)))
             for m in OPEN_STEP.finditer(r.stderr)]
    out = {"decode_ms": float(dec.group(2)), "decode_tps": float(dec.group(3)),
           "prefill_ms": float(pre.group(2)) if pre else float("nan"),
           "prompt_tokens": int(pre.group(1)) if pre else 0}
    out["ttft_s"] = out["prefill_ms"] / 1000.0
    if steps:                                   # the NPU-side split, which totals do not show
        out["part0_ms"] = statistics.median(s[2] for s in steps)
        out["lmhead_ms"] = statistics.median(s[3] for s in steps)
    return out


def summarise(name: str, runs: list[dict], keys: list[str]) -> None:
    runs = [r for r in runs if r]
    if not runs:
        print(f"  {name:8s} no usable runs")
        return
    for k in keys:
        vals = [r[k] for r in runs if k in r and r[k] == r[k]]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        spread = (hi - lo) / lo * 100 if lo else 0
        print(f"  {name:8s} {k:14s} min {lo:9.2f}  med {statistics.median(vals):9.2f}  "
              f"max {hi:9.2f}   spread {spread:5.1f}%  (n={len(vals)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the closed engine's model tag, e.g. qwen3:4b")
    ap.add_argument("--model-dir", required=True, type=Path, help="the container both engines read")
    ap.add_argument("--kernels", required=True, type=Path, help="the open kernel set (manifest.json + xclbins)")
    ap.add_argument("--open-cli", required=True, type=Path, help="open_qwen36_cli(.exe)")
    ap.add_argument("--closed", type=Path, default=Path("C:/Program Files/flm/flm.exe"))
    ap.add_argument("--config", type=Path, default=Path("utilities/bench-configs/bench-1k.json"),
                    help="the closed engine's bench config; its LAST stage is the one compared")
    ap.add_argument("--base", type=Path, default=None, help="reuse a scratch base made by an earlier run")
    ap.add_argument("--rounds", type=int, default=3, help="timed rounds per engine, after one discarded warm-up")
    ap.add_argument("--positions", default="", help="extra decode positions to measure on the open side, e.g. 0,1024")
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--pmode", default="performance", help="performance | turbo | balanced | ... | none")
    ap.add_argument("--no-gemm-block", action="store_true", help="prefill the open side one token at a time")
    a = ap.parse_args()

    set_power_mode(a.pmode)
    base = scratch_base(a.model_dir, a.closed, a.base)

    # One closed stage to learn the prompt length it reaches, so the open side prefills the
    # SAME number of tokens. Discarded as the warm-up (reason 2 in the module docstring).
    print("\nwarm-up (discarded)")
    warm = run_closed(a.closed, base, a.model, a.config)
    n_prompt = (warm or {}).get("prompt_tokens") or 1005
    print(f"  closed warm-up: prompt {n_prompt} tokens"
          + (f", TTFT {warm['ttft_s']:.2f} s, decode {warm['decode_tps']:.2f} tok/s" if warm else ""))
    # Synthetic ids of the same COUNT: timing depends on how many tokens there are, not which
    # (same shapes, same dispatches). The open side's text is meaningless and is not read.
    ids = [(1000 + (i * 7919) % 20000) for i in range(n_prompt)]
    ow = run_open(a.open_cli, a.model_dir, a.kernels, ids, a.max_tokens, 0, not a.no_gemm_block, a.pmode)
    if ow:
        print(f"  open warm-up:   TTFT {ow['ttft_s']:.2f} s, decode {ow['decode_tps']:.2f} tok/s")

    print(f"\n{a.rounds} interleaved rounds, {n_prompt}-token prompt")
    closed_runs, open_runs = [], []
    for r in range(a.rounds):
        print(f"  round {r + 1}/{a.rounds}", flush=True)
        closed_runs.append(run_closed(a.closed, base, a.model, a.config))
        time.sleep(2)
        open_runs.append(run_open(a.open_cli, a.model_dir, a.kernels, ids, a.max_tokens, 0,
                                  not a.no_gemm_block, a.pmode))
        time.sleep(2)

    print(f"\n=== {a.model}, {n_prompt}-token prompt, pmode {a.pmode} ===")
    summarise("closed", closed_runs, ["ttft_s", "decode_ms", "decode_tps"])
    summarise("open", open_runs, ["ttft_s", "decode_ms", "decode_tps", "part0_ms", "lmhead_ms"])
    ct = [r["ttft_s"] for r in closed_runs if r]
    ot = [r["ttft_s"] for r in open_runs if r]
    cd = [r["decode_ms"] for r in closed_runs if r]
    od = [r["decode_ms"] for r in open_runs if r]
    if ct and ot:
        print(f"\n  TTFT   open / closed = {min(ot) / min(ct):.2f}x   (on minima, the honest direction)")
    if cd and od:
        print(f"  decode open / closed = {min(od) / min(cd):.2f}x")

    # The decode sweep is open-side only: the closed engine's bench picks its own context
    # lengths, so a matched point needs its own stage rather than a position argument.
    for p in [int(x) for x in a.positions.split(",") if x.strip()]:
        runs = [run_open(a.open_cli, a.model_dir, a.kernels, ids[:19], a.max_tokens, p, False, a.pmode)
                for _ in range(a.rounds)]
        summarise(f"open@{p}", runs, ["decode_ms", "part0_ms", "lmhead_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
