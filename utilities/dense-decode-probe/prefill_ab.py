r"""Prefill A/B on the NPU: (cli, kernel set, route) variants interleaved, one CLI process a run, with a control.

    python prefill_ab.py --model <dir> --out <jsonl> [--lengths 512,1024,2048] [--reps 3] [--route bfp16]
        [--control name=cli::set[::route]] name=cli::set[::route] [name=cli::set[::route] ...]

Run it as one `dibs run --measure` unit. Each length runs --reps rounds, the variant order reversing every round, after one
discarded run. The control (a variant whose speed is known) runs at --control-len before each length and once at the end;
if its median is more than 5 % off the known value, rerun the unit. Prompts are utilities/granite-combined-bench's prose ids.
Prints one line a run and a SUMMARY (min / median / max tok/s) a cell; every run's stage lines go to --out.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "granite-combined-bench"))
PREFILL = re.compile(r"prefill (\d+) tokens: ([\d.]+) ms")
STAGE = re.compile(r"^\s+(gemm-prompt.*|mid .*|step \d: .*)$", re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lengths", default="512,1024,2048")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--route", default="bfp16")
    ap.add_argument("--control")
    ap.add_argument("--control-len", type=int, default=512)
    ap.add_argument("variants", nargs="+")
    a = ap.parse_args()
    win = lambda p: re.sub(r"^/([a-zA-Z])/", r"\1:/", p)

    def parse(v: str):
        name, rest = v.split("=", 1)
        parts = rest.split("::")
        return name, win(parts[0]), win(parts[1]), parts[2] if len(parts) > 2 else a.route

    vs = [parse(v) for v in a.variants]
    ctl = parse(a.control) if a.control else None
    lengths = [int(n) for n in a.lengths.split(",")]
    import bench  # noqa: E402  (granite-combined-bench: the tokenizer and the prose prompt)
    tok = bench.tokenizer(a.model)
    res: dict[tuple[str, int], list[float]] = {}
    with tempfile.TemporaryDirectory(prefix="prefill_ab_") as tmp, open(a.out, "a", encoding="utf-8") as f:
        ids = {}
        for n in sorted(set(lengths + [a.control_len, 512])):
            ids[n] = Path(tmp) / f"ids{n}.txt"
            ids[n].write_text(",".join(map(str, bench.body_ids(tok, "prose", n))))

        def one(name, cli, kset, route, n, rep):
            cmd = [cli, "--model", str(a.model), "--kernels", kset, "--ids-file", str(ids[n]), "--prefill-mode", route,
                   "--gemm-block", "--max-tokens", "1"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            txt = p.stdout + p.stderr
            m = PREFILL.search(txt)
            ms = float(m[2]) if m else float("nan")
            row = {"variant": name, "n": n, "rep": rep, "ms": ms, "tok_s": n / ms * 1000 if m else None,
                   "stages": STAGE.findall(txt)}
            res.setdefault((name, n), []).append(row["tok_s"] or 0)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{name} pp{n} rep{rep}: {ms:.0f} ms ({row['tok_s'] or 0:.0f} tok/s)", flush=True)
            if not m:
                print(txt[-1500:], flush=True)

        # the NPU can come out of another job's power-mode change slow for the first run
        subprocess.run([vs[0][1], "--model", str(a.model), "--kernels", vs[0][2], "--ids-file", str(ids[512]),
                        "--prefill-mode", vs[0][3], "--gemm-block", "--max-tokens", "1"], capture_output=True, text=True)
        for li, n in enumerate(lengths):
            if ctl:
                one("ctl:" + ctl[0], *ctl[1:], a.control_len, li)
            for rep in range(a.reps):
                for name, cli, kset, route in (vs if rep % 2 == 0 else vs[::-1]):
                    one(name, cli, kset, route, n, rep)
        if ctl:
            one("ctl:" + ctl[0], *ctl[1:], a.control_len, -1)
    for (name, n), v in res.items():
        print(f"SUMMARY {name} pp{n}: min {min(v):.0f} median {statistics.median(v):.0f} max {max(v):.0f} tok/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
