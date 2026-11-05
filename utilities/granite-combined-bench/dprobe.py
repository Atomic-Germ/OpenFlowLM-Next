r"""Rows-pass and plain-step timings for several (cli, kernel set) configs at several depths, in one process tree.

    python dprobe.py --model <dir> --work <dir> --depths 0,1024,2048 --cfg label=<cli>@<set> [--cfg ...] [--what uno,tg]

uno: `--uno N --gemm-block` and the pass lines (draft / verify layers and head, decode step); tg: tg128 tok/s after the depth.
Wrap in `dibs run --measure` inside controlled.py; configs alternate order per depth.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

PASS = re.compile(r"uno pass (\w+)\s*: ([\d.]+) ms \(embed [\d.]+, patch [\d.]+, layers ([\d.]+) \[submit [\d.]+, blocked [\d.]+\], head ([\d.]+)\)")
STEP = re.compile(r"decode step: ([\d.]+) ms")
CYC = re.compile(r"uno: (\d+) tokens in (\d+) cycles \(([\d.]+) a cycle")
IDENT = re.compile(r"UNO (IDENTICAL|DIFFERS)")


def run(cli: str, kern: str, model: str, ids: list[int], work: Path, extra: list[str]) -> str:
    f = work / "dprobe_ids.txt"
    f.write_text(",".join(map(str, ids)))
    cmd = [cli, "--model", model, "--kernels", kern, "--ids-file", str(f), "--prefill-mode", "fast", "--quiet",
           *(["--gemm-block"] if len(ids) > 1 else []), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--depths", default="0,1024,2048")
    ap.add_argument("--cfg", action="append", required=True)
    ap.add_argument("--what", default="uno,tg")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    cfgs = [(c.split("=", 1)[0], *c.split("=", 1)[1].rsplit("@", 1)) for c in a.cfg]
    tok = bench.tokenizer(Path(a.model))
    for li, n in enumerate(int(x) for x in a.depths.split(",")):
        ids = bench.body_ids(tok, "prose", n) if n else [100264]
        for label, cli, kern in (cfgs if li % 2 == 0 else cfgs[::-1]):
            if "uno" in a.what:
                txt = run(cli, kern, a.model, ids, a.work, ["--uno", str(a.tokens)])
                rows = {m[1]: (float(m[2]), float(m[3]), float(m[4])) for m in PASS.finditer(txt)}
                st = STEP.search(txt)
                cy = CYC.search(txt)
                idn = IDENT.search(txt)
                tail = "" if rows else txt[-600:]
                print(f"UNO depth {n} {label}: " + " ".join(f"{k} {v[0]:.1f} (layers {v[1]:.1f} head {v[2]:.1f})" for k, v in rows.items())
                      + f"; step {st[1] if st else '?'}; {cy[3] if cy else '?'} tok/cycle; {idn[1] if idn else '?'} {tail}", flush=True)
            if "tg" in a.what:
                txt = run(cli, kern, a.model, ids, a.work, ["--max-tokens", "129", "--repeat", str(a.reps)])
                v = [float(m[3]) for m in bench.DECODE.finditer(txt)]
                print(f"TG depth {n} {label}: " + " ".join(f"{x:.2f}" for x in v) + " tok/s", flush=True)
            if "rows" in a.what:
                txt = run(cli, kern, a.model, ids, a.work, ["--rows-check", "17"])
                m = re.search(r"rows-check: .*", txt)
                print(f"ROWS depth {n} {label}: {m[0] if m else txt[-300:]} {'ROWS PASS' if 'ROWS PASS' in txt else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
