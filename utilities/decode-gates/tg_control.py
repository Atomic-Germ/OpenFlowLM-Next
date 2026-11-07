r"""tg128 at given depths for (cli, kernel set) configs, interleaved per depth, between two control decodes.

    python tg_control.py --model <dir> --control <cli>@<kernels> --known 22.96 --lengths 0,512 \
        --cfg old=<cli>@<kernels> --cfg new=<cli>@<kernels> [--out tg.jsonl]

Wrap it in `dibs run --measure`. The control is a tg128 at ctx 0 whose CLEAN value you already know: dibs counts every
track of one session as one holder, so a sibling's NPU run can land inside a unit that still prints CLEAN, and a control
more than 5% off its known value voids the unit. Each (depth, config) is one CLI process with --repeat 3.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "utilities" / "granite-combined-bench"))
import bench  # noqa: E402


def tg(cli: str, kernels: str, model: str, ids: list[int], n: int, repeat: int, work: Path) -> list[float]:
    (work / "tg_ids.txt").write_text(",".join(map(str, ids)))
    cmd = [cli, "--model", model, "--kernels", kernels, "--ids-file", str(work / "tg_ids.txt"), "--prefill-mode", "bfp16",
           "--quiet", *(["--gemm-block"] if n else []), "--max-tokens", "129", "--repeat", str(repeat)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return [float(m[3]) for m in bench.DECODE.finditer(p.stdout + p.stderr)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--control", required=True, help="<cli>@<kernels>")
    ap.add_argument("--known", type=float, required=True, help="the control's known CLEAN tok/s")
    ap.add_argument("--lengths", default="0")
    ap.add_argument("--cfg", action="append", default=[], help="label=<cli>@<kernels>")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    work = Path(a.out).resolve().parent if a.out else Path.cwd()
    tok = bench.tokenizer(Path(a.model))
    ccli, ckern = a.control.rsplit("@", 1)
    cfgs = [(c.split("=", 1)[0], *c.split("=", 1)[1].rsplit("@", 1)) for c in a.cfg]

    def control(when: str) -> bool:
        v = tg(ccli, ckern, a.model, [100264], 0, 1, work)
        ok = bool(v) and abs(v[0] - a.known) / a.known <= 0.05
        print(f"CONTROL {when}: {v} tok/s (known {a.known}) -> {'OK' if ok else 'OFF: VOID UNIT'}", flush=True)
        return ok

    good = control("start")
    res: dict[tuple[int, str], list[float]] = {}
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    for li, n in enumerate(int(x) for x in a.lengths.split(",")):
        ids = bench.body_ids(tok, "prose", n) if n else [100264]
        for label, cli, kern in (cfgs if li % 2 == 0 else cfgs[::-1]):
            vals = tg(cli, kern, a.model, ids, n, 3, work)
            res[(n, label)] = vals
            if out:
                out.write(json.dumps({"n": n, "cfg": label, "tok_s": vals}) + "\n")
            print(f"ctx {n:5d} {label:12s} tok/s {vals}", flush=True)
    good &= control("end")
    for (n, label), v in res.items():
        if v:
            print(f"SUMMARY ctx {n} {label}: min {min(v):.2f} median {statistics.median(v):.2f} max {max(v):.2f} tok/s")
    print("UNIT", "VALID" if good else "VOID")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
