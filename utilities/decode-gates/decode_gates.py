r"""Decode gates against the first kernel set named, and the real-step A/B (wrap stepab in `dibs run --measure`).

    python decode_gates.py logits  --cli <exe> --model <dir> --sets <base> <cand> ... [--prompt 70|2100]
    python decode_gates.py greedy  --cli <exe> --model <dir> --sets <base> <cand> ... [--tokens 96]
    python decode_gates.py stepab  --cli <exe> --model <dir> --sets <a> <b> ... [--position 0] [--rounds 2]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def logits(a) -> int:
    ids = ",".join(str(1000 + (i * 7919) % 50000) for i in range(a.prompt))
    work = Path(a.work or tempfile.mkdtemp(prefix="decode_gates_"))
    res = {}
    for n, k in enumerate(a.sets):
        pre = str(work / f"lg{a.prompt}_{n}")
        for f in glob.glob(pre + "_*.bin"):
            os.remove(f)
        r = subprocess.run([a.cli, "--model", a.model, "--kernels", k, "--ids", ids, "--max-tokens", "20",
                            "--dump-logits", pre, "--quiet"], capture_output=True, text=True)
        toks = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("token ")]
        files = sorted(glob.glob(pre + "_t*.bin"), key=lambda x: int(x.rsplit("_t", 1)[1][:-4]))
        res[k] = (toks, [hashlib.md5(open(f, "rb").read()).hexdigest() for f in files])
        print(f"{k}: {len(toks)} tokens, {len(files)} logit files, exit {r.returncode}", flush=True)
    b = res[a.sets[0]]
    ok = True
    for k in a.sets[1:]:
        v = res[k]
        same = sum(x == y for x, y in zip(v[1], b[1]))
        good = v[0] == b[0] and v[1] == b[1] and len(b[1]) == 20
        ok &= good
        print(f"{k}: tokens identical {v[0] == b[0]}, logit files byte-identical {same}/{len(b[1])} -> "
              f"{'BIT-EXACT' if good else 'DIFFERS'}", flush=True)
    return 0 if ok else 1


def greedy(a) -> int:
    sys.path.insert(0, str(REPO / "utilities" / "uno-bench"))
    from uno_bench import prompt_ids  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [json.loads(l) for l in (REPO / "utilities" / "uno-ref" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    ok = True
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "ids.txt"
        for p in prompts:
            f.write_text(",".join(map(str, prompt_ids(tok, p["messages"]))))
            got = {}
            for k in a.sets:
                r = subprocess.run([a.cli, "--model", a.model, "--kernels", k, "--ids-file", str(f), "--max-tokens",
                                    str(a.tokens), "--quiet"], capture_output=True, text=True)
                got[k] = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("token ")]
            b = got[a.sets[0]]
            for k in a.sets[1:]:
                same = got[k] == b and len(b) == a.tokens
                ok &= same
                print(f"{p['name']}: {k} {'IDENTICAL' if same else 'DIFFERS'} over {len(b)} tokens", flush=True)
    print("ALL IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


def stepab(a) -> int:
    step = re.compile(r"submit-ahead, same ctx\s+([\d.]+)\s+([\d.]+)")
    best = {}
    for rnd in range(a.rounds):
        for k in (a.sets if rnd % 2 == 0 else a.sets[::-1]):
            r = subprocess.run([a.cli, "--model", a.model, "--kernels", k, "--ids", "100264", "--at-position",
                                str(a.position), "--bench-decode", "5", "--bench-step", "12"],
                               env=dict(os.environ, OFLM_OPEN_STEP_TRACE="1"), capture_output=True, text=True)
            out = r.stdout + r.stderr
            m = step.search(out)
            print(f"round {rnd} {k}: exit {r.returncode}, step {m[1] if m else '?'} / {m[2] if m else '?'} ms", flush=True)
            for l in out.splitlines():
                if l.lstrip().startswith(("dx ", "dxh ", "lm ", "ln ")):
                    print("   ", l.strip(), flush=True)
            if m:
                o = best.setdefault(k, [1e9, 1e9])
                best[k] = [min(o[0], float(m[1])), min(o[1], float(m[2]))]
    for k, (mn, md) in best.items():
        print(f"SUMMARY {k} @{a.position}: step min {mn:.1f} ms, best median {md:.1f} ms ({1000 / md:.2f} tok/s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["logits", "greedy", "stepab"])
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--prompt", type=int, default=70)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    return {"logits": logits, "greedy": greedy, "stepab": stepab}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
