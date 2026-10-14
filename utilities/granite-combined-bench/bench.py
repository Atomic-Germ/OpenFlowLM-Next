r"""Granite 4.2 3B prefill / decode / Uno-at-depth numbers on one kernel set, both prefill routes.

    python bench.py prefill|decode|uno --cli <exe> --model <dir> --kernels <set> --out <jsonl> [--routes fast,bfp16]

Wrap each subcommand in `dibs run --measure`; every result row is one CLI process. Prompts are real repo text
(prose: specs/open-engine/spec.md, code: this repo's python designs), cut to the exact token count, so no row
depends on repetition.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "utilities" / "uno-bench"))
from uno_bench import parse as parse_uno  # noqa: E402

PREFILL = re.compile(r"prefill (\d+) tokens: ([\d.]+) ms")
DECODE = re.compile(r"decode (\d+) tokens: ([\d.]+) ms/token \(([\d.]+) tok/s\)")
CODE_FILES = ["open_kernels/designs/dxl/dxl.py", "open_kernels/recipes/dense.py", "open_kernels/designs/gemm_bfp16/gemm_bfp16.py",
              "open_kernels/designs/dxl/dxl_runtime.py", "utilities/uno-bench/uno_bench.py"]


def tokenizer(model: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(model / "tokenizer.json"))


def body_ids(tok, kind: str, n: int) -> list[int]:
    if kind == "prose":
        text = (REPO / "specs/open-engine/spec.md").read_text(encoding="utf-8", errors="ignore")
    else:
        text = "\n\n".join((REPO / f).read_text(encoding="utf-8", errors="ignore") for f in CODE_FILES if (REPO / f).exists())
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) < n:
        raise SystemExit(f"only {len(ids)} {kind} tokens, need {n}")
    return ids[:n]


def chat_prompt(tok, model: Path, kind: str, n: int) -> list[int]:
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(model))
    ask = {"chat": "Summarize the text above in plain language for a new teammate.",
           "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}[kind]
    marker = "\u0001"
    tmpl = hf.apply_chat_template([{"role": "user", "content": marker + "\n\n" + ask}], add_generation_prompt=True, tokenize=False)
    head, tail = tmpl.split(marker)
    h, t = hf.encode(head, add_special_tokens=False), hf.encode("\n\n" + ask + tail, add_special_tokens=False)
    return h + body_ids(tok, "prose" if kind == "chat" else "code", n - len(h) - len(t)) + t


def run_cli(a, route: str, ids: list[int], extra: list[str], tmp: Path) -> str:
    f = tmp / "ids.txt"
    f.write_text(",".join(map(str, ids)))
    cmd = [str(a.cli), "--model", str(a.model), "--kernels", str(a.kernels), "--ids-file", str(f), "--prefill-mode", route,
           "--quiet", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr


def emit(out, row: dict) -> None:
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(json.dumps(row), flush=True)


def summary(name: str, v: list[float], unit: str) -> None:
    print(f"SUMMARY {name}: n={len(v)} min {min(v):.2f} median {statistics.median(v):.2f} max {max(v):.2f} {unit}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prefill", "decode", "uno"])
    ap.add_argument("--cli", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--kernels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--routes", default="fast,bfp16")
    ap.add_argument("--lengths", default="512,1024,2048")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()
    tok = tokenizer(a.model)
    lengths = [int(x) for x in a.lengths.split(",")]
    routes = a.routes.split(",")
    tmp = a.out.parent
    with open(a.out, "w", encoding="utf-8") as out:
        if a.mode == "prefill":
            res: dict = {}
            for n in lengths:
                ids = body_ids(tok, "prose", n)
                for rep in range(a.reps):
                    for r in (routes if rep % 2 == 0 else routes[::-1]):
                        m = PREFILL.search(run_cli(a, r, ids, ["--gemm-block", "--max-tokens", "1"], tmp))
                        ms = float(m[2]) if m else float("nan")
                        res.setdefault((r, n), []).append(n / ms * 1000)
                        emit(out, {"mode": "prefill", "route": r, "n": n, "rep": rep, "ms": ms, "tok_s": n / ms * 1000})
            for (r, n), v in res.items():
                summary(f"prefill {r} pp{n}", v, "tok/s")
        elif a.mode == "decode":
            res = {}
            for n in lengths:
                ids = body_ids(tok, "prose", n)
                for r in routes:
                    txt = run_cli(a, r, ids, ["--gemm-block", "--max-tokens", str(a.tokens + 1), "--repeat", str(a.reps)], tmp)
                    for i, m in enumerate(DECODE.finditer(txt)):
                        res.setdefault((r, n), []).append(float(m[3]))
                        emit(out, {"mode": "decode", "route": r, "n": n, "rep": i, "steps": int(m[1]), "ms_tok": float(m[2]), "tok_s": float(m[3])})
                    if (r, n) not in res or len(res[(r, n)]) < a.reps:
                        print("WARNING: decode lines missing:\n" + txt[-1500:], flush=True)
            for (r, n), v in res.items():
                summary(f"decode {r} after {n}", v, "tok/s")
        else:
            for n in lengths:
                for kind in ("chat", "code"):
                    ids = chat_prompt(tok, a.model, kind, n)
                    for r in routes:
                        txt = run_cli(a, r, ids, ["--gemm-block", "--uno", str(a.tokens)], tmp)
                        try:
                            row = parse_uno(txt)
                        except ValueError as e:
                            row = {"error": str(e)[-1500:]}
                        emit(out, {"mode": "uno", "route": r, "n": len(ids), "kind": kind, **row})
    return 0


if __name__ == "__main__":
    sys.exit(main())
