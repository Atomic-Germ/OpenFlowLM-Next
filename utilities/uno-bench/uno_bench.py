"""Uno against plain decode on the NPU over a prompt file: `open_qwen36_cli --uno N` per prompt, tabulated.

Run it under the NPU lock's timing gate (see .opencode/skill/open-dxl-rows); a busy machine inflates the ratio.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

UNO = re.compile(r"uno: (\d+) tokens in (\d+) cycles \(([\d.]+) a cycle, accepted-draft histogram ([\d ]+)\): "
                 r"([\d.]+) ms/token \(([\d.]+) tok/s\)")
DECODE = re.compile(r"decode: (\d+) tokens, ([\d.]+) ms/token \(([\d.]+) tok/s\); uno is ([\d.]+)x")
VERDICT = re.compile(r"UNO (IDENTICAL to decode|DIFFERS from decode at token (\d+))")


def parse(text: str) -> dict:
    u, d, v = UNO.search(text), DECODE.search(text), VERDICT.search(text)
    if not (u and d and v):
        raise ValueError("uno-bench: the CLI's --uno report did not parse:\n" + text[-2000:])
    return {"tokens": int(u[1]), "cycles": int(u[2]), "per_cycle": float(u[3]),
            "histogram": [int(h) for h in u[4].split()], "uno_tok_s": float(u[6]), "decode_tok_s": float(d[3]),
            "speedup": float(d[4]), "identical": v[2] is None, "first_diff": None if v[2] is None else int(v[2])}


def prompt_ids(tok, messages: list[dict]) -> list[int]:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cli", required=True, help="open_qwen36_cli")
    ap.add_argument("--model", required=True, help="the model dir (model.q4nx + uno.q4nx)")
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--tokenizer", required=True, help="dir or HF id with the chat template")
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "uno-ref" / "prompts.jsonl"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out", required=True, help="jsonl, one result a prompt")
    a, extra = ap.parse_known_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = []
    with open(a.out, "w", encoding="utf-8") as out:
        for line in Path(a.prompts).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            ids = prompt_ids(tok, p["messages"])
            cmd = [a.cli, "--model", a.model, "--kernels", a.kernels, "--ids", ",".join(map(str, ids)),
                   "--uno", str(a.tokens), "--quiet", *extra]
            run = subprocess.run(cmd, capture_output=True, text=True)
            r = {"name": p["name"], "prompt_tokens": len(ids), **parse(run.stdout + run.stderr)}
            rows.append(r)
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"{r['name']:<12} {r['prompt_tokens']:>5} {r['per_cycle']:>6.2f} {r['uno_tok_s']:>8.2f} "
                  f"{r['decode_tok_s']:>8.2f} {r['speedup']:>6.2f}x  {'identical' if r['identical'] else 'DIFFERS'}",
                  flush=True)

    gmean = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
    per_cycle = sum(r["tokens"] for r in rows) / sum(r["cycles"] for r in rows)
    print(f"{len(rows)} prompts x {a.tokens}: {per_cycle:.2f} tokens a cycle, speedup {gmean:.2f}x (geometric mean), "
          f"{'all identical' if all(r['identical'] for r in rows) else 'NOT ALL IDENTICAL'}")
    return 0 if all(r["identical"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
