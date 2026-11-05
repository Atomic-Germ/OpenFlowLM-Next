r"""96 greedy tokens on the six uno-ref prompts and the 1024 / 2048-token chat and code prompts, each candidate set against the first.

    python greedy_cmp.py --cli <exe> --model <dir> --sets <base> <cand> ... [--tokens 96] [--only six|deep]

The deep prompts block-prefill with OFLM_OPEN_ATTN_FA=0 (host attention) so the caches of two sets are bit-equal and the
comparison tests decode alone. Wrap in `dibs run --measure`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "utilities" / "uno-bench"))
sys.path.insert(0, str(REPO / "utilities" / "granite-combined-bench"))
from uno_bench import prompt_ids  # noqa: E402
import bench  # noqa: E402


def run(cli: str, model: str, kern: str, ids: list[int], n: int, block: bool, work: Path) -> list[str]:
    f = work / "ids.txt"
    f.write_text(",".join(map(str, ids)))
    env = dict(os.environ, **({"OFLM_OPEN_ATTN_FA": "0"} if block else {}))
    r = subprocess.run([cli, "--model", model, "--kernels", kern, "--ids-file", str(f), "--max-tokens", str(n), "--quiet",
                        *(["--gemm-block"] if block else [])], capture_output=True, text=True, env=env)
    return [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("token ")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--only", choices=["six", "deep"], default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(a.model)
    tok = bench.tokenizer(Path(a.model))
    jobs: list[tuple[str, list[int], bool]] = []
    if a.only != "deep":
        for p in (json.loads(ln) for ln in (REPO / "utilities/uno-ref/prompts.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()):
            jobs.append((p["name"], prompt_ids(hf, p["messages"]), False))
    if a.only != "six":
        for n in (1024, 2048):
            for kind in ("chat", "code"):
                jobs.append((f"{kind}{n}", bench.chat_prompt(tok, Path(a.model), kind, n), True))
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for name, ids, block in jobs:
            got = {k: run(a.cli, a.model, k, ids, a.tokens, block, Path(td)) for k in a.sets}
            base = got[a.sets[0]]
            for k in a.sets[1:]:
                first = next((i for i, (x, y) in enumerate(zip(got[k], base)) if x != y), None)
                same = got[k] == base and len(base) == a.tokens
                ok &= same
                print(f"{name}: {Path(k).name} {'IDENTICAL' if same else f'DIFFERS at token {first} (lengths {len(got[k])} / {len(base)})'}", flush=True)
    print("ALL IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
