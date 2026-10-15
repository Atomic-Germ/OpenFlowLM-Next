r"""The bfp16 block prefill route against the bf16 route (OPEN-PREFILL-BFP16), on one kernel set.

    python prefill_mode_check.py --cli <open_qwen36_cli.exe> --model <model dir> --kernels <set> \
        [--cases random,chat,decode,ppl,timing] [--tokens 512] [--max-tokens 80]

Each case runs the same ids through `--gemm-block` twice, OFLM_OPEN_PREFILL_MODE unset (the default route) and `bfp16`:
  random   512 random ids, the logits of every position: worst / mean correlation, argmax agreement, top-5 sets
  chat     each prompt of utilities/uno-ref/prompts.jsonl through the model's chat template, the same
  decode   those prompts greedy-decoded --max-tokens: identical tokens, or the first position they differ and the
           reference's top-2 logit margin there
  ppl      a real text (this repo's README), teacher-forced: perplexity under each route and the mean KL(bf16 || bfp16)
  timing   prefill ms at 512 and 2048 tokens, 3 runs each, the mode order alternating
Needs the NPU; run it under the shared lock's `timing` gate on a quiet box. Logits are f32 files, one per position.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def run(cli: Path, model: Path, kernels: Path, ids: list[int], mode: str, prefix: str | None, max_tokens: int = 1):
    env = dict(os.environ)
    env.pop("OFLM_OPEN_PREFILL_MODE", None)
    if mode != "fast":
        env["OFLM_OPEN_PREFILL_MODE"] = mode
    cmd = [str(cli), "--model", str(model), "--kernels", str(kernels), "--ids", ",".join(map(str, ids)),
           "--max-tokens", str(max_tokens), "--gemm-block"]
    if prefix:
        cmd += ["--dump-logits", prefix, "--prefill-logits"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f"run failed (mode {mode}, exit {p.returncode}):\n{p.stderr[-2500:]}")
    toks = [int(m.group(1)) for m in re.finditer(r"^token (\d+)$", p.stdout, re.M)]
    pre = re.search(r"prefill (\d+) tokens: ([\d.]+) ms", p.stderr)
    if mode == "bfp16" and "bfp16" not in p.stderr:
        print("  WARNING: the load log does not mention bfp16; this kernel set may have no such route", file=sys.stderr)
    return toks, float(pre.group(2)) if pre else float("nan"), p.stderr


def load_positions(prefix: str, n: int) -> np.ndarray:
    return np.stack([np.fromfile(f"{prefix}_p{i}.bin", np.float32) for i in range(n)]).astype(np.float64)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    corr = np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)])
    am = a.argmax(1) == b.argmax(1)
    top2 = np.sort(a, 1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    t5 = np.mean([set(np.argsort(-x)[:5]) == set(np.argsort(-y)[:5]) for x, y in zip(a, b)])
    return {"positions": len(a), "corr_min": float(corr.min()), "corr_mean": float(corr.mean()), "argmax": int(am.sum()),
            "flips_margin": [round(float(m), 3) for m in margin[~am]], "top5_same": float(t5)}


def show(name: str, r: dict) -> None:
    print(f"{name}: {r['positions']} positions, corr min {r['corr_min']:.6f} mean {r['corr_mean']:.6f}, argmax "
          f"{r['argmax']}/{r['positions']}" + (f" (flips at reference top-2 margins {r['flips_margin']})" if r["flips_margin"] else "")
          + f", top-5 set equal {100 * r['top5_same']:.1f}%")


def logsoftmax(x: np.ndarray) -> np.ndarray:
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def chat_ids(model: Path) -> list[tuple[str, list[int]]]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model))
    out = []
    for line in (REPO / "utilities" / "uno-ref" / "prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            text = tok.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
            out.append((p["name"], tok.encode(text, add_special_tokens=False)))
    return out


def text_ids(model: Path, n: int) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    text = (REPO / "README.md").read_text(encoding="utf-8", errors="ignore")
    return tok.encode(text, add_special_tokens=False).ids[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--kernels", required=True, type=Path)
    ap.add_argument("--cases", default="random,chat,decode,ppl,timing")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--vocab", type=int, default=100352)
    a = ap.parse_args()
    cases = a.cases.split(",")
    with tempfile.TemporaryDirectory(prefix="bfp16_check_") as tmp:
        t = Path(tmp)
        if "random" in cases:
            rng = np.random.default_rng(7)
            ids = [int(i) for i in rng.integers(0, a.vocab - 1, a.tokens)]
            for m in ("fast", "bfp16"):
                run(a.cli, a.model, a.kernels, ids, m, str(t / f"rand_{m}"))
            show("random ids", compare(load_positions(str(t / "rand_fast"), len(ids)), load_positions(str(t / "rand_bfp16"), len(ids))))
        if "chat" in cases or "decode" in cases:
            for name, ids in chat_ids(a.model):
                res = {}
                for m in ("fast", "bfp16"):
                    res[m] = run(a.cli, a.model, a.kernels, ids, m, str(t / f"{name}_{m}"),
                                 a.max_tokens if "decode" in cases else 1)
                if "chat" in cases:
                    show(f"chat {name} ({len(ids)} tok)", compare(load_positions(str(t / f"{name}_fast"), len(ids)),
                                                                   load_positions(str(t / f"{name}_bfp16"), len(ids))))
                if "decode" in cases:
                    x, y = res["fast"][0], res["bfp16"][0]
                    same = x == y
                    first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
                    where = ""
                    if first is not None:
                        # _t<i> holds the logits token i was picked from; both routes share tokens 0 .. i - 1
                        top2 = np.sort(np.fromfile(t / f"{name}_fast_t{first}.bin", np.float32))[-2:]
                        where = f" (reference top-2 margin there {float(top2[1] - top2[0]):.3f})"
                    print(f"decode {name}: {len(x)} vs {len(y)} tokens, "
                          f"{'IDENTICAL' if same else 'differ at token ' + str(first) + where}")
        if "ppl" in cases:
            ids = text_ids(a.model, a.tokens)
            lp = {}
            for m in ("fast", "bfp16"):
                run(a.cli, a.model, a.kernels, ids, m, str(t / f"ppl_{m}"))
                lp[m] = logsoftmax(load_positions(str(t / f"ppl_{m}"), len(ids)))
            nll = {m: -np.mean([lp[m][i, ids[i + 1]] for i in range(len(ids) - 1)]) for m in lp}
            kl = float(np.mean(np.sum(np.exp(lp["fast"]) * (lp["fast"] - lp["bfp16"]), 1)))
            print(f"perplexity over {len(ids) - 1} predictions: bf16 route {np.exp(nll['fast']):.4f}, bfp16 route "
                  f"{np.exp(nll['bfp16']):.4f} (ratio {np.exp(nll['bfp16'] - nll['fast']):.4f}); mean KL(bf16 || bfp16) {kl:.5f} nats")
        if "timing" in cases:
            rng = np.random.default_rng(3)
            for n in (512, 2048):
                ids = [int(i) for i in rng.integers(0, a.vocab - 1, n)]
                ms = {"fast": [], "bfp16": []}
                for rep in range(3):
                    for m in (("fast", "bfp16") if rep % 2 == 0 else ("bfp16", "fast")):
                        ms[m].append(run(a.cli, a.model, a.kernels, ids, m, None)[1])
                for m, v in ms.items():
                    print(f"prefill {n} tokens, {m}: " + ", ".join(f"{x:.0f}" for x in v) + f" ms, min {min(v):.0f} ms = {n / min(v) * 1000:.0f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
