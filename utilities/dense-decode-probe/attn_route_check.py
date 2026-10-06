"""The dense block route's two attention paths against each other, on one kernel set.

OPEN-PREFILL-ATTN's full-model check for a dense family: the same prompt prefilled through
`--gemm-block` with the attention products (the default where the set declares
attn_block.prep) and with the T single-token dxB dispatches (OFLM_OPEN_ATTN_BLOCK=0). The
products are bf16 with a bf16 P, so the two are not bit-exact by design. The gate is the
argmax at every compared position, the same greedy continuation, and logits corr >= 0.9998:
the distance the routes the engine already ships keep from each other. Measured on
Qwen3-8B, 2026-10-04, 19 tokens: dxB block route against the sequential route 0.99989 at
36 layers (0.99998 at 8), the products against the sequential route 0.99987. The 35B's
0.99999 was an 8-layer prefix. Top-5 agreement is reported, not gated: a near-tie at the
fifth place flips with any rounding.
A prompt of up to 64 tokens dumps every position's logits; a longer one only its last (a
lm_head pass and 600 KB a position would otherwise dominate the run).

    python attn_route_check.py --cli <open_qwen36_cli.exe> --model <model dir> --kernels <set>
        [--tokens 19,600,981] [--max-tokens 16] [--layers N] [--seq]

--seq adds the sequential route (no --gemm-block: every token through dx) as a third run
on the short prompts, so the products' distance from dxB can be read against the distance
two routes the engine already ships keep from each other.

Prompts are real text through the model's own tokenizer.json and chat template, cut to
each length. A prompt over 256 tokens spans blocks, so the later blocks read cached rows
the earlier ones wrote. Needs the NPU: run it on a quiet box.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

TEXT = ("The lighthouse keeper had kept the same routine for thirty-one years. Every evening at dusk "
        "he climbed the hundred and twelve steps, wound the clockwork that turned the lens, trimmed "
        "the wick, and wrote three lines in the log: the weather, the ships he had seen, and one "
        "thing he had noticed that day that nobody else would have. ") * 40


def prompt_ids(model: Path, n: int) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    head = tok.encode("<|im_start|>user\n", add_special_tokens=False).ids
    tail = tok.encode("\nSummarise the story above in two sentences.<|im_end|>\n<|im_start|>assistant\n",
                      add_special_tokens=False).ids
    body = tok.encode(TEXT, add_special_tokens=False).ids[: max(0, n - len(head) - len(tail))]
    return head + body + tail


def run(cli: Path, model: Path, kernels: Path, ids: list[int], max_tokens: int, prefix: str, products: bool,
        layers: int = 0, block: bool = True):
    env = dict(os.environ)
    env["OFLM_OPEN_ATTN_BLOCK"] = "1" if products else "0"
    cmd = [str(cli), "--model", str(model), "--kernels", str(kernels), "--ids", ",".join(map(str, ids)),
           "--max-tokens", str(max_tokens), "--dump-logits", prefix]
    if block:
        cmd.append("--gemm-block")
    if layers:
        cmd += ["--layers", str(layers)]
    if len(ids) <= 64:
        cmd.append("--prefill-logits")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f"run failed (products={products}, exit {p.returncode}):\n{p.stderr[-2000:]}")
    toks = [int(m.group(1)) for m in re.finditer(r"^token (\d+)$", p.stdout, re.M)]
    pre = re.search(r"prefill (\d+) tokens: ([\d.]+) ms", p.stderr)
    return toks, float(pre.group(2)) if pre else float("nan"), p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--kernels", required=True, type=Path)
    ap.add_argument("--tokens", default="19,600,981")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--layers", type=int, default=0, help="a prefix of the model (0 = all)")
    ap.add_argument("--seq", action="store_true", help="also the sequential route, on prompts of <= 64 tokens")
    a = ap.parse_args()
    ok_all = True
    with tempfile.TemporaryDirectory(prefix="attn_route_") as tmp:
        for n in (int(x) for x in a.tokens.split(",")):
            ids = prompt_ids(a.model, n)
            res = {}
            for products in (False, True):
                prefix = str(Path(tmp) / f"n{n}_{'ag' if products else 'dxb'}")
                res[products] = run(a.cli, a.model, a.kernels, ids, a.max_tokens, prefix, products, a.layers)
            if a.seq and len(ids) <= 64:
                run(a.cli, a.model, a.kernels, ids, a.max_tokens, str(Path(tmp) / f"n{n}_seq"), False, a.layers, block=False)
                worst = {"dxb": 1.0, "ag": 1.0}
                for pos in range(len(ids)):
                    s_ = np.fromfile(Path(tmp) / f"n{n}_seq_p{pos}.bin", np.float32).astype(np.float64)
                    for tag in worst:
                        y_ = np.fromfile(Path(tmp) / f"n{n}_{tag}_p{pos}.bin", np.float32).astype(np.float64)
                        worst[tag] = min(worst[tag], float(np.corrcoef(s_, y_)[0, 1]))
                print(f"  {len(ids)} tokens, worst corr against the sequential route: dxB {worst['dxb']:.7f}, "
                      f"products {worst['ag']:.7f}")
            worst_corr, top1, top5 = 1.0, 0, 0
            positions = sorted(int(f.stem.rsplit("_p", 1)[1]) for f in Path(tmp).glob(f"n{n}_dxb_p*.bin"))
            for pos in positions:
                x = np.fromfile(Path(tmp) / f"n{n}_dxb_p{pos}.bin", np.float32).astype(np.float64)
                y = np.fromfile(Path(tmp) / f"n{n}_ag_p{pos}.bin", np.float32).astype(np.float64)
                worst_corr = min(worst_corr, float(np.corrcoef(x, y)[0, 1]))
                top1 += int(x.argmax() == y.argmax())
                top5 += int(set(np.argsort(-x)[:5]) == set(np.argsort(-y)[:5]))
            same = res[False][0] == res[True][0]
            k = len(positions)
            ok = k > 0 and worst_corr >= 0.9998 and top1 == k and same
            ok_all &= ok
            print(f"{'PASS' if ok else 'FAIL'} {len(ids)} tokens: argmax {top1}/{k}, top-5 {top5}/{k} positions, "
                  f"worst corr {worst_corr:.7f}, continuation {'identical' if same else 'DIFFERS'} "
                  f"({len(res[True][0])} tokens); prefill dxB {res[False][1]:.0f} ms, products {res[True][1]:.0f} ms")
            if not same:
                print(f"  dxB:      {res[False][0]}\n  products: {res[True][0]}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
