r"""Per-position KL, top-1 agreement, perplexity and greedy agreement of prefill routes against a reference run.

    python analyze.py --corpus <dir with prompts.txt, meta.json> --ref <dir> --routes name=<dir>[,name=<dir>...] [--delete]

Every run dir holds p<k>_logits.f32 (n x V float32, every prompt position) and p<k>_gen.txt (greedy ids), as written by
llama_ref_dump and `open_qwen36_cli --ids-list`. KL is KL(ref || route) per position in nats, over the common vocabulary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def logsoftmax(x: np.ndarray) -> np.ndarray:
    m = x.max(1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def load(d: Path, k: int, n: int, V: int | None):
    raw = np.fromfile(d / f"p{k}_logits.f32", np.float32)
    if V is None:
        V = raw.size // n
    return raw.reshape(n, -1)[:, :V] if raw.size // n >= (V or 0) else raw.reshape(n, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--routes", required=True)
    a = ap.parse_args()
    meta = json.loads((a.corpus / "meta.json").read_text())
    ids = [list(map(int, l.split(","))) for l in (a.corpus / "prompts.txt").read_text().split()]
    routes = dict(r.split("=", 1) for r in a.routes.split(","))
    routes = {k: Path(v) for k, v in routes.items()}
    pos_kl = {r: [] for r in routes}
    pos_kl_len = {r: {} for r in routes}
    top1 = {r: [] for r in routes}
    nll = {r: [] for r in ["ref", *routes]}
    kl_pair = {}
    greedy = {r: [] for r in routes}
    first_div = {r: [] for r in routes}
    pair_names = []
    names = list(routes)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            kl_pair[(x, y)] = []
            pair_names.append((x, y))
    greedy_pair = {p: [] for p in pair_names}
    per_prompt = []
    for m in meta:
        k, n = m["k"], m["tokens"]
        if not all((d / f"p{k}_logits.f32").exists() for d in [a.ref, *routes.values()]):
            continue
        R = np.fromfile(a.ref / f"p{k}_logits.f32", np.float32).reshape(n, -1)
        V = min([R.shape[1]] + [np.fromfile(d / f"p{k}_logits.f32", np.float32).size // n for d in routes.values()])
        R = R[:, :V]
        lr = logsoftmax(R)
        pr = np.exp(lr)
        tgt = np.array(ids[k][1:])
        nll["ref"].append(-lr[np.arange(n - 1), tgt].mean())
        lp = {}
        for r, d in routes.items():
            X = np.fromfile(d / f"p{k}_logits.f32", np.float32).reshape(n, -1)[:, :V]
            lx = logsoftmax(X)
            lp[r] = lx
            kl = (pr * (lr - lx)).sum(1)
            pos_kl[r].append(kl)
            pos_kl_len[r].setdefault(n, []).append(kl)
            top1[r].append(R.argmax(1) == X.argmax(1))
            nll[r].append(-lx[np.arange(n - 1), tgt].mean())
            g_ref = (a.ref / f"p{k}_gen.txt").read_text().strip().split(",")
            g_r = (d / f"p{k}_gen.txt").read_text().strip().split(",") if (d / f"p{k}_gen.txt").exists() else []
            L = min(len(g_ref), len(g_r))
            fd = next((j for j in range(L) if g_ref[j] != g_r[j]), L)
            greedy[r].append(fd == L)
            first_div[r].append(fd)
        for (x, y) in pair_names:
            kl_pair[(x, y)].append((np.exp(lp[x]) * (lp[x] - lp[y])).sum(1))
            gx = (routes[x] / f"p{k}_gen.txt").read_text().strip().split(",")
            gy = (routes[y] / f"p{k}_gen.txt").read_text().strip().split(",")
            L = min(len(gx), len(gy))
            greedy_pair[(x, y)].append(gx[:L] == gy[:L])
        per_prompt.append((k, m["kind"], n, {r: float(pos_kl[r][-1].mean()) for r in routes}))
    N = len(per_prompt)
    print(f"{N} prompts, {sum(p[2] for p in per_prompt)} positions")
    print("\nroute vs reference (per-position KL(ref || route), nats)")
    print("| route | KL mean | KL median | KL p99 | top-1 agree | greedy identical (64) | mean first divergence |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for r in routes:
        kl = np.concatenate(pos_kl[r])
        t1 = np.concatenate(top1[r])
        print(f"| {r} | {kl.mean():.5f} | {np.median(kl):.5f} | {np.quantile(kl, .99):.5f} | {100 * t1.mean():.2f}% | "
              f"{sum(greedy[r])}/{N} | {np.mean(first_div[r]):.1f} |")
    print("\nKL mean by prompt length")
    lens = sorted({p[2] for p in per_prompt})
    print("| route | " + " | ".join(str(n) for n in lens) + " |")
    print("|---|" + "--:|" * len(lens))
    for r in routes:
        print(f"| {r} | " + " | ".join(f"{np.concatenate(pos_kl_len[r][n]).mean():.5f}" if n in pos_kl_len[r] else "-" for n in lens) + " |")
    print("\nperplexity over the whole corpus (teacher-forced mean NLL -> ppl); ratio to reference")
    for r in ["ref", *routes]:
        print(f"  {r}: ppl {np.exp(np.mean(nll[r])):.4f}  ratio {np.exp(np.mean(nll[r]) - np.mean(nll['ref'])):.5f}")
    for (x, y) in pair_names:
        kl = np.concatenate(kl_pair[(x, y)])
        print(f"\n{x} vs {y}: KL({x} || {y}) mean {kl.mean():.5f} p99 {np.quantile(kl, .99):.5f}; greedy identical {sum(greedy_pair[(x, y)])}/{N}")
    print("\nper-prompt mean KL (kind, tokens, per route)")
    for k, kind, n, d in per_prompt:
        print(f"  p{k} {kind} {n}: " + ", ".join(f"{r} {v:.5f}" for r, v in d.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
