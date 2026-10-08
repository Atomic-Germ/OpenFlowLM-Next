"""CPU reference for K2-Horizon-7B-Uno: the two-pass cycle of github.com/ifm-ai/uno
(nano_vllm_uno/engine/two_pass_decoding.py) over K2-Horizon-7B, offline only.

Measures what the NPU port will reproduce: tokens per cycle (two L-row forwards), the
accepted-length distribution, and that greedy Uno == greedy AR. Weights come from the
BF16 GGUF (its q/k rows already in HF's split-half order), so the base is the
container's source; --base q4_1 round-trips every projection and the head through gguf's
own Q4_1 quantizer, which is what the container holds.

    python uno_ref.py --gguf K2-Horizon-7B-BF16.gguf --adapter uno/adapter_model.safetensors \
        --tokenizer base --prompts prompts.jsonl --L 8 --max-new 128 --out out.json

Never run it while timing the NPU: it saturates the CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HID, NL, NH, KVH, HD, FF, VOCAB = 4096, 36, 32, 8, 128, 12288, 250624
GROUPS, EPS, THETA = 4, 1e-6, 1e7
LORA_SCALE = 8192.0 / 128      # lora_alpha / r: plain LoRA, not rsLoRA (nano_vllm_uno/utils/lora.py)
EOS = (1, 250019)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GGUF_NAME = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
             "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}


def _roundtrip(w: np.ndarray, qtype) -> np.ndarray:
    from gguf import quants
    return quants.dequantize(quants.quantize(w.astype(np.float32), qtype), qtype).reshape(w.shape)


def load_base(path: str, base_fmt: str) -> dict[str, torch.Tensor]:
    from gguf import GGMLQuantizationType, GGUFReader
    r = GGUFReader(path)
    t = {x.name: x for x in r.tensors}

    def mat(name: str, quant: bool) -> torch.Tensor:
        x = t[name]
        a = np.asarray(x.data).view(np.uint16).reshape(int(x.shape[1]), int(x.shape[0]))
        w = torch.from_numpy(a.copy()).view(torch.bfloat16)
        if quant and base_fmt == "q4_1":
            w = torch.from_numpy(_roundtrip(w.float().numpy(), GGMLQuantizationType.Q4_1)).to(torch.bfloat16)
        return w

    W = {"embed": mat("token_embd.weight", False), "head": mat("output.weight", True),
         "norm": torch.from_numpy(np.asarray(t["output_norm.weight"].data).copy())}
    for l in range(NL):
        for p in TARGETS:
            # a k2-horizon GGUF keeps HF's split-half q/k rows (q4nx/models/k2.py): no reorder
            W[f"{l}.{p}"] = mat(f"blk.{l}.{GGUF_NAME[p]}.weight", True)
        W[f"{l}.ln1"] = torch.from_numpy(np.asarray(t[f"blk.{l}.attn_norm.weight"].data).copy())
        W[f"{l}.ln2"] = torch.from_numpy(np.asarray(t[f"blk.{l}.ffn_norm.weight"].data).copy())
        print(f"\r  base layer {l + 1}/{NL}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return W


def load_lora(path: str, fmt: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    from gguf import GGMLQuantizationType
    from safetensors.torch import load_file
    sd = load_file(path)
    qt = {"q8": GGMLQuantizationType.Q8_0, "q4_1": GGMLQuantizationType.Q4_1}.get(fmt)
    out = {}
    for l in range(NL):
        for p in TARGETS:
            mod = "self_attn" if p in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            pre = f"model.layers.{l}.{mod}.{p}"
            if f"{pre}.lora_A.weight" not in sd:
                pre = "base_model.model." + pre
            a, b = sd[f"{pre}.lora_A.weight"], sd[f"{pre}.lora_B.weight"]
            if qt is not None:
                a = torch.from_numpy(_roundtrip(a.numpy(), qt))
                b = torch.from_numpy(_roundtrip(b.numpy(), qt))
            out[f"{l}.{p}"] = (a.to(torch.bfloat16), (b * LORA_SCALE).to(torch.bfloat16))
    return out


def group_rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    g = x.float().reshape(*x.shape[:-1], GROUPS, -1)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + EPS)
    return (g.reshape(x.shape) * w).to(torch.bfloat16)


class K2:
    """K2-Horizon-7B with a KV cache and an optional per-row LoRA mask."""

    def __init__(self, W, lora, max_ctx: int):
        self.W, self.lora = W, lora
        self.k = torch.zeros(NL, max_ctx, KVH, HD, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float64) / HD))
        ang = torch.arange(max_ctx, dtype=torch.float64)[:, None] * inv[None, :]
        self.cos = torch.cat([ang.cos(), ang.cos()], -1).float()
        self.sin = torch.cat([ang.sin(), ang.sin()], -1).float()

    def _lin(self, l, p, x, mask):
        y = x @ self.W[f"{l}.{p}"].T
        if mask is not None:
            a, b = self.lora[f"{l}.{p}"]
            z = (x @ a.T) * mask[:, None]
            y = y + z @ b.T
        return y

    def _rope(self, x, pos):
        c, s = self.cos[pos][:, None, :], self.sin[pos][:, None, :]
        x = x.float()
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], -1)
        return (x * c + rot * s).to(torch.bfloat16)

    @torch.inference_mode()
    def forward(self, ids, pos0: int, lora_rows=None) -> torch.Tensor:
        """Rows `ids` at positions pos0.. (causal over the cache and each other); writes KV
        for every row; returns fp32 logits [T, VOCAB]. lora_rows: a 0/1 list, or None."""
        T = len(ids)
        mask = None if lora_rows is None or not any(lora_rows) else \
            torch.tensor(lora_rows, dtype=torch.bfloat16)
        pos = torch.arange(pos0, pos0 + T)
        x = self.W["embed"][torch.tensor(ids)]
        res = x.float()
        n = pos0 + T
        causal = torch.ones(T, n, dtype=torch.bool).tril(pos0)
        for l in range(NL):
            h = group_rms(res, self.W[f"{l}.ln1"])
            q = self._rope(self._lin(l, "q_proj", h, mask).view(T, NH, HD), pos)
            k = self._rope(self._lin(l, "k_proj", h, mask).view(T, KVH, HD), pos)
            v = self._lin(l, "v_proj", h, mask).view(T, KVH, HD)
            self.k[l, pos0:n], self.v[l, pos0:n] = k, v
            K = self.k[l, :n].float().repeat_interleave(NH // KVH, 1).transpose(0, 1)
            V = self.v[l, :n].float().repeat_interleave(NH // KVH, 1).transpose(0, 1)
            o = F.scaled_dot_product_attention(q.float().transpose(0, 1), K, V, attn_mask=causal)
            o = o.transpose(0, 1).reshape(T, NH * HD).to(torch.bfloat16)
            res = res + self._lin(l, "o_proj", o, mask).float()
            h = group_rms(res, self.W[f"{l}.ln2"])
            g = self._lin(l, "gate_proj", h, mask)
            u = self._lin(l, "up_proj", h, mask)
            res = res + self._lin(l, "down_proj", (F.silu(g.float()) * u.float()).to(torch.bfloat16), mask).float()
        return (group_rms(res, self.W["norm"]) @ self.W["head"].T).float()


def prefill(m: K2, ids, chunk=256) -> None:
    for i in range(0, len(ids) - 1, chunk):
        m.forward(ids[i:min(i + chunk, len(ids) - 1)], i)


def ar_generate(m: K2, prompt, max_new):
    """Greedy AR, one row per forward: the baseline Uno must equal."""
    seq = list(prompt)
    prefill(m, seq)
    out = []
    while len(out) < max_new:
        tok = int(m.forward([seq[-1]], len(seq) - 1)[0].argmax())
        seq.append(tok)
        out.append(tok)
        if tok in EOS:
            break
    return out


def uno_generate(m: K2, prompt, max_new, L, rng):
    """Greedy two-pass cycles. Returns (tokens, per-cycle accepted draft counts)."""
    seq = list(prompt)
    prefill(m, seq)
    out, accepted = [], []
    while len(out) < max_new:
        n = len(seq)
        noise = rng.integers(1, VOCAB, L - 1).tolist()      # uniform ids in [1, mask_token_id)
        la = m.forward([seq[-1]] + noise, n - 1, [0] + [1] * (L - 1))
        c = int(la[0].argmax())
        drafts = la[1:].argmax(-1).tolist()
        lb = m.forward([c] + drafts, n)
        a = lb.argmax(-1).tolist()
        k = 0
        while k < L - 1 and drafts[k] == a[k]:
            k += 1
        new = [c] + drafts[:k] + [a[k]]
        accepted.append(k)
        for t in new:
            seq.append(t)
            out.append(t)
            if t in EOS or len(out) >= max_new:
                return out, accepted
    return out, accepted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer.json + chat_template.jinja")
    ap.add_argument("--prompts", required=True, help="jsonl: {\"name\", \"messages\"} per line")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--base", choices=("bf16", "q4_1"), default="bf16")
    ap.add_argument("--lora", choices=("bf16", "q8", "q4_1"), default="bf16")
    ap.add_argument("--ar", action="store_true", help="also run greedy AR and compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    t0 = time.time()
    W = load_base(a.gguf, a.base)
    lora = load_lora(a.adapter, a.lora)
    print(f"loaded in {time.time() - t0:.0f} s (base {a.base}, lora {a.lora})", file=sys.stderr)

    results = []
    for line in Path(a.prompts).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        text = tok.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        m = K2(W, lora, len(ids) + a.max_new + 2 * a.L)
        t0 = time.time()
        u, acc = uno_generate(m, ids, a.max_new, a.L, np.random.default_rng(a.seed))
        r = {"name": p["name"], "prompt_tokens": len(ids), "tokens": len(u), "cycles": len(acc),
             "tokens_per_cycle": len(u) / max(1, len(acc)), "accepted": acc, "uno": u,
             "seconds": time.time() - t0}
        if a.ar:
            m = K2(W, lora, len(ids) + a.max_new + 2 * a.L)
            ar = ar_generate(m, ids, a.max_new)
            n = min(len(ar), len(u))
            first = next((i for i in range(n) if ar[i] != u[i]), None)
            r.update(ar=ar, identical=(ar[:n] == u[:n]), first_diff=first)
        results.append(r)
        print(f"{p['name']}: {len(u)} tok / {len(acc)} cycles = {r['tokens_per_cycle']:.2f} per cycle"
              + (f", identical to AR: {r['identical']} (first diff {r['first_diff']})" if a.ar else "")
              + f"  [{r['seconds']:.0f} s]", file=sys.stderr)
        print(repr(tok.decode(u))[:300], file=sys.stderr)
        Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=1))

    allacc = [k for r in results for k in r["accepted"]]
    hist = np.bincount(allacc, minlength=a.L)
    tot_tok, tot_cyc = sum(r["tokens"] for r in results), sum(r["cycles"] for r in results)
    summary = {"tokens_per_cycle": tot_tok / max(1, tot_cyc), "tokens_per_forward": tot_tok / max(1, 2 * tot_cyc),
               "accepted_hist": hist.tolist()}
    # causal attention: an L' < L run would have accepted min(k, L'-1) on the same cycle
    summary["tokens_per_cycle_at"] = {Lp: float(np.mean([2 + min(k, Lp - 1) for k in allacc]))
                                      for Lp in range(2, a.L + 1)} if allacc else {}
    Path(a.out).write_text(json.dumps({"args": vars(a), "summary": summary, "results": results}, indent=1))
    print(json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
