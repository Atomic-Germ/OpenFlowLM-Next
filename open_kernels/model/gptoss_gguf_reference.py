"""fp32 numpy forward of GPT-OSS over its GGUF -- the independent reference the NPU route is
checked against (OPEN-FAMILY-GPTOSS).

It reads the GGUF directly (ggml-org's MXFP4 file: Q8_0 attention / embedding / head, MXFP4
experts, F32 norms, biases, sinks and router; BF16 is read too) and implements the published model, not this tree's recipe: the
YaRN frequencies are computed here from OpenAI's formula (gpt_oss/torch/model.py, the same
as transformers' `_compute_yarn_parameters` with truncate=False), not from
`ModelSpec.rope_inv_freq`, and nothing is padded or requantised.

The forward is layer-major over the whole sequence, so each expert a layer routes to is
decoded once for every token that picked it:

    python open_kernels/model/gptoss_gguf_reference.py --gguf model.gguf --config config.json \
        --ids-file prompt.ids --out DIR [--dump-layers 0,1,23] [--positions all|last|N,M]

writes `logits.f32` ([T, vocab] at the chosen positions), `positions.json`, the per-layer
residuals asked for (`layer{l}_res.f32` after attention, `layer{l}_out.f32` after the MoE,
`layer{l}_xm.f32` the post-attention norm, `layer{l}_route.json` the experts and weights), and
`report.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gguf_pool import read_gguf  # noqa: E402

ALPHA = 1.702
LIMIT = 7.0
MXFP4 = 39
MXFP4_BLOCK = 17                     # e8m0 exponent byte + 16 bytes of nibbles per 32 values
KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def e8m0_half(e: np.ndarray) -> np.ndarray:
    """ggml_e8m0_to_fp32_half: 2^(e-128), with ggml's guard below 2."""
    x = e.astype(np.uint32)
    bits = np.where(x < 2, np.uint32(0x00200000) << x, (x - 1) << np.uint32(23))
    return bits.astype(np.uint32).view(np.float32)


def mxfp4_decode(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """GGUF MXFP4 blocks -> f32 [rows, cols]: qs[j] low nibble is value j, high nibble j+16."""
    b = raw.reshape(rows, cols // 32, MXFP4_BLOCK)
    d = e8m0_half(b[:, :, 0])[..., None]
    qs = b[:, :, 1:]
    lo = KVALUES[qs & 15]
    hi = KVALUES[qs >> 4]
    return (np.concatenate([lo, hi], axis=2) * d).reshape(rows, cols)


def q8_0_decode(raw: np.ndarray, narrow: bool = False) -> np.ndarray:
    """GGUF Q8_0 blocks (fp16 d, 32 int8) -> flat f32 (`narrow`: d rounded to bf16 first)."""
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    if narrow:
        d = bf(d)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


class Gguf:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.meta, self.tensors, base = read_gguf(self.path)
        align = int(self.meta.get("general.alignment", 32))
        self.base = base + (align - base % align) % align if align != 32 else base
        self.mm = np.memmap(self.path, np.uint8, "r")

    def info(self, name: str) -> dict:
        return self.tensors[name]

    def raw(self, name: str, nbytes: int) -> np.ndarray:
        t = self.tensors[name]
        o = self.base + t["offset"]
        return self.mm[o:o + nbytes]

    def array(self, name: str, narrow: bool = False) -> np.ndarray:
        """F32 / BF16 / Q8_0 tensor as f32, row-major (dims reversed); `narrow` rounds a Q8_0
        block scale to bf16 first, as the NPU's q8 half-tiles hold it."""
        t = self.tensors[name]
        shape = tuple(reversed(t["dims"]))
        n = int(np.prod(shape))
        if t["gguf_type"] == 0:
            return np.frombuffer(self.raw(name, 4 * n), np.float32).reshape(shape)
        if t["gguf_type"] == 30:
            u = np.frombuffer(self.raw(name, 2 * n), np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if t["gguf_type"] == 8:
            return q8_0_decode(np.asarray(self.raw(name, n // 32 * 34)), narrow).reshape(shape)
        raise ValueError(f"{name}: type {t['type']} is not F32/BF16/Q8_0")

    def rows(self, name: str, rows) -> np.ndarray:
        """Rows of a BF16 or Q8_0 [rows, cols] tensor as f32 (the embedding lookup)."""
        t = self.tensors[name]
        cols = t["dims"][0]
        o = self.base + t["offset"]
        rb = {30: 2 * cols, 8: cols // 32 * 34}[t["gguf_type"]]
        out = np.empty((len(rows), cols), np.float32)
        for i, r in enumerate(rows):
            b = np.asarray(self.mm[o + rb * r:o + rb * (r + 1)])
            if t["gguf_type"] == 30:
                out[i] = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                out[i] = q8_0_decode(b)
        return out

    def expert(self, name: str, e: int) -> np.ndarray:
        t = self.tensors[name]
        assert t["gguf_type"] == MXFP4, f"{name}: {t['type']}"
        cols, rows, ne = t["dims"]
        per = rows * cols // 32 * MXFP4_BLOCK
        o = self.base + t["offset"] + e * per
        return mxfp4_decode(np.asarray(self.mm[o:o + per]), rows, cols)


def yarn_inv_freq(cfg: dict) -> tuple[np.ndarray, float]:
    """OpenAI's gpt_oss RotaryEmbedding._compute_concentration_and_inv_freq."""
    hd = cfg["head_dim"]
    base = float(cfg["rope_theta"])
    sc = cfg.get("rope_scaling") or {}
    d_half = hd / 2
    freq = base ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    if not sc:
        return 1.0 / freq, 1.0
    factor = float(sc["factor"])
    ctx0 = float(sc["original_max_position_embeddings"])
    beta, alpha = float(sc.get("beta_fast", 32)), float(sc.get("beta_slow", 1))
    conc = 0.1 * math.log(factor) + 1.0
    low = d_half * math.log(ctx0 / (beta * 2 * math.pi)) / math.log(base)
    high = d_half * math.log(ctx0 / (alpha * 2 * math.pi)) / math.log(base)
    if sc.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    interp = 1.0 / (factor * freq)
    extrap = 1.0 / freq
    ramp = (np.arange(d_half) - low) / (high - low)
    mask = 1 - np.clip(ramp, 0, 1)
    return interp * (1 - mask) + extrap * mask, conc


def rope(x: np.ndarray, pos: np.ndarray, inv: np.ndarray, conc: float) -> np.ndarray:
    """Half-split rotation per head: x [T, H, hd]."""
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    cos = (np.cos(ang) * conc).astype(np.float32)[:, None, :]
    sin = (np.sin(ang) * conc).astype(np.float32)[:, None, :]
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return (x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)).astype(np.float32) * w


def attention(q, k, v, sinks, window: int):
    """q [T, H, hd], k / v [T, KVH, hd]: causal, `window` rows (0 = all), learned sinks."""
    T, H, hd = q.shape
    kvh = k.shape[1]
    g = H // kvh
    out = np.empty((T, H, hd), np.float32)
    scale = hd ** -0.5
    for t in range(T):
        s0 = max(0, t - window + 1) if window else 0
        kk, vv = k[s0:t + 1], v[s0:t + 1]                       # [L, KVH, hd]
        qh = q[t].reshape(kvh, g, hd)
        sc = np.einsum("kgd,lkd->kgl", qh.astype(np.float64), kk.astype(np.float64)) * scale
        sk = sinks.reshape(kvh, g, 1).astype(np.float64)
        m = np.maximum(sc.max(axis=-1, keepdims=True), sk)
        p = np.exp(sc - m)
        den = p.sum(axis=-1, keepdims=True) + np.exp(sk - m)
        out[t] = (np.einsum("kgl,lkd->kgd", p / den, vv.astype(np.float64))).reshape(H, hd)
    return out


def clamped_swiglu(gate, up):
    gate = np.minimum(gate, LIMIT)
    up = np.clip(up, -LIMIT, LIMIT)
    return (up + 1) * (gate / (1 + np.exp(-ALPHA * gate)))


def bf(x):
    """round to bf16 (round to nearest even) and back"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def bfp16(x, on=True):
    """round along the last axis to bfp16ebs8 and back, as gemm_bfp16's host packer does (designs/gemm_bfp16/
    pack.py encode_blocks: eight values share the largest exponent e, q = rint(x 2^(133 - e)) in [-128, 127])"""
    if not on:
        return x
    x = np.ascontiguousarray(x, np.float32)
    b = x.reshape(x.shape[:-1] + (x.shape[-1] // 8, 8))
    e = ((b.view(np.uint32) >> 23) & 0xFF).astype(np.int64).max(axis=-1, keepdims=True)
    q = np.clip(np.rint(np.ldexp(b.astype(np.float64), 133 - e)), -128, 127)
    return np.ldexp(q, e - 133).astype(np.float32).reshape(x.shape)


BFP16_PARTS = ("qkv", "o", "glu", "down", "w")


def forward(g: Gguf, cfg: dict, ids: list[int], dump_layers=(), out: Path | None = None,
            positions=None, npu_like=False, bfp16_like=()):
    """npu_like: a control that rounds where the NPU route does (recipes/gptoss.py) -- Q8_0 block
    scales to bf16, the norms' outputs, K / V as cached, the attention output and xm to bf16 -- so
    a gap between the NPU and the fp32 reference can be told apart from those roundings.

    bfp16_like: the block prefill's control (OPEN-PREFILL-GPTOSS) -- the named GEMMs' activations
    rounded to bfp16 as the host packs them (qkv: the normed input, o: the attention output, glu: xm,
    down: the SwiGLU output), "w" the attention weights rounded to bfp16 from their Q8_0 values, and
    K / V to bf16 as the cache keeps them. The experts' MXFP4 weights are exact in bfp16."""
    if bfp16_like:
        unknown = set(bfp16_like) - set(BFP16_PARTS)
        assert not unknown, unknown
    bq = lambda part, x: bfp16(x, part in bfp16_like)
    rb = bf if npu_like else (lambda x: x)
    rk = bf if (npu_like or bfp16_like) else (lambda x: x)
    if bfp16_like and "w" in bfp16_like:
        W = lambda n: bfp16(g.array(n))
    else:
        W = (lambda n: g.array(n, narrow=True)) if npu_like else g.array
    H, KVH, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    topk = cfg["num_experts_per_tok"]
    window = int(cfg["sliding_window"])
    types = cfg["layer_types"]
    inv, conc = yarn_inv_freq(cfg)
    T = len(ids)
    pos = np.arange(T)
    x = g.rows("token_embd.weight", ids)
    routes = {}
    t0 = time.time()
    for l in range(cfg["num_hidden_layers"]):
        p = f"blk.{l}."
        h = bq("qkv", rb(rms(x, g.array(p + "attn_norm.weight"), eps)))
        q = h @ W(p + "attn_q.weight").T + g.array(p + "attn_q.bias")
        k = h @ W(p + "attn_k.weight").T + g.array(p + "attn_k.bias")
        v = h @ W(p + "attn_v.weight").T + g.array(p + "attn_v.bias")
        q = rope(q.reshape(T, H, hd), pos, inv, conc)
        k = rk(rope(k.reshape(T, KVH, hd), pos, inv, conc))
        v = rk(v.reshape(T, KVH, hd))
        a = rb(attention(q, k, v, g.array(p + "attn_sinks.weight"),
                         window if types[l] == "sliding_attention" else 0))
        a = bq("o", a.reshape(T, H * hd))
        o = a @ W(p + "attn_output.weight").T + g.array(p + "attn_output.bias")
        res = x + o
        xm = rb(rms(res, g.array(p + "post_attention_norm.weight"), eps))
        logit = xm @ g.array(p + "ffn_gate_inp.weight").T + g.array(p + "ffn_gate_inp.bias")
        top = np.argsort(-logit, axis=-1, kind="stable")[:, :topk]
        sel = np.take_along_axis(logit, top, axis=-1).astype(np.float64)
        w = np.exp(sel - sel.max(axis=-1, keepdims=True))
        w = (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
        bg = g.array(p + "ffn_gate_exps.bias")
        bu = g.array(p + "ffn_up_exps.bias")
        bd = g.array(p + "ffn_down_exps.bias")
        moe = np.zeros_like(res)
        for e in np.unique(top):
            rows, slot = np.nonzero(top == e)
            xe = bq("glu", xm[rows])
            gate = xe @ g.expert(p + "ffn_gate_exps.weight", int(e)).T + bg[e]
            up = xe @ g.expert(p + "ffn_up_exps.weight", int(e)).T + bu[e]
            d = bq("down", clamped_swiglu(gate, up)) @ g.expert(p + "ffn_down_exps.weight", int(e)).T + bd[e]
            moe[rows] += w[rows, slot][:, None] * d
        x = res + moe
        routes[l] = {"experts": top.tolist(), "weights": w.tolist()}
        if out is not None and l in dump_layers:
            res.astype(np.float32).tofile(out / f"layer{l}_res.f32")
            xm.astype(np.float32).tofile(out / f"layer{l}_xm.f32")
            x.astype(np.float32).tofile(out / f"layer{l}_out.f32")
            (out / f"layer{l}_route.json").write_text(json.dumps(routes[l]))
        print(f"layer {l:2d} done {time.time() - t0:7.1f} s", file=sys.stderr, flush=True)
    hn = rb(rms(x, g.array("output_norm.weight"), eps))
    sel = list(range(T)) if positions is None else positions
    W = g.array("output.weight")
    logits = np.empty((len(sel), W.shape[0]), np.float32)
    for i0 in range(0, W.shape[0], 16384):
        logits[:, i0:i0 + 16384] = hn[sel] @ W[i0:i0 + 16384].T
    return logits, sel, routes, x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids-file", required=True, help="token ids, whitespace or comma separated")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dump-layers", default="")
    ap.add_argument("--positions", default="all", help="all | last | comma list")
    ap.add_argument("--npu-like", action="store_true", help="the rounding control (see forward)")
    ap.add_argument("--bfp16-like", default="", help="the block prefill's control: comma list of "
                    + "|".join(BFP16_PARTS) + ", or all")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    a.out.mkdir(parents=True, exist_ok=True)
    pos = (None if a.positions == "all" else [len(ids) - 1] if a.positions == "last"
           else [int(p) for p in a.positions.split(",")])
    dump = {int(l) for l in a.dump_layers.split(",") if l.strip()}
    g = Gguf(a.gguf)
    t0 = time.time()
    bl = BFP16_PARTS if a.bfp16_like == "all" else tuple(p for p in a.bfp16_like.split(",") if p)
    logits, sel, routes, _ = forward(g, cfg, ids, dump, a.out, pos, a.npu_like, bl)
    logits.tofile(a.out / "logits.f32")
    (a.out / "positions.json").write_text(json.dumps(sel))
    (a.out / "routes.json").write_text(json.dumps(routes))
    top = np.argsort(-logits, axis=-1)[:, :5]
    rep = {"gguf": a.gguf, "ids": ids, "positions": sel, "vocab": int(logits.shape[1]),
           "argmax": top[:, 0].tolist(), "top5": top.tolist(), "seconds": round(time.time() - t0, 1),
           "yarn_inv_freq": yarn_inv_freq(cfg)[0].tolist(), "yarn_concentration": yarn_inv_freq(cfg)[1]}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("argmax", "seconds")}))


if __name__ == "__main__":
    main()
