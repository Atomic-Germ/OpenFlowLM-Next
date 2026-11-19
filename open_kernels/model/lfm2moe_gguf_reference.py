"""fp32 numpy forward of LiquidAI LFM2-MoE (LFM2.5-24B-A2B, model_type lfm2_moe, GGUF arch lfm2moe) over its
GGUF -- the independent reference the NPU route is checked against (OPEN-FAMILY-LFM2MOE).

It reads the GGUF directly (our pure Q4_0 with a Q8_0 tied embedding; Q4_1, Q8_0, F32, F16 and BF16 are read
too) and implements the published model -- transformers' modular_lfm2_moe.py (+ lfm2's Lfm2ShortConv /
Lfm2Attention / Lfm2DecoderLayer) and llama.cpp's src/models/lfm2.cpp + lfm-base.cpp (build 12015), which
agree -- not this tree's recipe: nothing is sliced, regrouped or requantised.

  per layer:  h = rms(x) * attn_norm                      (HF operator_norm)
    conv:     [B | C | u] = h W_in (in_proj, 3 x hidden rows in that order);  bx = B * u
              conv[t] = w[:, 0] bx[t-2] + w[:, 1] bx[t-1] + w[:, 2] bx[t]   (depthwise, zero before t = 0;
              the GGUF's shortconv.conv.weight [hidden, 3]: the LAST tap pairs with this token)
              o = (C * conv) W_out
    attention: q, k, v = h Wq, h Wk, h Wv;  q, k = rms over each head * q_norm / k_norm, then NEOX RoPE
              (rotate_half, theta 1e7, every dim); causal; scale 1/sqrt(head_dim);  o = a Wo
              res = x + o;  xm = rms(res) * ffn_norm
    dense layers (the first num_dense_layers): x = res + down(silu(gate xm) * up xm)
    MoE layers: s = sigmoid(xm Wr) (F32 router); the top k of s + b (b = exp_probs_b, the selection bias);
              w = s of the chosen (unbiased) / their sum * routed_scaling_factor
              x = res + sum_e w_e down_e(silu(gate_e xm) * up_e xm)       (no shared expert)
  head:       logits = rms(x) * token_embd_norm @ token_embd^T   (HF embedding_norm; tied head)

HF's renormalisation adds 1e-6 to the sum, llama.cpp clamps the sum at 6.1e-5 (a no-op for four sigmoids);
this follows llama.cpp (no epsilon) unless --hf-eps. The forward is layer-major over the whole sequence:

    python open_kernels/model/lfm2moe_gguf_reference.py --gguf model.gguf --config config.json \
        --ids-file prompt.ids --out DIR [--dump-layers 0,2,3] [--positions all|last|N,M] [--npu-like]

writes `logits.f32` ([T, vocab] at the chosen positions), `positions.json`, `routes.json`, the per-layer
states asked for (`layer{l}_xn/res/xm/out.f32`, conv layers `layer{l}_b/c/u/y.f32`, attention layers
`layer{l}_q/k/v/a.f32`) and `report.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
# the GGUF reader and the elementwise helpers (rms, rotate_half RoPE, silu, bf16 rounding) are Flint's
# reference's; they hold no model logic
from flint_gguf_reference import Gguf, bf, rms, rope, silu  # noqa: E402
from hyv3_gguf_reference import attention  # noqa: E402  (causal, every row, double accumulation)


def rope_inv_freq(cfg: dict, hd: int) -> np.ndarray:
    """transformers' default rope: theta^(-2i/d) over the whole head (no scaling)."""
    theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", cfg.get("rope_theta")))
    return 1.0 / theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd)


def route(xm: np.ndarray, wr: np.ndarray, bias: np.ndarray, topk: int, scale: float, eps: float = 0.0):
    """Lfm2MoeTopKRouter (use_expert_bias, norm_topk_prob) / llama.cpp build_moe_ffn (sigmoid gate,
    exp_probs_b, norm_w, expert_weights_scale): s = sigmoid(xm Wr); the top k of s + bias (the larger first,
    ties to the lower expert as a stable sort gives); the weights are the UNBIASED s of the chosen,
    renormalised (sum + eps), times `scale`."""
    logit = xm.astype(np.float64) @ wr.astype(np.float64).T
    s = 1.0 / (1.0 + np.exp(-logit))
    top = np.argsort(-(s + bias.astype(np.float64)), axis=-1, kind="stable")[:, :topk]
    w = np.take_along_axis(s, top, axis=-1)
    w = w / (w.sum(axis=-1, keepdims=True) + eps) * scale
    return top, w.astype(np.float32)


def short_conv(h: np.ndarray, w_in: np.ndarray, w_conv: np.ndarray, w_out: np.ndarray, rb, dump=None):
    """Lfm2ShortConv over a whole sequence from a zero state. w_conv [hidden, taps]."""
    T, hid = h.shape
    bcu = h @ w_in.T
    B, C, u = bcu[:, :hid], bcu[:, hid:2 * hid], bcu[:, 2 * hid:]
    bx = (B * u).astype(np.float32)
    taps = w_conv.shape[1]
    conv = np.zeros_like(bx, dtype=np.float64)
    for k in range(taps):                 # tap k reaches back taps - 1 - k tokens
        lag = taps - 1 - k
        conv[lag:] += bx[:T - lag].astype(np.float64) * w_conv[:, k].astype(np.float64)
    y = rb((C * conv).astype(np.float32))
    if dump is not None:
        dump.update({"b": B, "c": C, "u": u, "y": y})
    return y @ w_out.T


def forward(g: Gguf, cfg: dict, ids: list[int], dump_layers=(), out: Path | None = None,
            positions=None, npu_like=False, layers=None, hf_eps=False):
    """npu_like: a control that rounds where the NPU route does (recipes/lfm2moe.py) -- the q4 block scales
    to bf16 (a no-op for a BF16-sourced pure Q4_0), the norms' outputs (xn, xm, hn), the conv's gated
    output y (bf16 into the out GEMV) and its taps, K / V as cached, the attention output, the q_norm /
    k_norm weights and each slot's h (the down GEMV's f32 table input is rounded to bf16 by
    gemv_q4_prep_f32) to bf16 -- so a gap between the NPU and the fp32 reference can be told apart from
    those roundings. The head keeps its Q8_0 fp16 scales (f32 on the NPU, exact)."""
    rb = bf if npu_like else (lambda x: x)
    W = (lambda n: g.array(n, narrow=True)) if npu_like else g.array
    X = (lambda n, e: g.expert(n, e, narrow=True)) if npu_like else g.expert
    hid = cfg["hidden_size"]
    H, KVH = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or hid // H
    eps = float(cfg.get("norm_eps", 1e-5))
    topk = cfg["num_experts_per_tok"]
    scale = float(cfg.get("routed_scaling_factor", 1.0))
    n_dense = int(cfg.get("num_dense_layers", 0))
    types = cfg["layer_types"]
    inv = rope_inv_freq(cfg, hd)
    T = len(ids)
    pos = np.arange(T)
    x = g.rows("token_embd.weight", ids)
    routes = {}
    t0 = time.time()
    n_layers = cfg["num_hidden_layers"] if layers is None else layers
    for l in range(n_layers):
        p = f"blk.{l}."
        d: dict = {}
        h = rb(rms(x, g.array(p + "attn_norm.weight"), eps))
        if types[l] == "conv":
            o = short_conv(h, W(p + "shortconv.in_proj.weight"), rb(g.array(p + "shortconv.conv.weight")),
                           W(p + "shortconv.out_proj.weight"), rb, d)
        elif types[l] == "full_attention":
            q = (h @ W(p + "attn_q.weight").T).reshape(T, H, hd)
            k = (h @ W(p + "attn_k.weight").T).reshape(T, KVH, hd)
            v = (h @ W(p + "attn_v.weight").T).reshape(T, KVH, hd)
            q = rope(rms(q, rb(g.array(p + "attn_q_norm.weight")), eps), pos, inv)
            k = rope(rms(k, rb(g.array(p + "attn_k_norm.weight")), eps), pos, inv)
            k, v = rb(k), rb(v)
            a = rb(attention(q, k, v))
            o = a.reshape(T, H * hd) @ W(p + "attn_output.weight").T
            d.update({"q": q, "k": k, "v": v, "a": a})
        else:
            raise ValueError(f"layer {l}: layer type {types[l]!r}")
        res = x + o
        xm = rb(rms(res, g.array(p + "ffn_norm.weight"), eps))
        if l < n_dense:
            f = silu(xm @ W(p + "ffn_gate.weight").T) * (xm @ W(p + "ffn_up.weight").T)
            x = (res + rb(f) @ W(p + "ffn_down.weight").T).astype(np.float32)
        else:
            top, w = route(xm, g.array(p + "ffn_gate_inp.weight"), g.array(p + "exp_probs_b.bias"), topk, scale,
                           1e-6 if hf_eps else 0.0)
            moe = np.zeros_like(res, dtype=np.float64)
            for e in np.unique(top):
                rows, slot = np.nonzero(top == e)
                xe = xm[rows]
                f = silu(xe @ X(p + "ffn_gate_exps.weight", int(e)).T) * (xe @ X(p + "ffn_up_exps.weight", int(e)).T)
                moe[rows] += rb(w[rows, slot][:, None] * f) @ X(p + "ffn_down_exps.weight", int(e)).T
            x = (res + moe).astype(np.float32)
            routes[l] = {"experts": top.tolist(), "weights": w.tolist()}
        if out is not None and l in dump_layers:
            for nm, arr in [("xn", h), ("res", res), ("xm", xm), ("out", x)] + list(d.items()):
                np.asarray(arr, np.float32).tofile(out / f"layer{l}_{nm}.f32")
            if l in routes:
                (out / f"layer{l}_route.json").write_text(json.dumps(routes[l]))
        print(f"layer {l:2d} ({types[l]}) done {time.time() - t0:7.1f} s", file=sys.stderr, flush=True)
    hn = rb(rms(x, g.array("token_embd_norm.weight"), eps))
    sel = list(range(T)) if positions is None else positions
    head = "output.weight" if "output.weight" in g.tensors else "token_embd.weight"
    Wh = g.array(head)
    logits = np.empty((len(sel), Wh.shape[0]), np.float32)
    for i0 in range(0, Wh.shape[0], 16384):
        logits[:, i0:i0 + 16384] = hn[sel] @ Wh[i0:i0 + 16384].T
    return logits, sel, routes, x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids-file", required=True, help="token ids, whitespace or comma separated")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dump-layers", default="")
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers (bring-up)")
    ap.add_argument("--positions", default="all", help="all | last | comma list")
    ap.add_argument("--npu-like", action="store_true", help="the rounding control (see forward)")
    ap.add_argument("--hf-eps", action="store_true", help="HF's +1e-6 in the top-k renormalisation")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    a.out.mkdir(parents=True, exist_ok=True)
    pos = (None if a.positions == "all" else [len(ids) - 1] if a.positions == "last"
           else [int(p) for p in a.positions.split(",")])
    dump = {int(l) for l in a.dump_layers.split(",") if l.strip()}
    g = Gguf(a.gguf)
    t0 = time.time()
    logits, sel, routes, _ = forward(g, cfg, ids, dump, a.out, pos, a.npu_like, a.layers, a.hf_eps)
    logits.tofile(a.out / "logits.f32")
    (a.out / "positions.json").write_text(json.dumps(sel))
    (a.out / "routes.json").write_text(json.dumps(routes))
    top = np.argsort(-logits, axis=-1)[:, :5]
    rep = {"gguf": a.gguf, "ids": ids, "positions": sel, "vocab": int(logits.shape[1]),
           "argmax": top[:, 0].tolist(), "top5": top.tolist(), "seconds": round(time.time() - t0, 1),
           "npu_like": a.npu_like, "hf_eps": a.hf_eps}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("argmax", "seconds")}))


if __name__ == "__main__":
    main()
