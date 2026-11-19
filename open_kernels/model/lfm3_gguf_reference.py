"""fp32 numpy forward of LiquidAI LFM3 (LFM3-1.7B, model_type lfm3, GGUF arch lfm3) over its GGUF -- the
independent reference the NPU route is checked against (OPEN-FAMILY-LFM3).

It reads the GGUF directly (Q4_0, Q6_K, Q8_0, F32, F16, BF16) and implements the published model --
the Hub's `modeling_lfm3.py` (Lfm3DecoderLayer / Lfm3Attention / Lfm3ShortConv / Lfm3RotaryEmbedding) --
not this tree's recipe: nothing is sliced, regrouped or requantised.

  per layer:  h = rms(x) * attn_norm                                  (HF operator_norm)
    conv:     [B | C | u] = h W_in (3 x hidden rows in that order);  bx = B * u
              conv[t] = sum_k w[:, k] bx[t - 6 + k]       (depthwise, 7 taps, zero before t = 0; the LAST tap
              pairs with this token);  o = (C * conv) W_out
    attention: q, k, v = h Wq, h Wk, h Wv;  q, k = rms over each head * q_norm / k_norm, then rotate_half RoPE
              with the zero-tail progression (theta^(-2i/hd) for the first partial * hd / 2 angles, zero for
              the rest; theta 1e6 and half the dims on `full_attention`, 1e4 and all of them on
              `sliding_attention`); causal, the newest 256 rows on a sliding layer; scale 1/sqrt(hd);
              a = softmax(qk) v * sigmoid(h Wgate) (one gate a head);  o = a Wo
    both:     res = x + rms(o) * post_attention_norm      (the sandwich norm on the block output)
              x = res + rms(down(silu(gate xm) * up xm)) * post_ffw_norm,   xm = rms(res) * ffn_norm
  head:       logits = rms(x) * output_norm @ token_embd^T      (HF embedding_norm; tied head)

    python open_kernels/model/lfm3_gguf_reference.py --gguf model.gguf --config config.json \
        --ids-file prompt.ids --out DIR [--dump-layers 0,2,3] [--positions all|last|N,M] [--npu-like]
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
from flint_gguf_reference import Gguf, bf, rms, rope, silu, attention  # noqa: E402

Q6_K = 14


def decode_q6k(raw: np.ndarray) -> np.ndarray:
    """GGUF Q6_K super-blocks (256 values, 210 B: ql[128] qh[64] int8 scales[16] f16 d) -> flat f32."""
    b = np.asarray(raw).reshape(-1, 210)
    ql, qh = b[:, :128], b[:, 128:192]
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)[:, 0]
    out = np.empty((b.shape[0], 256), np.float32)
    for half in range(2):                                  # two 128-value chunks
        l = ql[:, 64 * half:64 * half + 64]
        h = qh[:, 32 * half:32 * half + 32].astype(np.int32)
        s = sc[:, 8 * half:8 * half + 8]
        lo, hi = (l[:, :32] & 15).astype(np.int32), (l[:, :32] >> 4).astype(np.int32)
        lo2, hi2 = (l[:, 32:] & 15).astype(np.int32), (l[:, 32:] >> 4).astype(np.int32)
        q = [(lo | ((h >> 0 & 3) << 4)) - 32, (lo2 | ((h >> 2 & 3) << 4)) - 32,
             (hi | ((h >> 4 & 3) << 4)) - 32, (hi2 | ((h >> 6 & 3) << 4)) - 32]
        for j in range(4):
            sj = np.repeat(s[:, [j * 2, j * 2 + 1]], 16, axis=1)       # 16 values per scale
            out[:, 128 * half + 32 * j:128 * half + 32 * j + 32] = d[:, None] * sj * q[j]
    return out.reshape(-1)


class Gguf3(Gguf):
    def _bytes(self, kind, n):
        return n // 256 * 210 if kind == Q6_K else super()._bytes(kind, n)

    def _decode(self, raw, kind, n, narrow):
        return decode_q6k(raw) if kind == Q6_K else super()._decode(raw, kind, n, narrow)


def inv_freq(theta: float, hd: int, partial: float) -> np.ndarray:
    """Lfm3RotaryEmbedding.compute_default_rope_parameters: the progression of the FULL head dim for the
    first int(partial * hd) // 2 angles, zero (the dimensions stay unrotated) for the rest."""
    n = int(partial * hd) // 2
    f = 1.0 / theta ** (np.arange(0, 2 * n, 2, dtype=np.float64) / hd)
    return np.concatenate([f, np.zeros(hd // 2 - n)])


def short_conv(h, w_in, w_conv, w_out, rb, dump=None):
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


def forward(g: Gguf3, cfg: dict, ids: list[int], dump_layers=(), out: Path | None = None,
            positions=None, npu_like=False, layers=None, expand_gate=False):
    """npu_like: a control rounding where the NPU route will (norm outputs, the conv's y, K / V, the
    attention output and the gate product to bf16; the q4 block scales to bf16)."""
    rb = bf if npu_like else (lambda x: x)
    W = (lambda n: g.array(n, narrow=True)) if npu_like else g.array
    hid = cfg["hidden_size"]
    H, KVH = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or hid // H
    eps = float(cfg.get("norm_eps", 1e-5))
    types = cfg["layer_types"]
    rp = cfg["rope_parameters"]
    invs = {t: inv_freq(float(rp[t]["rope_theta"]), hd, float(rp[t].get("partial_rotary_factor", 1.0)))
            for t in set(types) & set(rp)}
    win = int(cfg.get("sliding_window") or 0)
    T = len(ids)
    pos = np.arange(T)
    x = g.rows("token_embd.weight", ids)
    t0 = time.time()
    n_layers = cfg["num_hidden_layers"] if layers is None else layers
    for l in range(n_layers):
        p = f"blk.{l}."
        d: dict = {}
        h = rb(rms(x, g.array(p + "attn_norm.weight"), eps))
        if types[l] == "conv":
            o = short_conv(h, W(p + "shortconv.in_proj.weight"), rb(g.array(p + "shortconv.conv.weight")),
                           W(p + "shortconv.out_proj.weight"), rb, d)
        else:
            q = (h @ W(p + "attn_q.weight").T).reshape(T, H, hd)
            k = (h @ W(p + "attn_k.weight").T).reshape(T, KVH, hd)
            v = (h @ W(p + "attn_v.weight").T).reshape(T, KVH, hd)
            inv = invs[types[l]]
            q = rope(rms(q, rb(g.array(p + "attn_q_norm.weight")), eps), pos, inv)
            k = rope(rms(k, rb(g.array(p + "attn_k_norm.weight")), eps), pos, inv)
            k, v = rb(k), rb(v)
            a = attention(q, k, v, np.full(H, -np.inf, np.float32),
                          win if types[l] == "sliding_attention" else 0)
            gate = (h @ W(p + "attn_gate.weight").T).astype(np.float64)       # [T, H], or [T, H * hd] (derived GGUF)
            gate = 1.0 / (1.0 + np.exp(-(gate[:, :, None] if gate.shape[1] == H else gate.reshape(T, H, hd))))
            a = rb((a * gate).astype(np.float32))
            o = a.reshape(T, H * hd) @ W(p + "attn_output.weight").T
            d.update({"q": q, "k": k, "v": v, "a": a})
        res = x + rb(rms(o, g.array(p + "post_attention_norm.weight"), eps))
        xm = rb(rms(res, g.array(p + "ffn_norm.weight"), eps))
        f = silu(xm @ W(p + "ffn_gate.weight").T) * (xm @ W(p + "ffn_up.weight").T)
        f = rms(rb(f) @ W(p + "ffn_down.weight").T, g.array(p + "post_ffw_norm.weight"), eps)
        x = (res + rb(f)).astype(np.float32)
        if out is not None and l in dump_layers:
            for nm, arr in [("xn", h), ("res", res), ("xm", xm), ("out", x)] + list(d.items()):
                np.asarray(arr, np.float32).tofile(out / f"layer{l}_{nm}.f32")
        print(f"layer {l:2d} ({types[l]}) done {time.time() - t0:7.1f} s", file=sys.stderr, flush=True)
    hn = rb(rms(x, g.array("output_norm.weight"), eps))
    sel = list(range(T)) if positions is None else positions
    head = "output.weight" if "output.weight" in g.tensors else "token_embd.weight"
    Wh = g.array(head)
    logits = np.empty((len(sel), Wh.shape[0]), np.float32)
    for i0 in range(0, Wh.shape[0], 16384):
        logits[:, i0:i0 + 16384] = hn[sel] @ Wh[i0:i0 + 16384].T
    return logits, sel, x


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
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()]
    a.out.mkdir(parents=True, exist_ok=True)
    pos = (None if a.positions == "all" else [len(ids) - 1] if a.positions == "last"
           else [int(p) for p in a.positions.split(",")])
    dump = {int(l) for l in a.dump_layers.split(",") if l.strip()}
    g = Gguf3(a.gguf)
    t0 = time.time()
    logits, sel, _ = forward(g, cfg, ids, dump, a.out, pos, a.npu_like, a.layers)
    logits.tofile(a.out / "logits.f32")
    (a.out / "positions.json").write_text(json.dumps(sel))
    top = np.argsort(-logits, axis=-1)[:, :5]
    rep = {"gguf": a.gguf, "ids": ids, "positions": sel, "vocab": int(logits.shape[1]),
           "argmax": top[:, 0].tolist(), "top5": top.tolist(), "seconds": round(time.time() - t0, 1),
           "npu_like": a.npu_like}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("argmax", "seconds")}))


if __name__ == "__main__":
    main()
