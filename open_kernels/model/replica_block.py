"""The block prefill's host stages, in numpy, over T tokens at once -- the
reference for src/open_qwen36/block_host.cpp (OPEN-PREFILL-BATCH).

Same math as replica.py's linear_decode / attn_decode / route, with the
projections taken as inputs (the engine gets them from the GEMM) and every
shape a parameter, so the stages run on small random weights. The kernels
keep the conv state and the KV rows in bf16, so those are rounded here too;
everything else is fp64.

    python open_kernels/model/replica_block.py --fixture <dir>   # random small case for block_host_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def bf16_round(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 (round to nearest even) -> fp64, as the engine's f32_to_bf16."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16
    return r.view(np.float32).astype(np.float64)


def rms(x, w, eps):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def l2n(a, eps=1e-6):
    return a / np.sqrt((a ** 2).sum(-1, keepdims=True) + eps)


def deltanet_block(qkv, z, xn, convw, Wa, Wb, A, dtb, nw, conv_state, S, *, key_heads, value_heads, head_dim,
                   t_real=None, eps=1e-6):
    """T tokens through the linear-attention layer's middle: conv, gated delta rule, gated norm.
    qkv [T, 2*key_w + vw] and z [T, vw] are the fused projection's outputs (pre-activation),
    xn [T, hid] the normed layer input (alpha / beta read it). conv_state [taps-1, nch] and
    S [value_heads, rows, head_dim] are updated in place through the first t_real tokens
    (rows past head_dim stay zero). Returns og [T, vw] (zero past t_real)."""
    T = qkv.shape[0]
    t_real = T if t_real is None else t_real
    key_w = key_heads * head_dim
    vw = value_heads * head_dim
    grp = value_heads // key_heads
    taps = convw.shape[0]
    lanes = Wa.shape[1]
    og = np.zeros((T, vw))
    for t in range(t_real):
        rows = np.vstack([conv_state, bf16_round(qkv[t])[None, :]])
        c = silu((convw * rows).sum(0))
        conv_state[:] = rows[1:]
        q = l2n(c[:key_w].reshape(key_heads, head_dim))
        k = l2n(c[key_w:2 * key_w].reshape(key_heads, head_dim))
        v = c[2 * key_w:].reshape(value_heads, head_dim)
        al = (xn[t] @ Wa)[:value_heads] + dtb
        decay = np.exp(A * np.log1p(np.exp(al)))
        beta = sigmoid((xn[t] @ Wb)[:value_heads])
        o = np.zeros((value_heads, head_dim))
        for h in range(value_heads):
            kk, qq = k[h // grp], q[h // grp]
            Sh = S[h, :head_dim]
            Sh *= decay[h]
            delta = beta[h] * (v[h] - Sh.T @ kk)
            Sh += np.outer(kk, delta)
            o[h] = (Sh.T @ qq) / np.sqrt(head_dim)
        og[t] = (rms(o, nw, eps)).reshape(vw) * silu(z[t])
    del lanes, taps
    return og


def rope(t_, p, inv_freq):
    """Partial RoPE over the first 2 * len(inv_freq) dims of the last axis, half-split."""
    half = len(inv_freq)
    ang = p * inv_freq
    c, s = np.cos(ang), np.sin(ang)
    y = np.array(t_, dtype=np.float64, copy=True)
    x1, x2 = t_[..., :half], t_[..., half:2 * half]
    y[..., :half] = x1 * c - x2 * s
    y[..., half:2 * half] = x2 * c + x1 * s
    return y


def attention_block(q, k, v, gate, qn, kn, inv_freq, kv, pos0, *, nh, kvh, hd, t_real=None, eps=1e-6):
    """T tokens through the full-attention layer's middle. q / gate [T, nh*hd], k / v [T, kvh*hd]
    are the fused projection's outputs; kv [rows, 2*kvh*hd] the layer's cache rows
    ([K_t | V_t], bf16-valued), rows [0, pos0) valid on entry and [pos0, pos0 + t_real) written.
    Returns og [T, nh*hd], the gated attention output (zero past t_real)."""
    T = q.shape[0]
    t_real = T if t_real is None else t_real
    grp = nh // kvh
    og = np.zeros((T, nh * hd))
    for t in range(t_real):
        p = pos0 + t
        qh = rope(rms(q[t].reshape(nh, hd), qn, eps), p, inv_freq)
        kh = rope(rms(k[t].reshape(kvh, hd), kn, eps), p, inv_freq)
        vh = v[t].reshape(kvh, hd)
        kv[p, :kvh * hd] = bf16_round(kh.reshape(-1))
        kv[p, kvh * hd:] = bf16_round(vh.reshape(-1))
        K = kv[:p + 1, :kvh * hd].reshape(p + 1, kvh, hd)
        V = kv[:p + 1, kvh * hd:].reshape(p + 1, kvh, hd)
        o = np.zeros((nh, hd))
        for h in range(nh):
            s = (K[:, h // grp] @ qh[h]) / np.sqrt(hd)
            a = np.exp(s - s.max())
            o[h] = (a / a.sum()) @ V[:, h // grp]
        og[t] = (o * sigmoid(gate[t].reshape(nh, hd))).reshape(-1)
    return og


def router_block(xm, Wr, topk):
    """(probs [T, E], idx [T, topk], w [T, topk]): softmax over the experts, the top-k by
    probability (lowest index on a tie, as router_fin picks), renormalised."""
    lg = xm @ Wr
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    idx = np.argsort(-p, axis=-1, kind="stable")[:, :topk]
    w = np.take_along_axis(p, idx, -1)
    w /= w.sum(-1, keepdims=True)
    return p, idx.astype(np.int32), w


# ---- a random small case, written raw for the C++ test (block_host_test.cpp reads the shapes
# from shapes.txt and every array as little-endian f32 / bf16 / i32)

def random_case(seed=0):
    rng = np.random.default_rng(seed)
    g = dict(T=6, t_real=5, hid=64, key_heads=2, value_heads=4, head_dim=8, taps=4, lanes=32,
             nh=4, kvh=2, hd=16, rot=8, pos0=3, E=8, topk=3, s_rows=10)
    f = lambda *s: rng.standard_normal(s).astype(np.float32).astype(np.float64)  # noqa: E731  (f32-exact, as the engine sees them)
    key_w, vw = g["key_heads"] * g["head_dim"], g["value_heads"] * g["head_dim"]
    nch = 2 * key_w + vw
    case = dict(g)
    case["qkv"] = f(g["T"], nch)
    case["z"] = f(g["T"], vw)
    case["xn"] = f(g["T"], g["hid"])
    case["convw"] = f(g["taps"], nch) * 0.5
    case["Wa"] = f(g["hid"], g["lanes"]) * 0.1
    case["Wb"] = f(g["hid"], g["lanes"]) * 0.1
    case["A"] = -np.exp(f(g["value_heads"]) * 0.3)
    case["dtb"] = f(g["value_heads"]) * 0.1
    case["nw"] = 1 + 0.1 * f(g["head_dim"])
    case["conv_state"] = bf16_round(f(g["taps"] - 1, nch))
    S = np.zeros((g["value_heads"], g["s_rows"], g["head_dim"]))
    S[:, :g["head_dim"]] = f(g["value_heads"], g["head_dim"], g["head_dim"]) * 0.2
    case["S"] = S
    qw, kvw = g["nh"] * g["hd"], g["kvh"] * g["hd"]
    case["q"], case["k"], case["v"], case["gate"] = f(g["T"], qw), f(g["T"], kvw), f(g["T"], kvw), f(g["T"], qw)
    case["qn"], case["kn"] = 1 + 0.1 * f(g["hd"]), 1 + 0.1 * f(g["hd"])
    case["inv_freq"] = 1e4 ** (-np.arange(g["rot"] // 2) / (g["rot"] // 2))
    kv = np.zeros((g["pos0"] + g["T"], 2 * kvw))
    kv[:g["pos0"]] = bf16_round(f(g["pos0"], 2 * kvw))
    case["kv"] = kv
    case["xm"] = f(g["T"], g["hid"])
    case["Wr"] = f(g["hid"], g["E"]) * 0.3
    # every f32 input exactly representable, so the C++ (which reads f32) sees the same numbers
    for k, v in case.items():
        if isinstance(v, np.ndarray) and k != "inv_freq":
            case[k] = v.astype(np.float32).astype(np.float64)
    return case


def run_case(case):
    c = dict(case)
    conv_state, S, kv = c["conv_state"].copy(), c["S"].copy(), c["kv"].copy()
    og_lin = deltanet_block(c["qkv"], c["z"], c["xn"], c["convw"], c["Wa"], c["Wb"], c["A"], c["dtb"], c["nw"],
                            conv_state, S, key_heads=c["key_heads"], value_heads=c["value_heads"],
                            head_dim=c["head_dim"], t_real=c["t_real"])
    og_att = attention_block(c["q"], c["k"], c["v"], c["gate"], c["qn"], c["kn"], c["inv_freq"], kv, c["pos0"],
                             nh=c["nh"], kvh=c["kvh"], hd=c["hd"], t_real=c["t_real"])
    probs, idx, w = router_block(c["xm"], c["Wr"], c["topk"])
    return dict(og_lin=og_lin, conv_state=conv_state, S=S, og_att=og_att, kv=kv, probs=probs, idx=idx, w=w)


def write_fixture(out: Path, case: dict, res: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scalars = {k: v for k, v in case.items() if isinstance(v, int)}
    (out / "shapes.txt").write_text("".join(f"{k} {v}\n" for k, v in scalars.items()), newline="\n")
    arrays = {k: v for k, v in case.items() if isinstance(v, np.ndarray)}
    arrays.update({"out_" + k: v for k, v in res.items()})
    for k, v in arrays.items():
        if v.dtype == np.int32:
            (out / f"{k}.i32").write_bytes(np.ascontiguousarray(v).tobytes())
        elif k == "inv_freq":                       # the engine keeps the rotary frequencies in f64
            (out / f"{k}.f64").write_bytes(np.ascontiguousarray(v, dtype=np.float64).tobytes())
        else:
            (out / f"{k}.f32").write_bytes(np.ascontiguousarray(v, dtype=np.float32).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    case = random_case(a.seed)
    write_fixture(Path(a.fixture), case, run_case(case))
    print(f"wrote {a.fixture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
