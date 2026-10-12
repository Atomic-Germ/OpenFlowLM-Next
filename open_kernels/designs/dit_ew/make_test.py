r"""Test vectors for one dit_ew op, for open_kernels/harness's run_kernel.

    python make_test.py --op <op> --out <testdir> [--T 4608] [--seed 0]
    DE_SPEC=<testdir>/spec.json contents -> build_design.py designs/dit_ew/dit_ew.py <build>
    python make_test.py --op <op> --out <testdir> --build <build>    (writes run.cfg)
    open_kernels\harness\out\run_kernel.exe <testdir>\run.cfg
    python compare.py <testdir>

ops: ln_mod, res_ln_mod, qk, swiglu, euler, silu. Shapes follow FLUX.2 [klein] 4B:
hidden 3072, 24 heads of 128, the SwiGLU half 9216 read out of an [T, 18432] ff_in
output, q/k read out of an [T, 9216] fused QKV buffer, latents [T, 128] with the
velocity in an [T, 1024] (N-padded proj_out) buffer.

The reference (ref.npz) is float64 math on the bf16 inputs, the formulas of diffusers'
transformer_flux2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

EL = 3072
HD = 128
THETA = 2000.0


def bf(x):
    return np.asarray(x, dtype=np.float32).astype(bfloat16)


def rope_freqs():
    return 1.0 / (THETA ** (np.arange(0, 32, 2, dtype=np.float64) / 32))


def rope_tables() -> np.ndarray:
    """fp32 [3072]: FINE[p] = cos[16], sin[16] of p*f (p < 64), COARSE[a] of 64a*f (a < 8)."""
    f = rope_freqs()
    t = np.zeros(EL, dtype=np.float32)
    for p in range(64):
        a = p * f
        t[p * 32:p * 32 + 16], t[p * 32 + 16:p * 32 + 32] = np.cos(a), np.sin(a)
    for a in range(8):
        ang = 64 * a * f
        o = 64 * 32 + a * 32
        t[o:o + 16], t[o + 16:o + 32] = np.cos(ang), np.sin(ang)
    return t


QWEN_THETA = 1e6


def qwen_rope_tables() -> np.ndarray:
    """fp32 [3072]: T64[a], T8[b], T1[c] = cos[64], sin[64] of (64a | 8b | c) * f_j."""
    f = 1.0 / (QWEN_THETA ** (np.arange(0, 128, 2, dtype=np.float64) / 128))
    t = np.zeros(EL, dtype=np.float32)
    for k, step in enumerate((64, 8, 1)):
        for i in range(8):
            a = i * step * f
            o = k * 8 * 128 + i * 128
            t[o:o + 64], t[o + 64:o + 128] = np.cos(a), np.sin(a)
    return t


def qwen_qk_params(wq, wk) -> np.ndarray:
    e0 = np.zeros(EL, dtype=bfloat16)
    e0[:HD], e0[HD:2 * HD] = bf(wq), bf(wk)
    return np.concatenate([e0, qwen_rope_tables().view(np.uint16).view(bfloat16)])


def qk_params(wq, wk) -> np.ndarray:
    """dit_ew qk parameter block: 3 elements (bf16 slots)."""
    e0 = np.zeros(EL, dtype=bfloat16)
    e0[:HD], e0[HD:2 * HD] = bf(wq), bf(wk)
    tab = rope_tables().view(np.uint16).view(bfloat16)
    return np.concatenate([e0, tab])


REF_T = 10                        # diffusers' time coordinate of an edit's first reference


def rope_ref(y, g, n_txt, grid_w):
    """Interleaved 4-axis RoPE of y [..., 128] (float64) for global token indices g:
    text (0, 0, 0, l), generated (0, h, w, 0), then reference (REF_T, h, w, 0)."""
    f = rope_freqs()
    pos = np.zeros((len(g), 4))
    txt = g < n_txt
    pos[txt, 3] = g[txt]
    k = g[~txt] - n_txt
    ref = k >= grid_w * grid_w
    k = np.where(ref, k - grid_w * grid_w, k)
    pos[~txt, 0] = np.where(ref, REF_T, 0)
    pos[~txt, 1], pos[~txt, 2] = k // grid_w, k % grid_w
    ang = np.concatenate([np.outer(pos[:, a], f) for a in range(4)], axis=1)   # [T, 64]
    cos = np.repeat(np.cos(ang), 2, axis=1)[:, None, :]
    sin = np.repeat(np.sin(ang), 2, axis=1)[:, None, :]
    rot = np.stack([-y[..., 1::2], y[..., 0::2]], axis=-1).reshape(y.shape)
    return y * cos + rot * sin


def ln(x, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    return (x - m) / np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True,
                    choices=["ln_mod", "res_ln_mod", "qk", "swiglu", "euler", "silu",
                             "rms", "res_rms", "qk_qwen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", default=None, help="build dir: write run.cfg against it")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--edit", action="store_true",
                    help="qk: an edit's [text | generated | reference] sequence at 512² "
                         "(grid 32, T = 512 + 2 x 1024)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    op = a.op
    T = a.T or {"silu": 16, "euler": 4096}.get(op, 2560 if a.edit else 4608)
    row = lambda T_, off=0, ld=EL, E=1: {"off": off, "ld": ld, "T": T_, "E": E}  # noqa: E731
    pvec = lambda n: (rng.standard_normal((n, EL)) * 0.5).astype(np.float32)  # noqa: E731
    bufs, ref, spec = {}, {}, {"op": op}

    def with_pad(p):
        """P = [pad, p..., pad]: dit_ew reads one vector either side of the run."""
        return np.concatenate([np.zeros(EL, bfloat16), bf(p).reshape(-1), np.zeros(EL, bfloat16)])

    if op == "ln_mod":
        x = bf(rng.standard_normal((T, EL)) * 2 + rng.standard_normal((T, 1)))
        p = pvec(2)                                     # shift, scale
        bufs = {"X": x, "P": with_pad(p)}
        spec |= {"a": row(T), "y": row(T), "p_off": EL, "n_par": 2,
                 "idx": {"shift": 0, "scale": 1}}
        ps = bf(p).astype(np.float64)
        ref["Y"] = ln(x.astype(np.float64)) * (1 + ps[1]) + ps[0]
    elif op == "res_ln_mod":
        x = bf(rng.standard_normal((T, EL)) * 4)
        y = bf(rng.standard_normal((T, EL)))
        p = pvec(3)                                     # gate, shift, scale
        bufs = {"X": x, "B": y, "P": with_pad(p)}
        spec |= {"a": row(T), "b": row(T), "y": row(T), "z": row(T), "p_off": EL, "n_par": 3,
                 "idx": {"gate": 0, "shift": 1, "scale": 2}}
        ps = bf(p).astype(np.float64)
        xn = x.astype(np.float64) + ps[0] * y.astype(np.float64)
        ref["Z"] = xn
        ref["Y"] = ln(bf(xn).astype(np.float64)) * (1 + ps[2]) + ps[1]
    elif op == "qk":
        qkv = bf(rng.standard_normal((T, 3 * EL)) * 2)
        wq, wk = rng.uniform(0.5, 1.5, HD), rng.uniform(0.5, 1.5, HD)
        n_txt, grid_w = 512, 32 if a.edit else 64
        bufs = {"X": qkv, "B": qkv, "P": with_pad(qk_params(wq, wk).reshape(3, EL))}
        spec |= {"a": row(T, 0, 3 * EL), "b": row(T, EL, 3 * EL), "y": row(T), "z": row(T),
                 "p_off": EL, "n_par": 3, "tok0": 0, "n_txt": n_txt, "grid_w": grid_w,
                 "heads": 24}
        g = np.arange(T)
        for name, cols, w in (("Y", slice(0, EL), wq), ("Z", slice(EL, 2 * EL), wk)):
            h = qkv[:, cols].astype(np.float64).reshape(T, 24, HD)
            hn = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6) * bf(w).astype(np.float64)
            ref[name] = rope_ref(hn, g, n_txt, grid_w).reshape(T, EL)
    elif op == "swiglu":
        ff = bf(rng.standard_normal((T, 6 * EL)) * 2)
        bufs = {"X": ff, "B": ff}
        spec |= {"a": row(T, 0, 6 * EL, 3), "b": row(T, 3 * EL, 6 * EL, 3),
                 "y": row(T, 0, 3 * EL, 3)}
        g, u = ff[:, :3 * EL].astype(np.float64), ff[:, 3 * EL:].astype(np.float64)
        ref["Y"] = g / (1 + np.exp(-g)) * u
    elif op == "euler":
        n = -(-T // 24)
        n += (-n) % 16                                  # whole elements per core
        Tp = n * 24
        x = np.zeros((Tp, 128), bfloat16)
        x[:T] = bf(rng.standard_normal((T, 128)))
        v = np.zeros((Tp, 1024), bfloat16)
        v[:T, :128] = bf(rng.standard_normal((T, 128)))
        dt = np.float32(-0.2734)
        p = np.zeros((1, EL), dtype=np.float32)
        pb = np.zeros(EL, bfloat16)
        pb[:2] = np.array([dt], dtype=np.float32).view(np.uint16).view(bfloat16)
        bufs = {"X": x, "B": v, "P": np.concatenate([np.zeros(EL, bfloat16), pb,
                                                      np.zeros(EL, bfloat16)])}
        tile = lambda ld: {"kind": "tile24", "off": 0, "ld": ld, "n": n}  # noqa: E731
        spec |= {"a": tile(128), "b": tile(1024), "y": tile(128), "p_off": EL, "n_par": 1,
                 "real_T": T}
        ref["Y"] = x.astype(np.float64) + float(dt) * v[:, :128].astype(np.float64)
    elif op in ("rms", "res_rms"):         # Qwen3 RMSNorm over W = 2560 of the 3072 element
        W = 2560
        x = np.zeros((T, EL), bfloat16)
        x[:, :W] = bf(rng.standard_normal((T, W)) * 3)
        wv = np.zeros((1, EL), np.float32)
        wv[0, :W] = rng.uniform(0.2, 2.0, W)
        spec |= {"op": "ln_mod" if op == "rms" else "res_ln_mod", "norm": "rms", "W": W,
                 "a": row(T), "y": row(T), "p_off": EL, "n_par": 1, "idx": {"scale": 0}}
        bufs = {"X": x, "P": with_pad(wv)}
        xf = x.astype(np.float64)
        if op == "res_rms":
            y = np.zeros((T, EL), bfloat16)
            y[:, :W] = bf(rng.standard_normal((T, W)))
            bufs["B"] = y
            spec |= {"unit_gate": 1, "b": row(T), "z": row(T)}
            xf = xf + y.astype(np.float64)
            ref["Z"] = xf
            xf = bf(xf).astype(np.float64)
        n = xf[:, :W] / np.sqrt((xf[:, :W] ** 2).mean(-1, keepdims=True) + 1e-6)
        yref = np.zeros((T, EL))
        yref[:, :W] = bf(n).astype(np.float64) * bf(wv[0, :W]).astype(np.float64)
        ref["Y"] = yref
    elif op == "qk_qwen":                  # fused q|k|v [T, 6144]: A = q 0-23, B = q 24-31, k, v
        T = a.T or 512
        qkv = bf(rng.standard_normal((T, 2 * EL)) * 2)
        wq, wk = rng.uniform(0.5, 1.5, HD), rng.uniform(0.5, 1.5, HD)
        bufs = {"X": qkv, "B": qkv, "P": with_pad(qwen_qk_params(wq, wk).reshape(3, EL))}
        spec |= {"op": "qk", "rope": "qwen", "a": row(T, 0, 2 * EL), "b": row(T, EL, 2 * EL),
                 "y": row(T), "z": row(T), "p_off": EL, "n_par": 3, "heads": 24,
                 "b_q_heads": 8, "b_k_heads": 8}
        f = 1.0 / (QWEN_THETA ** (np.arange(0, 128, 2, dtype=np.float64) / 128))
        ang = np.outer(np.arange(T), f)
        cos = np.concatenate([np.cos(ang)] * 2, 1)[:, None, :]
        sin = np.concatenate([np.sin(ang)] * 2, 1)[:, None, :]

        def norm_rope(hd, w):
            hn = hd / np.sqrt((hd ** 2).mean(-1, keepdims=True) + 1e-6) * bf(w).astype(np.float64)
            rot = np.concatenate([-hn[..., 64:], hn[..., :64]], -1)
            return hn * cos + rot * sin

        x64 = qkv.astype(np.float64).reshape(T, 48, HD)
        q = norm_rope(x64[:, :32], wq)
        k = norm_rope(x64[:, 32:40], wk)
        ref["Y"] = q[:, :24].reshape(T, EL)
        ref["Z"] = np.concatenate([q[:, 24:], k, x64[:, 40:]], 1).reshape(T, EL)
    elif op == "silu":
        x = bf(rng.standard_normal((T, EL)) * 3)
        bufs = {"X": x}
        spec |= {"a": row(T), "y": row(T)}
        xf = x.astype(np.float64)
        ref["Y"] = xf / (1 + np.exp(-xf))

    outs = {"Y": ref["Y"].shape}
    if "Z" in ref:
        outs["Z"] = ref["Z"].shape
    sizes = {k: int(np.asarray(bufs[k]).size) if k in bufs else EL for k in ("X", "B", "P")}
    sizes |= {k: int(np.prod(outs[k])) if k in outs else EL for k in ("Y", "Z")}
    if op == "euler":
        sizes["Y"] = int(np.asarray(bufs["X"]).size)
    spec["sizes"] = sizes
    (out / "spec.json").write_text(json.dumps(spec))
    for k in ("X", "B", "P"):
        np.asarray(bufs.get(k, np.zeros(EL, bfloat16)), dtype=bfloat16).tofile(out / f"{k}.bin")
    np.savez(out / "ref.npz", op=op, **{f"ref_{k}": v for k, v in ref.items()},
             **{f"shape_{k}": np.array(v.shape) for k, v in ref.items()})

    if a.build:
        b = Path(a.build).resolve()
        cfg = ["device", f"xclbin G {b / 'final.xclbin'}", f"kernelx k G {b / 'insts.bin'}"]
        cfg += [f"buf {k} {sizes[k] * 2} {k}.bin" for k in ("X", "B", "P")]
        cfg += [f"buf {k} {sizes[k] * 2}" for k in ("Y", "Z")]
        cfg += ["run k X B P Y Z"] * a.runs
        cfg += [f"dump {k} {k}.out {sizes[k] * 2}" for k in outs] + [""]
        (out / "run.cfg").write_text("\n".join(cfg))
    print(f"{op}: T={T} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
