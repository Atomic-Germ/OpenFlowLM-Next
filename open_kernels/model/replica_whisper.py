r"""Whisper-large-v3-turbo's encoder in numpy, decomposed exactly as the open engine runs it.

    python open_kernels/model/replica_whisper.py --model-dir DIR --goldens DIR [--exact] [--clip NAME]

Every matrix product goes through `gemm()`, which is what the NPU does: A and B rounded to
bf16 (round-to-nearest-even), products accumulated in fp32, C in fp32. Everything else --
bias, LayerNorm, GELU, attention, residual -- is fp32 on the host, as the engine's host
side will be. The conv stem is an im2col GEMM with K ordered tap-major (K index =
tap * C_in + channel), so each im2col row is three contiguous time-major rows:

  conv1  A[t] = [mel[t-1] | mel[t] | mel[t+1]]      3000 x 384   -> 1280
  conv2  A[t] = [h1[2t-1] | h1[2t] | h1[2t+1]]      1500 x 3840  -> 1280  (stride 2)

with zero rows outside [0, T). Q|K|V is one GEMM of N = 3840 with a fused bias whose K third
is zero (k_proj has no bias). Padding to M = 1536 is left out: padded rows are zeros that
the engine never lets into attention as keys, so they cannot change a real row.

What it is for. `--exact` keeps fp32 weights and inputs everywhere, and must reproduce
transformers' float64 goldens to fp32 rounding: that checks the decomposition. Without it,
the bf16 figures are the CEILING the engine can reach -- whisper-xdna measured that 0.999
cosine is not reachable at 32 layers in bf16, so the engine's gates are calibrated against
this replica, not set in advance.

Reports, per encoder layer, cosine against the golden in float64 over the 1500 real rows:
chained (the replica's own previous output) and teacher-forced (the golden input of that
layer), then the same for enc.out and the four decoder layers' cross K and V.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

D, H, HD, FFN, NL, NDEC = 1280, 20, 64, 5120, 32, 4
EPS = 1e-5


def bf16(x: np.ndarray) -> np.ndarray:
    """Round fp32 to bf16 (nearest, ties to even) and return it as fp32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return r.view(np.float32)


class Ops:
    def __init__(self, exact: bool):
        self.q = (lambda x: np.asarray(x, np.float32)) if exact else bf16

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """C[M,N] = A[M,K] @ B[K,N]: bf16 operands, fp32 accumulation and output."""
        return self.q(a) @ b          # b is already quantised at load


def erf(x: np.ndarray) -> np.ndarray:
    """numpy has no erf. Abramowitz & Stegun 7.1.26 in float64: |error| <= 1.5e-7, about
    one fp32 ulp of erf's range, and vectorised (math.erf per element is ~10^9 calls here)."""
    z = x.astype(np.float64)
    s, a = np.sign(z), np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return (s * y).astype(np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + erf(x / np.float32(np.sqrt(2.0))))).astype(np.float32)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True, dtype=np.float32)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True, dtype=np.float32)
    return ((x - mu) / np.sqrt(var + np.float32(EPS)) * w + b).astype(np.float32)


def attention(qkv: np.ndarray) -> np.ndarray:
    """Bidirectional multi-head attention over all T rows; qkv is [T, 3*D] fp32."""
    t = qkv.shape[0]
    q = qkv[:, :D].reshape(t, H, HD).transpose(1, 0, 2)
    k = qkv[:, D:2 * D].reshape(t, H, HD).transpose(1, 0, 2)
    v = qkv[:, 2 * D:].reshape(t, H, HD).transpose(1, 0, 2)
    s = (q @ k.transpose(0, 2, 1)) * np.float32(HD ** -0.5)
    s -= s.max(axis=2, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=2, keepdims=True)
    return (p @ v).transpose(1, 0, 2).reshape(t, D).astype(np.float32)


def im2col(x: np.ndarray, stride: int) -> np.ndarray:
    """x is [T_in, C] time-major; returns [T_out, 3*C] with taps (t*s-1, t*s, t*s+1)."""
    t_in, c = x.shape
    xp = np.zeros((t_in + 2, c), np.float32)
    xp[1:-1] = x
    t_out = (t_in - 1) // stride + 1
    idx = np.arange(t_out) * stride            # tap -1 sits at xp[idx], tap 0 at idx+1, ...
    return np.concatenate([xp[idx], xp[idx + 1], xp[idx + 2]], axis=1)


def conv_b(w: np.ndarray) -> np.ndarray:
    """Conv1d weight [C_out, C_in, 3] -> im2col B [3*C_in, C_out], K = tap*C_in + c."""
    return np.ascontiguousarray(w.transpose(2, 1, 0).reshape(-1, w.shape[0]))


def load(model_dir: Path, ops: Ops) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    raw = load_file(str(model_dir / "model.safetensors"))
    f = {k.removeprefix("model."): v.astype(np.float32) for k, v in raw.items()}
    W: dict[str, np.ndarray] = {}
    q = ops.q
    W["conv1.B"] = q(conv_b(f["encoder.conv1.weight"]))
    W["conv1.bias"] = f["encoder.conv1.bias"]
    W["conv2.B"] = q(conv_b(f["encoder.conv2.weight"]))
    W["conv2.bias"] = f["encoder.conv2.bias"]
    W["pos"] = f["encoder.embed_positions.weight"]
    for i in range(NL):
        p = f"encoder.layers.{i}."
        W[f"{i}.qkv.B"] = q(np.concatenate([f[p + f"self_attn.{n}_proj.weight"] for n in "qkv"]).T)
        W[f"{i}.qkv.bias"] = np.concatenate([f[p + "self_attn.q_proj.bias"], np.zeros(D, np.float32),
                                             f[p + "self_attn.v_proj.bias"]])
        W[f"{i}.o.B"] = q(f[p + "self_attn.out_proj.weight"].T)
        W[f"{i}.o.bias"] = f[p + "self_attn.out_proj.bias"]
        W[f"{i}.fc1.B"] = q(f[p + "fc1.weight"].T)
        W[f"{i}.fc1.bias"] = f[p + "fc1.bias"]
        W[f"{i}.fc2.B"] = q(f[p + "fc2.weight"].T)
        W[f"{i}.fc2.bias"] = f[p + "fc2.bias"]
        for n in ("self_attn_layer_norm", "final_layer_norm"):
            W[f"{i}.{n}.w"], W[f"{i}.{n}.b"] = f[p + n + ".weight"], f[p + n + ".bias"]
    W["ln.w"], W["ln.b"] = f["encoder.layer_norm.weight"], f["encoder.layer_norm.bias"]
    xk = [f[f"decoder.layers.{l}.encoder_attn.k_proj.weight"] for l in range(NDEC)]
    xv = [f[f"decoder.layers.{l}.encoder_attn.v_proj.weight"] for l in range(NDEC)]
    W["xkv.B"] = q(np.concatenate([m for pair in zip(xk, xv) for m in pair]).T)   # [D, 8*D]
    W["xkv.bias"] = np.concatenate(
        [b for l in range(NDEC) for b in (np.zeros(D, np.float32),
                                          f[f"decoder.layers.{l}.encoder_attn.v_proj.bias"])])
    return W


def stem(W, ops: Ops, mel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h1 = gelu(ops.gemm(im2col(mel.T.astype(np.float32), 1), W["conv1.B"]) + W["conv1.bias"])
    h2 = gelu(ops.gemm(im2col(h1, 2), W["conv2.B"]) + W["conv2.bias"]) + W["pos"]
    return h1, h2.astype(np.float32)


def layer(W, ops: Ops, i: int, x: np.ndarray) -> np.ndarray:
    h = layer_norm(x, W[f"{i}.self_attn_layer_norm.w"], W[f"{i}.self_attn_layer_norm.b"])
    qkv = ops.gemm(h, W[f"{i}.qkv.B"]) + W[f"{i}.qkv.bias"]
    x = x + ops.gemm(attention(qkv), W[f"{i}.o.B"]) + W[f"{i}.o.bias"]
    h = layer_norm(x, W[f"{i}.final_layer_norm.w"], W[f"{i}.final_layer_norm.b"])
    h = gelu(ops.gemm(h, W[f"{i}.fc1.B"]) + W[f"{i}.fc1.bias"])
    return (x + ops.gemm(h, W[f"{i}.fc2.B"]) + W[f"{i}.fc2.bias"]).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--goldens", required=True, type=Path)
    ap.add_argument("--clip", default=None, help="one clip name (default: all in meta.json)")
    ap.add_argument("--exact", action="store_true", help="fp32 operands: checks the decomposition")
    ap.add_argument("--json", type=Path, default=None, help="write the report here")
    ap.add_argument("--save-out", type=Path, default=None,
                    help="write each clip's enc.out to DIR/<clip>.enc_out.npy, for whisper_decode_check.py")
    args = ap.parse_args()

    from safetensors.numpy import load_file
    ops = Ops(args.exact)
    W = load(args.model_dir, ops)
    meta = json.loads((args.goldens / "meta.json").read_text(encoding="utf-8"))
    clips = [args.clip] if args.clip else list(meta["clips"])
    mode = "exact-fp32" if args.exact else "bf16-operands"
    report: dict = {"mode": mode, "clips": {}}

    for name in clips:
        g = load_file(str(args.goldens / f"{name}.safetensors"))
        r: dict = {}
        h1, x = stem(W, ops, g["mel"])
        r["conv1"] = {"cos": cos(h1, g["conv1"]), "rel": rel(h1, g["conv1"])}
        r["conv2"] = {"cos": cos(x, g["conv2"]), "rel": rel(x, g["conv2"])}
        r["layers"] = []
        for i in range(NL):
            x = layer(W, ops, i, x)
            tf = layer(W, ops, i, g[f"enc.hidden.{i}"])
            gold = g[f"enc.hidden.{i + 1}"]
            r["layers"].append({"chained_cos": cos(x, gold), "chained_rel": rel(x, gold),
                                "forced_cos": cos(tf, gold), "forced_rel": rel(tf, gold)})
            print(f"{name} L{i:02d} chained cos {r['layers'][-1]['chained_cos']:.8f} "
                  f"rel {r['layers'][-1]['chained_rel']:.3e}  forced cos "
                  f"{r['layers'][-1]['forced_cos']:.8f}", flush=True)
        out = layer_norm(x, W["ln.w"], W["ln.b"])
        r["enc.out"] = {"cos": cos(out, g["enc.out"]), "rel": rel(out, g["enc.out"])}
        if args.save_out:
            args.save_out.mkdir(parents=True, exist_ok=True)
            np.save(args.save_out / f"{name}.enc_out.npy", out)
        xkv = ops.gemm(out, W["xkv.B"]) + W["xkv.bias"]
        r["cross"] = []
        for l in range(NDEC):
            k, v = xkv[:, 2 * l * D:(2 * l + 1) * D], xkv[:, (2 * l + 1) * D:(2 * l + 2) * D]
            r["cross"].append({"k_cos": cos(k, g[f"dec.{l}.xk"]), "v_cos": cos(v, g[f"dec.{l}.xv"])})
        print(f"{name} [{mode}] conv1 {r['conv1']['cos']:.8f} conv2 {r['conv2']['cos']:.8f} "
              f"enc.out {r['enc.out']['cos']:.8f} (rel {r['enc.out']['rel']:.3e}) "
              f"cross k/v min {min(min(c.values()) for c in r['cross']):.8f}", flush=True)
        report["clips"][name] = r

    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
