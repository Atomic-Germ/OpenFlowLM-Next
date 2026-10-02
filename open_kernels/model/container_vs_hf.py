r"""Check a converted Qwen3.5 / 3.8 container against the original HF safetensors, tensor by
tensor: dequantize what the container holds (as the NPU pool will see it) and correlate it
with the source weight after the transform the converter is expected to have applied.

A slice compare cannot catch a conversion bug -- the kernels and replica_qwen35.py read the
same container bytes and agree on them. This is the check that the bytes are the model.

    python open_kernels/model/container_vs_hf.py --model-dir DIR --hf-shard model-00001-of-000NN.safetensors
        [--layers 0,3] [--prefix model.language_model.]

Only the layers whose tensors are in the given shard are compared (pass the shard that
holds them; the HF index says which).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from q4nx import Q4NX  # noqa: E402


class Shard:
    """The bf16 / f32 tensors of one safetensors file, read without torch."""

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n

    def has(self, name):
        return name in self.h

    def get(self, name) -> np.ndarray:
        t = self.h[name]
        o0, o1 = t["data_offsets"]
        self.f.seek(self.base + o0)
        b = self.f.read(o1 - o0)
        if t["dtype"] == "BF16":
            a = (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif t["dtype"] == "F32":
            a = np.frombuffer(b, np.float32)
        else:
            raise ValueError(f"{name}: dtype {t['dtype']}")
        return a.reshape(t["shape"]).astype(np.float64)


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--hf-shard", required=True)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--prefix", default="model.language_model.")
    a = ap.parse_args()
    q = Q4NX(Path(a.model_dir) / "model.q4nx")
    hf = Shard(Path(a.hf_shard))
    cfg = json.loads((Path(a.model_dir) / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    nh, hd = t["num_attention_heads"], t["head_dim"]

    def ours(name, shape):
        """The container tensor as the NPU sees it: quantized ones dequantized (q8 re-quantized
        to q4_1 by default, as the pool holds it), bf16 / f32 as stored."""
        d = q.tensors[name]["dtype"]
        if d == "BF16":
            return q.bf16(name).astype(np.float64)
        if d == "F32":
            return q.f32(name).astype(np.float64)
        return q.matmul_w(name, *shape).astype(np.float64)

    bad = 0
    for l in map(int, a.layers.split(",")):
        c, h = f"model.layers.{l}.", f"{a.prefix}layers.{l}."
        if not hf.has(h + "input_layernorm.weight"):
            print(f"layer {l}: not in this shard")
            continue
        rows = []
        # (container name, HF name, expected transform of the HF tensor)
        pairs = [("input_layernorm.weight", "input_layernorm.weight", lambda w: 1 + w),
                 ("post_attention_layernorm.weight", "post_attention_layernorm.weight", lambda w: 1 + w),
                 ("mlp.up_proj.weight", "mlp.up_proj.weight", None),
                 ("mlp.gate_proj.weight", "mlp.gate_proj.weight", None),
                 ("mlp.down_proj.weight", "mlp.down_proj.weight", None)]
        if hf.has(h + "linear_attn.in_proj_qkv.weight"):
            pairs += [("linear_attn.qkv_proj.weight", "linear_attn.in_proj_qkv.weight", None),
                      ("self_attn.gate_proj.weight", "linear_attn.in_proj_z.weight", None),
                      ("linear_attn.ssm_out_proj.weight", "linear_attn.out_proj.weight", None),
                      ("linear_attn.ssm_alpha_proj.bf16.weight", "linear_attn.in_proj_a.weight", None),
                      ("linear_attn.ssm_beta_proj.bf16.weight", "linear_attn.in_proj_b.weight", None),
                      ("linear_attn.ssm_a", "linear_attn.A_log", lambda w: -np.exp(w)),
                      ("linear_attn.ssm_dt.bias", "linear_attn.dt_bias", None),
                      # HF [channels, 1, taps]; the container holds it transposed, [taps, channels]
                      ("linear_attn.ssm_conv1d.weight", "linear_attn.conv1d.weight",
                       lambda w: w.reshape(w.shape[0], -1).T),
                      ("linear_attn.ssm_norm.weight", "linear_attn.norm.weight", None)]
        else:
            def deinterleave(w):
                # HF: q_proj rows are [q_h | gate_h] per head; the container holds [all q | all gate]
                w = w.reshape(nh, 2, hd, -1)
                return np.concatenate([w[:, 0].reshape(nh * hd, -1), w[:, 1].reshape(nh * hd, -1)])
            pairs += [("self_attn.q_proj.weight", "self_attn.q_proj.weight", deinterleave),
                      ("self_attn.k_proj.weight", "self_attn.k_proj.weight", None),
                      ("self_attn.v_proj.weight", "self_attn.v_proj.weight", None),
                      ("self_attn.o_proj.weight", "self_attn.o_proj.weight", None),
                      ("self_attn.q_norm.weight", "self_attn.q_norm.weight", lambda w: 1 + w),
                      ("self_attn.k_norm.weight", "self_attn.k_norm.weight", lambda w: 1 + w)]
        for cn, hn, tf in pairs:
            w = hf.get(h + hn)
            want = tf(w) if tf else w
            want = want.reshape(want.shape[0], -1) if want.ndim > 1 else want
            got = ours(c + cn, want.shape if want.ndim == 2 else (want.size,))
            got = got.reshape(want.shape)
            r = corr(got, want)
            raw = corr(got, w.reshape(got.shape)) if tf is not None and w.size == got.size else float("nan")
            flag = "" if r > 0.99 else "   <-- MISMATCH"
            bad += r <= 0.99
            rows.append(f"  {cn:42s} {str(want.shape):16s} corr {r:9.6f}  (untransformed {raw:9.6f}){flag}")
        print(f"layer {l} ({'linear' if hf.has(h + 'linear_attn.in_proj_qkv.weight') else 'full'}):")
        print("\n".join(rows))
    print("ALL MATCH" if not bad else f"{bad} MISMATCH(ES)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
