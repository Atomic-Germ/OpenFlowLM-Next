"""Localize a dense-layer disagreement: compare each stage's DDR bounce region
in a dumped `act` buffer (make_decode's y_act<l>.bin, layer l, position 0)
with the fp64 replica's intermediate for the same layer.

    python open_kernels/model/dense_probe.py --model-dir DIR --layer 0 --out open_kernels/model/out_q3

Regions (recipes/qwen3.py DenseLayout): xn (bf16), q (f32), kvn (f32 k | v),
og (bf16, after attention), out (f32, o_proj), res (f32), xm (bf16), h (f32),
out2 (f32, down_proj). The layer input is xres<t>.bin (position 0: the
embedding), and the previous layers' outputs come from the NPU run itself,
so a stage's error is judged against the replica fed the NPU's own input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from recipes import dense as QR  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
from q4nx import Q4NX  # noqa: E402
import replica_dense as RD  # noqa: E402


def corr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if not np.isfinite(a).all():
        return float("nan"), "non-finite"
    c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    rel = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
    return c, f"maxrel {rel:.2e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "out_q3"))
    a = ap.parse_args()
    out = Path(a.out)
    md = Path(a.model_dir)
    spec = spec_from_model_dir(md)
    R = QR.recipe(spec)
    L, G = R.layout, R.geo
    m = Q4NX(md / "model.q4nx")
    l, t = a.layer, a.token
    sfx = "" if t == 0 else f"_t{t}"
    act = np.fromfile(out / f"y_act{l}{sfx}.bin", np.uint8)
    # the layer input: the NPU's previous residual (or the embedding at layer 0)
    xin = (np.fromfile(out / f"xres{t}.bin", np.float32) if l == 0
           else np.fromfile(out / f"y_res{l - 1}{sfx}.bin", np.float32)).astype(np.float64)
    pos = t
    pre = f"model.layers.{l}."
    hid, nh, kvh, hd, ff = spec.hidden, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.intermediate

    def region(off, n, dt):
        return act[off:off + n * np.dtype(dt).itemsize].view(dt).astype(np.float64)

    # ---- the replica, stage by stage, from the NPU's own layer input
    x = (RD.rms(xin, spec.norm_eps) * m.bf16(pre + "input_layernorm.weight"))
    print(f"layer {l} position {t}")
    print("  xn   ", corr(region(L.AD_XN, hid, bfloat16), x.astype(np.float32).astype(bfloat16).astype(np.float64)))
    xf = x.astype(np.float32)
    Wq = m.matmul_w(pre + "self_attn.q_proj.weight", nh * hd, hid)
    Wk = m.matmul_w(pre + "self_attn.k_proj.weight", kvh * hd, hid)
    Wv = m.matmul_w(pre + "self_attn.v_proj.weight", kvh * hd, hid)
    q_raw, k_raw, v_raw = xf @ Wq.T, xf @ Wk.T, xf @ Wv.T
    print("  q    ", corr(region(L.AD_Q, nh * hd, np.float32), q_raw))
    print("  k    ", corr(region(L.AD_KVN, kvh * hd, np.float32), k_raw))
    print("  v    ", corr(region(L.AD_KVN + kvh * hd * 4, kvh * hd, np.float32), v_raw))
    qn = m.bf16(pre + "self_attn.q_norm.weight") if spec.qk_norm else None
    kn = m.bf16(pre + "self_attn.k_norm.weight") if spec.qk_norm else None
    inv, rsc = spec.rope_inv_freq(ctx=pos + 1), spec.rope_scale()   # HF's own seq_len = pos + 1 rule
    if spec.qk_norm:
        q = RD.rope((RD.rms(q_raw.reshape(nh, hd), spec.norm_eps) * qn), pos, spec.rotary_dim, spec.rope_theta, inv, rsc)
        k = RD.rope((RD.rms(k_raw.reshape(kvh, hd), spec.norm_eps) * kn), pos, spec.rotary_dim, spec.rope_theta, inv, rsc)
    else:
        q = RD.rope(q_raw.reshape(nh, hd).astype(np.float64), pos, spec.rotary_dim, spec.rope_theta, inv, rsc)
        k = RD.rope(k_raw.reshape(kvh, hd).astype(np.float64), pos, spec.rotary_dim, spec.rope_theta, inv, rsc)
    v = v_raw.reshape(kvh, hd).astype(np.float64)
    # at position 0 the attention output is v of the matching kv head
    if pos == 0:
        og = np.stack([v[h // (nh // kvh)] for h in range(nh)])
    else:
        og = None
    if og is not None:
        print("  og   ", corr(region(L.AD_OG, nh * hd, bfloat16), og.reshape(-1)))
    og_npu = region(L.AD_OG, nh * hd, bfloat16)
    Wo = m.matmul_w(pre + "self_attn.o_proj.weight", hid, nh * hd)
    out_ref = og_npu.astype(np.float32) @ Wo.T
    print("  out  ", corr(region(L.AD_OUT, hid, np.float32), out_ref), "(from the NPU's og)")
    out_npu = region(L.AD_OUT, hid, np.float32)
    res_ref = xin + out_npu
    print("  res  ", corr(region(L.AD_RES, hid, np.float32), res_ref), "(from the NPU's out)")
    res_npu = region(L.AD_RES, hid, np.float32)
    xm_ref = RD.rms(res_npu, spec.norm_eps) * m.bf16(pre + "post_attention_layernorm.weight")
    print("  xm   ", corr(region(L.AD_XM, hid, bfloat16), xm_ref.astype(np.float32).astype(bfloat16).astype(np.float64)), "(from the NPU's res)")
    xm_npu = region(L.AD_XM, hid, bfloat16).astype(np.float32)
    Wup = m.matmul_w(pre + "mlp.up_proj.weight", ff, hid)
    Wg = m.matmul_w(pre + "mlp.gate_proj.weight", ff, hid)
    h_ref = RD.silu(xm_npu @ Wg.T) * (xm_npu @ Wup.T)
    print("  h    ", corr(region(L.AD_H, ff, np.float32), h_ref), "(from the NPU's xm)")
    h_npu = region(L.AD_H, ff, np.float32)
    Wd = m.matmul_w(pre + "mlp.down_proj.weight", hid, ff)
    print("  out2 ", corr(region(L.AD_OUT2, hid, np.float32), h_npu.astype(np.float32) @ Wd.T), "(from the NPU's h)")
    out2_npu = region(L.AD_OUT2, hid, np.float32)
    y = np.fromfile(out / f"y_res{l}{sfx}.bin", np.float32).astype(np.float64)
    print("  xres ", corr(y, res_npu + out2_npu), "(res + out2)")
    print("  layer", corr(y, np.fromfile(out / f"ref_res{l}{sfx}.bin", np.float32)), "(vs the replica end to end)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
