"""fp64 draft pass of one K2 layer (OPEN-UNO-LORA) from the padded uno.q4nx tensors the NPU reads, so only arithmetic differs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "model"))
from q4nx import Q4NX  # noqa: E402
from recipes import dxl as DXR  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes.load import load_spec  # noqa: E402
from recipes.pack import apply_op  # noqa: E402
from replica_dense import rms, rope, silu  # noqa: E402


def layer_ref(m, u, spec, layer, xres, mask):
    """xres [L, hid] -> the layer's output rows, row j at position j."""
    G = for_spec(spec).recipe(spec).geo
    hid, nh, kvh, hd, ff, qw, kvw = spec.hidden, spec.num_heads, spec.num_kv_heads, spec.head_dim, \
        spec.intermediate, G.QW, G.KVW
    eps, ng, inv = spec.norm_eps, spec.norm_groups, spec.rope_inv_freq()
    pre, upre = f"model.layers.{layer}.", f"model.layers.{layer}.uno."
    W = {p: m.matmul_w(pre + n, r, c) for p, (n, r, c) in {
        "q": ("self_attn.q_proj.weight", qw, hid), "k": ("self_attn.k_proj.weight", kvw, hid),
        "v": ("self_attn.v_proj.weight", kvw, hid), "o": ("self_attn.o_proj.weight", hid, qw),
        "g": ("mlp.gate_proj.weight", ff, hid), "u": ("mlp.up_proj.weight", ff, hid),
        "d": ("mlp.down_proj.weight", hid, ff)}.items()}
    shapes = DXR.lora_shapes(spec, G.N_CORES)
    U = {n: u.matmul_w(upre + n + ".weight", *shapes[n]) for n in shapes}
    ln1, ln2 = m.bf16(pre + "input_layernorm.weight"), m.bf16(pre + "post_attention_layernorm.weight")
    L = len(xres)
    K, V, out = np.zeros((L, kvh, hd)), np.zeros((L, kvh, hd)), []
    for j in range(L):
        x = (rms(xres[j], eps, ng) * ln1).astype(np.float32).astype(np.float64)
        z = mask[j] * (U["a_qkv"] @ x)
        q = W["q"] @ x + U["b_q"] @ z[:256]
        k = W["k"] @ x + U["b_k"] @ z[:256]
        v = W["v"] @ x + U["b_v"] @ z[256:512]
        q = rope(q.reshape(nh, hd), j, hd, spec.rope_theta, inv)
        K[j] = rope(k.reshape(kvh, hd), j, hd, spec.rope_theta, inv)
        V[j] = v.reshape(kvh, hd)
        og = np.zeros((nh, hd))
        for h in range(nh):
            kv = h // (nh // kvh)
            s = K[:j + 1, kv] @ q[h] / np.sqrt(hd)
            p = np.exp(s - s.max())
            og[h] = (p / p.sum()) @ V[:j + 1, kv]
        og = og.reshape(-1)
        zo = mask[j] * (U["a_o"] @ og)
        res = xres[j] + W["o"] @ og + U["b_o"] @ zo[:256]
        xm = (rms(res, eps, ng) * ln2).astype(np.float32).astype(np.float64)
        zg = mask[j] * (U["a_gu"] @ xm)
        g = W["g"] @ xm + U["b_g"] @ zg[:256]
        up = W["u"] @ xm + U["b_u"] @ zg[:256]
        hh = silu(g) * up
        zd = mask[j] * (U["a_d"] @ hh)
        out.append(res + W["d"] @ hh + U["b_d"] @ zd[:256])
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--model", required=True, help="the K2 model dir (model.q4nx + uno.q4nx)")
    ap.add_argument("--l", type=int, default=4)
    ap.add_argument("--spec", default=str(HERE.parents[1] / "recipes" / "specs" / "k2-horizon-7b.json"))
    ap.add_argument("--pack", action="store_true", help="write layer 0's LoRA pool as <fixture>/lora_L0.bin")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    fx, md = Path(a.fixture).resolve(), Path(a.model)
    spec = load_spec(Path(a.spec))
    X = DXR.layout(spec, a.l)
    u = Q4NX(md / "uno.q4nx")
    if a.pack:
        pool = np.zeros(X.LORA_BYTES, np.uint8)
        for op in DXR.lora_pack_plan(spec, a.l):
            apply_op(op, u, 0, pool)
        (fx / "lora_L0.bin").write_bytes(pool.tobytes())
        print(f"wrote {fx / 'lora_L0.bin'} ({pool.nbytes} B)")
    if a.compare:
        hid = spec.hidden
        xres = np.stack([np.fromfile(fx / f"xres{j}.bin", np.float32)[:hid].astype(np.float64) for j in range(a.l)])
        mask = [0.0] + [1.0] * (a.l - 1)
        ref = layer_ref(Q4NX(md / "model.q4nx"), u, spec, 0, xres, mask)
        base = layer_ref(Q4NX(md / "model.q4nx"), u, spec, 0, xres, [0.0] * a.l)
        y = np.fromfile(fx / "y_dxl_res.bin", np.float32).reshape(a.l, hid).astype(np.float64)
        ok = True
        for j in range(a.l):
            corr = float(np.corrcoef(y[j], ref[j])[0, 1])
            rel = float(np.abs(y[j] - ref[j]).max() / (np.abs(ref[j]).max() + 1e-30))
            lora = float(np.abs(ref[j] - base[j]).max() / (np.abs(ref[j]).max() + 1e-30))
            good = corr > 0.9999
            ok &= good
            print(f"row {j}: {'PASS' if good else 'FAIL'} vs fp64 draft: corr {corr:.6f} maxrel {rel:.2e} "
                  f"(the LoRA moves this row by {lora:.2e})")
        if a.l:
            dx = np.fromfile(fx / "y_res0.bin", np.float32)
            same = int((dx.view(np.uint32) != y[0].astype(np.float32).view(np.uint32)).sum())
            ok &= same == 0
            print(f"row 0 (the seed, LoRA off) vs dx: {same} of {hid} differ")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
