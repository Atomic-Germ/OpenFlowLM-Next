"""One GPT-OSS layer through gx_att, the host route and gx_moe on the NPU, from the GGUF, at the
position the streams were compiled for (1: one cached row, the new row at row 1; slot j = expert j),
against the fp32 GGUF reference (model/gptoss_gguf_reference.py).

    python designs/gptoss/check_layer.py --driver run_kernel --att BUILD --moe BUILD --gguf model.gguf \
        --config config.json --ids-file raw.ids --layer 0 --out DIR

The cached row 0 is token 0's K / V, computed here in fp32 from the GGUF and rounded to bf16 as
the kernel stores it. Each stage is reported twice: against the reference (the model's numbers:
the gap includes the q8 pools' bf16 scale and every bf16 / int16 activation boundary), and the
MoE again against the reference fed the NPU's own res / xm and the same experts (the kernels'
arithmetic alone). The route runs on the host exactly as core.cpp's gptoss_route does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
OK = HERE.parents[1]
sys.path.insert(0, str(OK))
sys.path.insert(0, str(OK / "model"))
from gptoss_gguf_reference import Gguf, attention, clamped_swiglu, rms, rope, yarn_inv_freq  # noqa: E402
from recipes import gptoss as GR  # noqa: E402
from recipes import gptoss_pack as P  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
from recipes.pack import ptab  # noqa: E402


def stats(name, got, ref):
    got, ref = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
    cos = float(got @ ref / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))
    mx = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
    print(f"  {name:28s} rel L2 {rel:.3e}  cos {cos:.9f}  max|d|/max|ref| {mx:.3e}", flush=True)
    return {"rel_l2": rel, "cos": cos, "max_rel": mx}


def q8(g, name):
    t = g.tensors[name]
    cols, rows = t["dims"]
    return P.q8_0_codes(g.raw(name, rows * cols // 32 * 34), rows, cols)


def expert_codes(g, name, e, read=P.mxfp4_codes):
    t = g.tensors[name]
    cols, rows, _ = t["dims"]
    per = rows * cols // 32 * 17
    o = g.base + t["offset"] + e * per
    return read(np.asarray(g.mm[o:o + per]), rows, cols)


def expert_tiles(g, name, e, rows, K, bias=None):
    """One expert's projection as the recipe's experts are held: mx4 half-tiles, or q8 ones."""
    if GR.mx4_on():
        return P.mx4_tiles(*expert_codes(g, name, e, P.mxfp4_nibbles), rows, K, bias)
    return P.q8_tiles(*expert_codes(g, name, e), rows, K, bias)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("att", "moe", "gguf", "config", "ids-file", "out", "driver"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    spec = spec_from_model_dir(Path(a.config).parent)
    R = GR.recipe(spec)
    L, G = R.layout, R.geo
    g = Gguf(a.gguf)
    ids = [int(t) for t in Path(a.ids_file).read_text().split()][:2]
    l, pos = a.layer, 1
    p = f"blk.{l}."
    H, KVH, hd, eps = G.NH, G.KVH, G.HD, G.EPS
    local = spec.layer_types[l] == "dense_local"
    inv, conc = yarn_inv_freq(cfg)

    # ---- the reference for tokens 0 and 1 through layer l (layers before l in fp32 too)
    x = g.rows("token_embd.weight", ids)
    from gptoss_gguf_reference import forward
    if l > 0:
        _, _, _, x = forward(g, dict(cfg, num_hidden_layers=l), ids)
    hn = rms(x, g.array(p + "attn_norm.weight"), eps)
    q = rope((hn @ g.array(p + "attn_q.weight").T + g.array(p + "attn_q.bias")).reshape(2, H, hd), np.arange(2), inv, conc)
    k = rope((hn @ g.array(p + "attn_k.weight").T + g.array(p + "attn_k.bias")).reshape(2, KVH, hd), np.arange(2), inv, conc)
    v = (hn @ g.array(p + "attn_v.weight").T + g.array(p + "attn_v.bias")).reshape(2, KVH, hd)
    att = attention(q, k, v, g.array(p + "attn_sinks.weight"), G.WINDOW if local else 0)
    o = att.reshape(2, H * hd) @ g.array(p + "attn_output.weight").T + g.array(p + "attn_output.bias")
    res_ref = (x + o)[pos]
    xm_ref = rms(res_ref, g.array(p + "post_attention_norm.weight"), eps)

    # ---- device buffers, as files run_kernel loads (the pool sparse: attention + the 4 routed slots)
    import subprocess
    driver = Path(a.driver).resolve()
    a.att, a.moe = str(Path(a.att).resolve()), str(Path(a.moe).resolve())

    def bf(vec, n):
        o = np.zeros(n, bfloat16)
        o[:len(vec)] = vec
        return o
    pool = np.lib.format.open_memmap(out / "pool.npy", "w+", np.uint8, (L.POOL_BYTES,))
    pool_bin = out / "pool.bin"
    for role, dst, rows, K in (("attn_q", L.POOL_Q, G.QW, G.HID), ("attn_k", L.POOL_K, G.KVW, G.HID),
                               ("attn_v", L.POOL_V, G.KVW, G.HID), ("attn_output", L.POOL_O, G.HID, G.QW)):
        c, s = q8(g, p + role + ".weight")
        t = P.q8_tiles(c, s, rows, K, g.array(p + role + ".bias")).reshape(-1)
        pool[dst:dst + t.size] = t
    consts = np.zeros(L.CD_BYTES, np.uint8)
    for vec, n, off in ((g.array(p + "attn_norm.weight"), G.HID, L.CD_LNW),
                        (g.array(p + "post_attention_norm.weight"), G.HID, L.CD_POSTLN),
                        (g.array(p + "attn_sinks.weight"), G.NH, L.CD_SINK)):
        b = bf(vec, n).view(np.uint8)
        consts[off:off + b.size] = b
    kv = np.zeros(2 * L.KV_ROW, np.uint8)
    kv[:L.KV_ROW] = np.concatenate([k[0].reshape(-1), v[0].reshape(-1)]).astype(bfloat16).view(np.uint8)
    ptab_b = ptab(L.MAX_CTX, G.ROT, spec.rope_theta, L.PTAB_ROW, spec.rope_inv_freq(),
                  G.WINDOW if local else 0, spec.rope_scale())
    xin = np.zeros(G.HID, np.float32)
    xin[:G.HR] = x[pos]
    for name, arr in (("consts", consts), ("kv", kv), ("ptab", ptab_b), ("xres", xin)):
        (out / f"{name}.bin").write_bytes(np.ascontiguousarray(arr).tobytes())

    def drive(name, lines):
        cfg_path = out / f"{name}.cfg"
        cfg_path.write_text("\n".join(lines) + "\n")
        r = subprocess.run([str(driver), str(cfg_path)], capture_output=True, text=True)
        print(r.stdout.strip(), flush=True)
        if r.returncode:
            raise RuntimeError(f"{name}: {r.stdout}\n{r.stderr}")

    def pool_file():
        pool.flush()
        with open(pool_bin, "wb") as f:                       # the .npy header off, sparse where zero
            f.truncate(L.POOL_BYTES)
            mm = np.memmap(pool_bin, np.uint8, "r+", shape=(L.POOL_BYTES,))
            nz = np.flatnonzero(pool.reshape(-1, 1 << 20).any(axis=1))
            for i in nz:
                mm[i << 20:(i + 1) << 20] = pool[i << 20:(i + 1) << 20]
            mm.flush()

    pool_file()
    drive("att", ["device", f"xclbin A {Path(a.att) / 'final.xclbin'}", f"kernelx ka A {Path(a.att) / 'insts.bin'}",
                  f"buf pool {L.POOL_BYTES} pool.bin", f"buf xres {G.HID * 4} xres.bin", f"buf consts {L.CD_BYTES} consts.bin",
                  f"buf kv {L.KV_BYTES} kv.bin", f"buf act {L.AD_BYTES}", f"buf ptab {L.PTAB_BYTES} ptab.bin",
                  "run ka pool xres consts kv act ptab", f"dump act act_att.bin {L.AD_BYTES}",
                  f"dump kv kv_att.bin {2 * L.KV_ROW}"])
    actb = np.fromfile(out / "act_att.bin", np.uint8)
    res = actb[L.AD_RES:L.AD_RES + G.HID * 4].view(np.float32).copy()
    xm = actb[L.AD_XM:L.AD_XM + G.HID * 2].view(bfloat16).astype(np.float32)
    krow = np.fromfile(out / "kv_att.bin", np.uint8)[L.KV_ROW:].view(bfloat16).astype(np.float32)
    rep = {"layer": l, "position": pos, "ids": ids, "local": local}
    rep["pad_zero"] = bool(not res[G.HR:].any() and not xm[G.HR:].any())
    print("attention half (vs fp32 reference):", flush=True)
    rep["k"] = stats("new K row", krow[:G.KVW], k[1].reshape(-1))
    rep["v"] = stats("new V row", krow[G.KVW:], v[1].reshape(-1))
    og = actb[L.AD_OG:L.AD_OG + G.QW * 2].view(bfloat16).astype(np.float32)
    rep["og"] = stats("attention out (og)", og, att[pos].reshape(-1))
    rep["res"] = stats("res = x + o", res[:G.HR], res_ref)
    rep["xm"] = stats("xm", xm[:G.HR], xm_ref)

    # ---- the host route, from the NPU's xm (core.cpp gptoss_route)
    logit = g.array(p + "ffn_gate_inp.weight") @ xm[:G.HR] + g.array(p + "ffn_gate_inp.bias")
    top = np.argsort(-logit, kind="stable")[:G.TOPK]
    w = np.exp(logit[top] - logit[top].max())
    w = (w / w.sum()).astype(np.float32)
    ref_logit = g.array(p + "ffn_gate_inp.weight") @ xm_ref + g.array(p + "ffn_gate_inp.bias")
    ref_top = np.argsort(-ref_logit, kind="stable")[:G.TOPK]
    rep["experts"], rep["ref_experts"], rep["weights"] = top.tolist(), ref_top.tolist(), w.tolist()
    print(f"route: experts {top.tolist()} (reference {ref_top.tolist()}), weights {np.round(w, 4).tolist()}")
    bd = g.array(p + "ffn_down_exps.bias")
    bsum = np.zeros(G.HID, np.float32)
    bsum[:G.HR] = (w[:, None] * bd[top]).sum(0)
    wv = np.zeros(8, np.float32)
    wv[:G.TOPK] = w
    actb[L.AD_W:L.AD_W + 32] = wv.view(np.uint8)
    actb[L.AD_BSUM:L.AD_BSUM + G.HID * 4] = bsum.view(np.uint8)
    actb.tofile(out / "act_route.bin")
    bu, bg = g.array(p + "ffn_up_exps.bias"), g.array(p + "ffn_gate_exps.bias")
    dense = {}
    for s_, e in enumerate(top):
        e = int(e)
        up = expert_tiles(g, p + "ffn_up_exps.weight", e, G.FF, G.HID, bu[e])
        gate = expert_tiles(g, p + "ffn_gate_exps.weight", e, G.FF, G.HID, bg[e])
        ug = np.stack([up, gate], axis=1).reshape(-1)          # P.expert_ug's [band][up | gate], either form
        pool[s_ * L.EXPERT_UG:s_ * L.EXPERT_UG + ug.size] = ug
        dn = expert_tiles(g, p + "ffn_down_exps.weight", e, G.HID, G.FF).reshape(-1)
        o0 = L.POOL_DOWN + s_ * L.UP_BYTES
        pool[o0:o0 + dn.size] = dn
        dense[e] = (g.expert(p + "ffn_up_exps.weight", e), g.expert(p + "ffn_gate_exps.weight", e),
                    g.expert(p + "ffn_down_exps.weight", e))
    pool_file()
    drive("moe", ["device", f"xclbin M {Path(a.moe) / 'final.xclbin'}", f"kernelx km M {Path(a.moe) / 'insts.bin'}",
                  f"buf pool {L.POOL_BYTES} pool.bin", f"buf xres {G.HID * 4} xres.bin",
                  f"buf act {L.AD_BYTES} act_route.bin", "run km pool xres act",
                  f"dump xres xres_moe.bin {G.HID * 4}", f"dump act act_moe.bin {L.AD_BYTES}"])
    y = np.fromfile(out / "xres_moe.bin", np.float32)
    h = np.fromfile(out / "act_moe.bin", np.uint8)[L.AD_H:L.AD_H + G.TOPK * G.FF * 4].view(np.float32).reshape(G.TOPK, G.FF)

    def moe(res_in, xm_in, experts, weights):
        acc = np.zeros(G.HR, np.float64)
        hs = []
        for s, e in enumerate(experts):
            Wu, Wg, Wd = dense[int(e)] if int(e) in dense else (g.expert(p + "ffn_up_exps.weight", int(e)),
                                                                 g.expert(p + "ffn_gate_exps.weight", int(e)),
                                                                 g.expert(p + "ffn_down_exps.weight", int(e)))
            hh = weights[s] * clamped_swiglu(Wg @ xm_in + bg[e], Wu @ xm_in + bu[e])
            hs.append(hh)
            acc += Wd @ hh.astype(np.float64) + weights[s] * bd[e]
        return res_in + acc, hs
    y_cond, h_cond = moe(res[:G.HR], xm[:G.HR], top, w)
    rw = np.exp(ref_logit[ref_top] - ref_logit[ref_top].max())
    y_ref, _ = moe(res_ref, xm_ref, ref_top, (rw / rw.sum()).astype(np.float32))
    rep["pad_zero"] = rep["pad_zero"] and bool(not y[G.HR:].any())
    print("MoE half (vs the reference fed the NPU's res / xm and the same experts):", flush=True)
    for s in range(G.TOPK):
        rep[f"h{s}"] = stats(f"h_{s} (expert {top[s]})", h[s, :G.HR], h_cond[s])
    rep["out_cond"] = stats("layer out", y[:G.HR], y_cond)
    print("MoE half (vs the fp32 reference):", flush=True)
    rep["out"] = stats("layer out", y[:G.HR], y_ref)
    print(f"padding stays zero: {rep['pad_zero']}")
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
