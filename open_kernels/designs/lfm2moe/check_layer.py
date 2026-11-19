"""One LFM2-MoE layer through its mixer (lm_conv or lm_att), the host route and lm_moe (or lm_dense for a
dense-MLP layer) on the NPU, from the GGUF, at position 1 (attention: the streams' compiled position, one cached
row, the new row at row 1; conv: the state holds token 0's bx; slot j = pool expert j), against the fp32 GGUF
reference (model/lfm2moe_gguf_reference.py).

    python designs/lfm2moe/check_layer.py --driver run_kernel --mix BUILD --ffn BUILD --gguf model.gguf \
        --config config.json --ids-file raw.ids --layer 3 --out DIR

Token 0's state (the conv's bx row, or the attention's K / V row) is computed here in fp32 from the GGUF and
stored as the kernel stores it (f32 / bf16). Each stage is reported against the reference (the model's
numbers: the gap includes every bf16 activation boundary; the q4_1 pools are exact), and the FFN again against
the reference fed the NPU's own res / xm and the same experts (the kernels' arithmetic alone). The route runs
on the host exactly as the runtime's hy_route does.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
OK = HERE.parents[1]
sys.path.insert(0, str(OK))
sys.path.insert(0, str(OK / "model"))
from flint_gguf_reference import Gguf, rms, rope, silu  # noqa: E402
from hyv3_gguf_reference import attention  # noqa: E402
from lfm2moe_gguf_reference import forward, rope_inv_freq, route  # noqa: E402
from recipes import lfm2moe as FR  # noqa: E402
from recipes import flint_pack as P  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
from recipes.pack import ptab  # noqa: E402


def stats(name, got, ref):
    got, ref = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
    cos = float(got @ ref / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))
    mx = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
    print(f"  {name:28s} rel L2 {rel:.3e}  cos {cos:.9f}  max|d|/max|ref| {mx:.3e}", flush=True)
    return {"rel_l2": rel, "cos": cos, "max_rel": mx}


def q4(g: Gguf, name: str) -> np.ndarray:
    t = g.tensors[name]
    cols = t["dims"][0]
    rows = int(np.prod(t["dims"][1:]))
    q41 = t["gguf_type"] == 3
    o = g.base + t["offset"]
    return P.blocks(g.mm[o:o + rows * cols // 32 * (20 if q41 else 18)], rows, cols, q41)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("mix", "ffn", "gguf", "config", "ids-file", "out", "driver"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    spec = spec_from_model_dir(Path(a.config).parent)
    R = FR.recipe(spec)
    L, G = R.layout, R.geo
    l, pos = a.layer, 1
    mixer, dense = FR.kinds(spec)[l]
    FF = FR.ffn(spec, dense)
    POOL = L.POOL_BYTES
    g = Gguf(a.gguf)
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()][:2]
    p = f"blk.{l}."
    H, KVH, hd, eps, hid = G.NH, G.KVH, G.HD, G.EPS, G.HID

    # ---- the reference for tokens 0 and 1 through layer l (layers before l in fp32 too)
    x = g.rows("token_embd.weight", ids)
    if l > 0:
        _, _, _, x = forward(g, cfg, ids, layers=l)
    hn = rms(x, g.array(p + "attn_norm.weight"), eps)
    if mixer == FR.MIX_CONV:
        bcu = hn @ g.array(p + "shortconv.in_proj.weight").T
        B, C, u = bcu[:, :hid], bcu[:, hid:2 * hid], bcu[:, 2 * hid:]
        bx = (B * u).astype(np.float32)
        wc = g.array(p + "shortconv.conv.weight")             # [hid, 3]
        conv1 = bx[0] * wc[:, 1] + bx[1] * wc[:, 2]           # t = 1: [0, bx0, bx1]
        y_ref = C[1] * conv1
        o = y_ref @ g.array(p + "shortconv.out_proj.weight").T
        res_ref = x[pos] + o
    else:
        q = rms((hn @ g.array(p + "attn_q.weight").T).reshape(2, H, hd), g.array(p + "attn_q_norm.weight"), eps)
        k = rms((hn @ g.array(p + "attn_k.weight").T).reshape(2, KVH, hd), g.array(p + "attn_k_norm.weight"), eps)
        v = (hn @ g.array(p + "attn_v.weight").T).reshape(2, KVH, hd)
        inv = rope_inv_freq(cfg, hd)
        q, k = rope(q, np.arange(2), inv), rope(k, np.arange(2), inv)
        att = attention(q, k, v)
        o = att.reshape(2, H * hd) @ g.array(p + "attn_output.weight").T
        res_ref = (x + o)[pos]
    xm_ref = rms(res_ref, g.array(p + "ffn_norm.weight"), eps)

    driver = Path(a.driver).resolve()
    a.mix, a.ffn = str(Path(a.mix).resolve()), str(Path(a.ffn).resolve())

    def bf(vec, n):
        o = np.zeros(n, bfloat16)
        o[:len(vec)] = vec
        return o
    pool = np.lib.format.open_memmap(out / "pool.npy", "w+", np.uint8, (POOL,))
    pool_bin = out / "pool.bin"
    consts = np.zeros(L.CD_BYTES, np.uint8)
    for vec, n, off in ((g.array(p + "attn_norm.weight"), hid, L.CD_LNW),
                        (g.array(p + "ffn_norm.weight"), hid, L.CD_POSTLN)):
        b = bf(vec, n).view(np.uint8)
        consts[off:off + b.size] = b
    if mixer == FR.MIX_CONV:
        ip = q4(g, p + "shortconv.in_proj.weight")
        for dst, r0 in ((L.POOL_B, 0), (L.POOL_C, hid), (L.POOL_U, 2 * hid)):
            t = P.matrix(ip, hid, hid, row0=r0)
            pool[dst:dst + t.size] = t
        t = P.matrix(q4(g, p + "shortconv.out_proj.weight"), hid, hid)
        pool[L.POOL_OUT:L.POOL_OUT + t.size] = t
        taps = np.ascontiguousarray(wc.T.astype(np.float32)).view(np.uint8).reshape(-1)   # [3][hid] f32
        consts[L.CD_CONV:L.CD_CONV + taps.size] = taps
        state = np.concatenate([np.zeros(hid, np.float32), bx[0]]).astype(np.float32)
        files = (("consts", consts), ("state", state), ("xres", x[pos].astype(np.float32)))
    else:
        for role, dst, rows, K in (("attn_q", L.POOL_Q, G.QW, hid), ("attn_k", L.POOL_K, G.KVW, hid),
                                   ("attn_v", L.POOL_V, G.KVW, hid), ("attn_output", L.POOL_O, hid, G.QW)):
            t = P.matrix(q4(g, p + role + ".weight"), rows, K)
            pool[dst:dst + t.size] = t
        for vec, n, off in ((g.array(p + "attn_q_norm.weight"), hd, L.CD_META),
                            (g.array(p + "attn_k_norm.weight"), hd, L.CD_META + 2 * hd)):
            b = bf(vec, n).view(np.uint8)
            consts[off:off + b.size] = b
        kv = np.zeros(2 * L.KV_ROW, np.uint8)
        kv[:L.KV_ROW] = np.concatenate([k[0].reshape(-1), v[0].reshape(-1)]).astype(bfloat16).view(np.uint8)
        ptab_b = ptab(L.MAX_CTX, G.ROT, spec.rope_theta, L.PTAB_ROW, spec.rope_inv_freq(), 0, 1.0)
        files = (("consts", consts), ("kv", kv), ("ptab", ptab_b), ("xres", x[pos].astype(np.float32)))
    for name, arr in files:
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
            f.truncate(POOL)
        mm = np.memmap(pool_bin, np.uint8, "r+", shape=(POOL,))
        nz = np.flatnonzero(pool.reshape(-1, 1 << 20).any(axis=1))
        for i in nz:
            mm[i << 20:(i + 1) << 20] = pool[i << 20:(i + 1) << 20]
        mm.flush()

    pool_file()
    head = ["device", f"xclbin A {Path(a.mix) / 'final.xclbin'}", f"kernelx ka A {Path(a.mix) / 'insts.bin'}",
            f"buf pool {POOL} pool.bin", f"buf xres {hid * 4} xres.bin", f"buf consts {L.CD_BYTES} consts.bin"]
    if mixer == FR.MIX_CONV:
        drive("mix", head + [f"buf state {L.STATE_BYTES} state.bin", f"buf act {L.AD_BYTES}",
                             "run ka pool xres consts state act", f"dump act act_mix.bin {L.AD_BYTES}",
                             f"dump state state_mix.bin {L.STATE_BYTES}"])
    else:
        drive("mix", head + [f"buf kv {L.KV_BYTES} kv.bin", f"buf act {L.AD_BYTES}",
                             f"buf ptab {L.PTAB_BYTES} ptab.bin", "run ka pool xres consts kv act ptab",
                             f"dump act act_mix.bin {L.AD_BYTES}", f"dump kv kv_mix.bin {2 * L.KV_ROW}"])
    actb = np.fromfile(out / "act_mix.bin", np.uint8)
    res = actb[L.AD_RES:L.AD_RES + hid * 4].view(np.float32).copy()
    xm = actb[L.AD_XM:L.AD_XM + hid * 2].view(bfloat16).astype(np.float32)
    rep = {"layer": l, "position": pos, "ids": ids, "mixer": mixer, "dense": dense}
    print(f"mixer half, layer {l} ({mixer}{', dense' if dense else ''}) (vs fp32 reference):", flush=True)
    if mixer == FR.MIX_CONV:
        f32 = lambda off: actb[off:off + hid * 4].view(np.float32)
        rep["b"] = stats("B", f32(L.AD_B), B[1])
        rep["c"] = stats("C", f32(L.AD_C), C[1])
        rep["u"] = stats("u", f32(L.AD_U), u[1])
        yb = actb[L.AD_Y:L.AD_Y + hid * 2].view(bfloat16).astype(np.float32)
        rep["y"] = stats("y = C conv (bf16)", yb, y_ref)
        st = np.fromfile(out / "state_mix.bin", np.float32)
        rep["state0"] = stats("new state0 (= bx0)", st[:hid], bx[0])
        rep["state1"] = stats("new state1 (= bx1)", st[hid:], bx[1])
    else:
        krow = np.fromfile(out / "kv_mix.bin", np.uint8)[L.KV_ROW:].view(bfloat16).astype(np.float32)
        rep["k"] = stats("new K row", krow[:G.KVW], k[1].reshape(-1))
        rep["v"] = stats("new V row", krow[G.KVW:], v[1].reshape(-1))
        og = actb[L.AD_OG:L.AD_OG + G.QW * 2].view(bfloat16).astype(np.float32)
        rep["og"] = stats("attention out (og)", og, att[pos].reshape(-1))
    rep["res"] = stats("res = x + o", res, res_ref)
    rep["xm"] = stats("xm", xm, xm_ref)

    # ---- the route (hy_route's math) from the NPU's xm, and the slots' weights into the pool
    ff = FF.FF
    if dense:
        nslot = FF.SLOTS
        slots = [("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", s) for s in range(nslot)]
        w = np.ones(nslot, np.float32)
        top = ref_top = np.arange(nslot)
    else:
        wr, bias = g.array(p + "ffn_gate_inp.weight"), g.array(p + "exp_probs_b.bias")
        top, w = route(xm[None], wr, bias, G.TOPK, FR.router_scale(spec))
        ref_top, ref_w = route(xm_ref[None], wr, bias, G.TOPK, FR.router_scale(spec))
        top, w, ref_top, ref_w = top[0], w[0], ref_top[0], ref_w[0]
        rep["experts"], rep["ref_experts"], rep["weights"] = top.tolist(), ref_top.tolist(), w.tolist()
        print(f"route: experts {top.tolist()} (reference {ref_top.tolist()}), weights {np.round(w, 4).tolist()}")
        slots = [("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", int(e)) for e in top]
        nslot = len(slots)
        wv = np.zeros(FR.WV, np.float32)
        wv[:nslot] = w
        actb[L.AD_W:L.AD_W + 4 * FR.WV] = wv.view(np.uint8)
    actb.tofile(out / "act_route.bin")
    blk = {}
    for nm in {s[i] for s in slots for i in range(3)}:
        blk[nm] = q4(g, p + nm)
    dw = {}
    for s_, (gn, un, dn, e) in enumerate(slots):
        r0 = e * ff
        ug = P.slot_ug(P.matrix(blk[gn], ff, hid, row0=r0), P.matrix(blk[un], ff, hid, row0=r0), hid)
        pool[s_ * FF.EXPERT_UG:s_ * FF.EXPERT_UG + ug.size] = ug
        if dense:       # down's columns [e ff, e ff + ff) of the [HID, DFF] matrix
            down = P.matrix(blk[dn], hid, ff, col0=e * ff)
            Wg, Wu = g.array(p + gn)[r0:r0 + ff], g.array(p + un)[r0:r0 + ff]
            Wd = g.array(p + dn)[:, e * ff:(e + 1) * ff]
        else:           # down's rows [e HID, e HID + HID) of the [E HID, ff] flattened expert tensor
            down = P.matrix(blk[dn], hid, ff, row0=e * hid)
            Wg, Wu, Wd = g.expert(p + gn, e), g.expert(p + un, e), g.expert(p + dn, e)
        o0 = FF.POOL_DOWN + s_ * FF.UP_BYTES
        pool[o0:o0 + down.size] = down
        dw[s_] = (Wg, Wu, Wd)
    pool_file()
    drive("ffn", ["device", f"xclbin M {Path(a.ffn) / 'final.xclbin'}", f"kernelx km M {Path(a.ffn) / 'insts.bin'}",
                  f"buf pool {POOL} pool.bin", f"buf xres {hid * 4} xres.bin",
                  f"buf act {L.AD_BYTES} act_route.bin", "run km pool xres act",
                  f"dump xres xres_ffn.bin {hid * 4}", f"dump act act_ffn.bin {L.AD_BYTES}"])
    y = np.fromfile(out / "xres_ffn.bin", np.float32)
    h = np.fromfile(out / "act_ffn.bin", np.uint8)[L.AD_H:L.AD_H + nslot * FF.H_STRIDE].view(np.float32)
    h = h.reshape(nslot, FF.H_STRIDE // 4)[:, :ff]

    def ffn(res_in, xm_in, wts):
        acc = np.zeros(hid, np.float64)
        hs = []
        for s_ in range(nslot):
            Wg, Wu, Wd = dw[s_]
            hh = wts[s_] * silu(Wg @ xm_in) * (Wu @ xm_in)
            hs.append(hh)
            acc += Wd.astype(np.float64) @ hh.astype(np.float64)
        return res_in + acc, hs
    y_cond, h_cond = ffn(res, xm, w)
    print("FFN half (vs the reference fed the NPU's res / xm and the same slots):", flush=True)
    for s_ in range(nslot):
        if dense and s_ not in (0, nslot - 1):
            continue
        rep[f"h{s_}"] = stats(f"h_{s_} ({slots[s_][0].split('.')[0]} {slots[s_][3]})", h[s_], h_cond[s_])
    rep["h_all"] = stats("h (every slot)", h, np.stack(h_cond))
    rep["out_cond"] = stats("layer out", y, y_cond)
    if dense or list(top) == list(ref_top):
        y_ref, _ = ffn(res_ref, xm_ref, w if dense else ref_w)
        print("FFN half (vs the fp32 reference):", flush=True)
        rep["out"] = stats("layer out", y, y_ref)
    else:
        print("FFN half: the NPU's route differs from the reference's; no fp32 comparison of the layer out")
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
