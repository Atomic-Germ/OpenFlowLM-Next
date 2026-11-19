"""One LFM3 layer through its mixer (lg_conv or lg_att) and lg_mlp on the NPU, from the (derived) GGUF, at
position 1 (attention: the streams' compiled position, one cached row, the new row at row 1; conv: the state
holds token 0's bx; the sliding window is wider than two rows), against the fp32 GGUF reference
(model/lfm3_gguf_reference.py).

    python designs/lfm3/check_layer.py --driver run_kernel --conv BUILD --att BUILD --mlp BUILD \
        --model-dir DIR --ids-file raw.ids --layer 3 --out DIR

Token 0's state (the conv's bx row, or the attention's K / V row) is computed here in fp32 from the GGUF and stored
as the kernel stores it (f32 / bf16). Each stage is reported against the reference (the model's numbers: the gap
includes every bf16 activation boundary; the q4_1 pools are exact), and the MLP half again against the reference
fed the NPU's own res / xm (the kernels' arithmetic alone).
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
from flint_gguf_reference import rms, rope, silu, attention  # noqa: E402
from lfm3_gguf_reference import Gguf3, forward, inv_freq  # noqa: E402
from recipes import lfm3 as FR  # noqa: E402
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


def law(a: np.ndarray) -> np.ndarray:
    """q4_1-law chunk bytes -> the pool's law: unchanged, or (LFM3_Q40) the Q4_0-native 4608-B chunks (the m plane,
    -8 d, dropped; fl_q4's Q4_0-native output)."""
    if not FR.q40_on():
        return a
    c = np.asarray(a).reshape(-1, 5120)
    return np.concatenate([c[:, :512], c[:, 1024:]], axis=1).reshape(-1)


def MAT(*a, **k):
    return law(P.matrix(*a, **k))


def UG(gate, up, hid):
    return law(P.slot_ug(gate, up, hid))


def q4(g, name: str) -> np.ndarray:
    t = g.tensors[name]
    cols = t["dims"][0]
    rows = int(np.prod(t["dims"][1:]))
    o = g.base + t["offset"]
    return P.blocks(g.mm[o:o + rows * cols // 32 * 18], rows, cols, False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("conv", "att", "model-dir", "ids-file", "out", "driver"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--mlp", default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--shared", action="store_true",
                    help="--conv / --att are the two streams of ONE image (lg_shared_conv / _att): the conv build's xclbin "
                         "is loaded for both, every dispatch takes the 6-buffer signature (implies --fused)")
    ap.add_argument("--fused", action="store_true",
                    help="--conv / --att are the one-dispatch convmlp / attmlp builds (the MLP runs inside them)")
    a = ap.parse_args()
    a.fused = a.fused or a.shared
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.model_dir)
    cfg = json.loads((mdir / "config.json").read_text())
    spec = spec_from_model_dir(mdir)
    R = FR.recipe(spec)
    L, G = R.layout, R.geo
    l, pos = a.layer, 1
    lt = spec.layer_types[l]
    mixer = FR.mixer_of(lt)
    FF = FR.ffn(spec)
    POOL = L.POOL_BYTES
    g = Gguf3(str(mdir / "model.gguf"))
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()][:2]
    p = f"blk.{l}."
    H, KVH, hd, eps, hid = G.NH, G.KVH, G.HD, G.EPS, G.HID

    # ---- the reference for tokens 0 and 1 through layer l (layers before l in fp32 too)
    if l > 0:
        _, _, x = forward(g, cfg, ids, layers=l)
    else:
        x = g.rows("token_embd.weight", ids)
    hn = rms(x, g.array(p + "attn_norm.weight"), eps)
    if mixer == FR.MIX_CONV:
        bcu = hn @ g.array(p + "shortconv.in_proj.weight").T
        B, C, u = bcu[:, :hid], bcu[:, hid:2 * hid], bcu[:, 2 * hid:]
        bx = (B * u).astype(np.float32)
        wc = g.array(p + "shortconv.conv.weight")             # [hid, 7]
        conv1 = bx[0] * wc[:, 5] + bx[1] * wc[:, 6]           # t = 1: [0 .. 0, bx0, bx1]
        y_ref = C[1] * conv1
        o = y_ref @ g.array(p + "shortconv.out_proj.weight").T
    else:
        q = rms((hn @ g.array(p + "attn_q.weight").T).reshape(2, H, hd), g.array(p + "attn_q_norm.weight"), eps)
        k = rms((hn @ g.array(p + "attn_k.weight").T).reshape(2, KVH, hd), g.array(p + "attn_k_norm.weight"), eps)
        v = (hn @ g.array(p + "attn_v.weight").T).reshape(2, KVH, hd)
        rp = cfg["rope_parameters"][cfg["layer_types"][l]]
        inv = inv_freq(float(rp["rope_theta"]), hd, float(rp.get("partial_rotary_factor", 1.0)))
        q, k = rope(q, np.arange(2), inv), rope(k, np.arange(2), inv)
        att = attention(q, k, v, np.full(H, -np.inf, np.float32), 0)
        gate = 1.0 / (1.0 + np.exp(-(hn @ g.array(p + "attn_gate.weight").T).reshape(2, H, hd).astype(np.float64)))
        att = (att * gate).astype(np.float32)
        o = att.reshape(2, H * hd) @ g.array(p + "attn_output.weight").T
        o = o[pos]
    res_ref = x[pos] + rms(o, g.array(p + "post_attention_norm.weight"), eps)
    xm_ref = rms(res_ref, g.array(p + "ffn_norm.weight"), eps)

    driver = Path(a.driver).resolve()
    mix_dir = Path(a.conv if mixer == FR.MIX_CONV else a.att).resolve()
    mlp_dir = Path(a.mlp).resolve() if a.mlp else None

    def bf(vec, n):
        o_ = np.zeros(n, bfloat16)
        o_[:len(vec)] = vec
        return o_
    pool = np.lib.format.open_memmap(out / "pool.npy", "w+", np.uint8, (POOL,))
    pool_bin = out / "pool.bin"
    consts = np.zeros(L.CD_BYTES, np.uint8)
    for nm, off in (("attn_norm", L.CD_LNW), ("post_attention_norm", L.CD_POSTOP), ("ffn_norm", L.CD_FFNN),
                    ("post_ffw_norm", L.CD_POSTFFN)):
        b = bf(g.array(p + nm + ".weight"), hid).view(np.uint8)
        consts[off:off + b.size] = b
    if mixer == FR.MIX_CONV:
        ip = q4(g, p + "shortconv.in_proj.weight")
        for dst, r0 in ((L.POOL_B, 0), (L.POOL_C, hid), (L.POOL_U, 2 * hid)):
            t = MAT(ip, hid, hid, row0=r0)
            pool[dst:dst + t.size] = t
        t = MAT(q4(g, p + "shortconv.out_proj.weight"), hid, hid)
        pool[L.POOL_OUT:L.POOL_OUT + t.size] = t
        taps = np.ascontiguousarray(wc.T.astype(np.float32)).view(np.uint8).reshape(-1)   # [7][hid] f32
        consts[L.CD_CONV:L.CD_CONV + taps.size] = taps
        state = np.zeros((G.TAPS - 1, hid), np.float32)
        state[-1] = bx[0]
        files = (("consts", consts), ("state", state), ("xres", x[pos].astype(np.float32)))
    else:
        for role, dst, rows, K in (("attn_q", L.POOL_Q, G.QW, hid), ("attn_gate", L.POOL_GATE, G.QW, hid),
                                   ("attn_k", L.POOL_K, G.KVW, hid), ("attn_v", L.POOL_V, G.KVW, hid),
                                   ("attn_output", L.POOL_O, hid, G.QW)):
            t = MAT(q4(g, p + role + ".weight"), rows, K)
            pool[dst:dst + t.size] = t
        for vec, n, off in ((g.array(p + "attn_q_norm.weight"), hd, L.CD_META),
                            (g.array(p + "attn_k_norm.weight"), hd, L.CD_META + 2 * hd)):
            b = bf(vec, n).view(np.uint8)
            consts[off:off + b.size] = b
        kv = np.zeros(2 * L.KV_ROW, np.uint8)
        kv[:L.KV_ROW] = np.concatenate([k[0].reshape(-1), v[0].reshape(-1)]).astype(bfloat16).view(np.uint8)
        ptab_b = ptab(L.MAX_CTX, G.ROT, spec.rope_theta, L.PTAB_ROW, FR.rope_table(spec, lt == "dense_local"), 0, 1.0)
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

    # ---- the MLP's slots into the pool (the same pool serves the mixer: different regions)
    ff = FF.FF
    blk = {nm: q4(g, p + nm) for nm in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")}
    dw = {}
    for s_ in range(FF.SLOTS):
        r0 = s_ * ff
        ug = UG(P.matrix(blk["ffn_gate.weight"], ff, hid, row0=r0), P.matrix(blk["ffn_up.weight"], ff, hid, row0=r0), hid)
        pool[s_ * FF.EXPERT_UG:s_ * FF.EXPERT_UG + ug.size] = ug
        down = MAT(blk["ffn_down.weight"], hid, ff, col0=s_ * ff)
        o0 = FF.POOL_DOWN + s_ * FF.UP_BYTES
        pool[o0:o0 + down.size] = down
        dw[s_] = (g.array(p + "ffn_gate.weight")[r0:r0 + ff], g.array(p + "ffn_up.weight")[r0:r0 + ff],
                  g.array(p + "ffn_down.weight")[:, s_ * ff:(s_ + 1) * ff])
    pool_file()
    img = Path(a.conv).resolve() if a.shared else mix_dir
    head = ["device", f"xclbin A {img / 'final.xclbin'}", f"kernelx ka A {mix_dir / 'insts.bin'}",
            f"buf pool {POOL} pool.bin", f"buf xres {hid * 4} xres.bin", f"buf consts {L.CD_BYTES} consts.bin"]
    if mixer == FR.MIX_CONV:
        (out / "ptab.bin").write_bytes(np.zeros(L.PTAB_BYTES, np.uint8).tobytes())
        sig = (["buf ptab %d ptab.bin" % L.PTAB_BYTES, "run ka pool xres consts state act ptab"] if a.shared
               else ["run ka pool xres consts state act"])
        drive("mix", head + [f"buf state {L.STATE_BYTES} state.bin", f"buf act {L.AD_BYTES}"] + sig + [ f"dump act act_mix.bin {L.AD_BYTES}", f"dump xres xres_mix.bin {hid * 4}",
                             f"dump state state_mix.bin {L.STATE_BYTES}"])
    else:
        drive("mix", head + [f"buf kv {L.KV_BYTES} kv.bin", f"buf act {L.AD_BYTES}",
                             f"buf ptab {L.PTAB_BYTES} ptab.bin", "run ka pool xres consts kv act ptab",
                             f"dump act act_mix.bin {L.AD_BYTES}", f"dump xres xres_mix.bin {hid * 4}", f"dump kv kv_mix.bin {2 * L.KV_ROW}"])
    actb = np.fromfile(out / "act_mix.bin", np.uint8)
    res = actb[L.AD_RES:L.AD_RES + hid * 4].view(np.float32).copy()
    xm = actb[L.AD_XM:L.AD_XM + hid * 2].view(bfloat16).astype(np.float32)
    rep = {"layer": l, "position": pos, "ids": ids, "mixer": mixer, "type": lt}
    print(f"mixer half, layer {l} ({lt}) (vs fp32 reference):", flush=True)
    if mixer == FR.MIX_CONV:
        f32 = lambda off: actb[off:off + hid * 4].view(np.float32)
        rep["b"] = stats("B", f32(L.AD_B), B[1])
        rep["c"] = stats("C", f32(L.AD_C), C[1])
        rep["u"] = stats("u", f32(L.AD_U), u[1])
        yb = actb[L.AD_Y:L.AD_Y + hid * 2].view(bfloat16).astype(np.float32)
        rep["y"] = stats("y = C conv (bf16)", yb, y_ref)
        st = np.fromfile(out / "state_mix.bin", np.float32).reshape(G.TAPS - 1, hid)
        rep["state_prev"] = stats("new state row 4 (= bx0)", st[-2], bx[0])
        rep["state_new"] = stats("new state row 5 (= bx1)", st[-1], bx[1])
        rep["state_zero"] = float(np.abs(st[:-2]).max())
    else:
        krow = np.fromfile(out / "kv_mix.bin", np.uint8)[L.KV_ROW:].view(bfloat16).astype(np.float32)
        rep["k"] = stats("new K row", krow[:G.KVW], k[1].reshape(-1))
        rep["v"] = stats("new V row", krow[G.KVW:], v[1].reshape(-1))
        og = actb[L.AD_OG:L.AD_OG + G.QW * 2].view(bfloat16).astype(np.float32)
        rep["og"] = stats("attention out * gate (og)", og, att[pos].reshape(-1))
    rep["res"] = stats("res = x + norm(o)", res, res_ref)
    rep["xm"] = stats("xm", xm, xm_ref)

    actb.tofile(out / "act_route.bin")
    if a.fused:        # the MLP ran inside the mixer's dispatch
        (out / "xres_mlp.bin").write_bytes((out / "xres_mix.bin").read_bytes())
        (out / "act_mlp.bin").write_bytes((out / "act_mix.bin").read_bytes())
    else:
      drive("mlp", ["device", f"xclbin M {mlp_dir / 'final.xclbin'}", f"kernelx km M {mlp_dir / 'insts.bin'}",
                  f"buf pool {POOL} pool.bin", f"buf xres {hid * 4} xres.bin", f"buf consts {L.CD_BYTES} consts.bin",
                  f"buf act {L.AD_BYTES} act_route.bin", "run km pool xres consts act",
                  f"dump xres xres_mlp.bin {hid * 4}", f"dump act act_mlp.bin {L.AD_BYTES}"])
    y = np.fromfile(out / "xres_mlp.bin", np.float32)
    h = np.fromfile(out / "act_mlp.bin", np.uint8)[L.AD_H:L.AD_H + FF.SLOTS * FF.H_STRIDE].view(np.float32)
    h = h.reshape(FF.SLOTS, FF.H_STRIDE // 4)[:, :ff]
    w2 = g.array(p + "post_ffw_norm.weight")

    def mlp(res_in, xm_in):
        acc = np.zeros(hid, np.float64)
        hs = []
        for s_ in range(FF.SLOTS):
            Wg, Wu, Wd = dw[s_]
            hh = silu(Wg @ xm_in) * (Wu @ xm_in)
            hs.append(hh)
            acc += Wd.astype(np.float64) @ hh.astype(np.float64)
        return res_in + rms(acc.astype(np.float32), w2, eps), hs
    y_cond, h_cond = mlp(res, xm)
    print("MLP half (vs the reference fed the NPU's res / xm):", flush=True)
    for s_ in (0, FF.SLOTS - 1):
        rep[f"h{s_}"] = stats(f"h_{s_}", h[s_], h_cond[s_])
    rep["h_all"] = stats("h (every slot)", h, np.stack(h_cond))
    rep["out_cond"] = stats("layer out", y, y_cond)
    y_ref, _ = mlp(res_ref, xm_ref)
    print("MLP half (vs the fp32 reference):", flush=True)
    rep["out"] = stats("layer out", y, y_ref)
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
