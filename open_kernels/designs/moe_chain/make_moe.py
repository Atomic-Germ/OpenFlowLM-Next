r"""MoE block of layer 0 (captured 3LiF decode step) as a host-driven chain of open
kernels, against an fp64 reference built from the CPU replica's weights.

Input: the residual after the attention block from designs/layer_chain
(ref_xres.bin, the replica's value). Chain:
  ln(post-attn norm) -> router -> for each routed expert e (reference order):
  gemv(up_e), gemv(gate_e), silu_mul, gemv(down_e), moe_axpy -> shared expert
  (gemv up/gate, silu_mul, gemv down) -> moe_fin (+ residual, sigmoid gate).
Reference uses bf16 xm and bf16 h exactly as the kernels do.

    cd tools/kernel-interp && MODEL_Q4NX=... python .../moe_chain/make_moe.py
    open-qwen-npu npu designs/moe_chain/run.cfg ; python compare_moe.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402  (PHLEGM_KERNEL_INTERP, MODEL_Q4NX, OPEN_KERNELS_CAPS)

KI = FX.kernel_interp()                    # phlegm's chain harness; in-repo successor: open_kernels/model/
sys.path.insert(0, str(KI))
os.environ["MODEL_Q4NX"] = FX.model_q4nx()
os.chdir(KI)
import decode_step as DS  # noqa: E402

POOL = FX.caps("m0d/blob_536870912_836fd8e49f35a0b6.bin")
PACK = FX.caps("m0d/000118.bo")
D = ".."                                   # designs/, relative to this cfg
OUT = "."
S = 163_840
CH = 5120
NE = 8


def silu(x):
    return x / (1 + np.exp(-x))


def bf(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float64)


def wr(name, arr):
    (HERE / name).write_bytes(np.ascontiguousarray(arr).tobytes())
    return np.ascontiguousarray(arr).nbytes


def main() -> int:
    m = DS.m
    x_res = np.fromfile(HERE.parent / "layer_chain" / "ref_xres.bin", np.float32).astype(np.float64)
    postln = m.bf16("model.layer.0.post_attention_layernorm.weight")
    xm = bf(DS.F.rms(x_res) * postln)                       # bf16, as the ln kernel emits
    router = m.bf16("model.layer.0.moe_router.weight").astype(np.float64)
    lg = xm @ router
    p = np.exp(lg - lg.max()); p /= p.sum()
    top = np.argsort(-p, kind="stable")[:NE]
    w8 = p[top] / p[top].sum()
    acc = np.zeros(2048)
    for e, ww in zip(top, w8):
        gt, up, Dn = DS.F.expert_weights(0, int(e))
        h = bf(silu(gt.astype(np.float64) @ xm) * (up.astype(np.float64) @ xm))
        acc += ww * (Dn.astype(np.float64) @ h)
    Wsg, Wsu, Ds = DS.F.shared_weights(0)
    hs = bf(silu(Wsg.astype(np.float64) @ xm) * (Wsu.astype(np.float64) @ xm))
    sh = Ds.astype(np.float64) @ hs
    sgw = m.bf16("model.layer.0.shared_expert_gate.weight").astype(np.float64)
    sg = 1 / (1 + np.exp(-(xm @ sgw)))
    out_ref = x_res + acc + sg * sh
    # also the replica's own moe_decode (fp32 xm/h) for scale
    out_rep = DS.moe_decode(0, x_res.copy())

    pool = np.memmap(POOL, np.uint8, "r")
    pack = np.fromfile(PACK, np.uint8)
    wr("xres.bin", x_res.astype(np.float32))
    wr("zero.bin", np.zeros(2048, np.float32))
    wr("postln.bin", pack[4096:8192])
    wr("router_w.bin", pack[12288:12288 + 1048576])
    wr("sgw.bin", pack[8192:12288])
    for k, e in enumerate(top):
        e = int(e)
        up = b"".join(bytes(pool[(8 * e + 2 * s) * S:(8 * e + 2 * s + 1) * S]) for s in range(4))
        gt = b"".join(bytes(pool[(8 * e + 2 * s + 1) * S:(8 * e + 2 * s + 2) * S]) for s in range(4))
        (HERE / f"w_up{k}.bin").write_bytes(up)
        (HERE / f"w_gate{k}.bin").write_bytes(gt)
        wr(f"w_down{k}.bin", np.asarray(pool[335_544_320 + e * 655_360:335_544_320 + (e + 1) * 655_360]))
        eb = np.zeros(1024, np.int32); eb[0] = k
        wr(f"e{k}.bin", eb)
    wr("w_su.bin", np.asarray(pool[503_316_480:503_316_480 + 655_360]))
    wr("w_sg.bin", np.asarray(pool[503_971_840:503_971_840 + 655_360]))
    wr("w_sd.bin", np.asarray(pool[504_627_200:504_627_200 + 655_360]))
    ref = np.zeros(1024, np.float32); ref[:256] = p; ref[256:264] = top.astype(np.int32).view(np.float32); ref[264:272] = w8
    wr("ref_rout.bin", ref)
    wr("ref_out.bin", out_ref.astype(np.float32))
    wr("ref_out_replica.bin", out_rep.astype(np.float32))
    wr("ref_xm.bin", xm.astype(np.float32))

    cfg = [
        "device",
        f"xclbin L {D}/ln/build/final.xclbin", f"kernelx ln L {D}/ln/build/insts.bin",
        f"xclbin R {D}/router/build/final.xclbin", f"kernelx rt R {D}/router/build/insts.bin",
        f"xclbin U {D}/gemv_q4/build_exp_up/final.xclbin", f"kernelx gexp U {D}/gemv_q4/build_exp_up/insts.bin",
        f"xclbin W {D}/gemv_q4/build_exp_down/final.xclbin", f"kernelx gdown W {D}/gemv_q4/build_exp_down/insts.bin",
        f"xclbin M {D}/silu_mul/build/final.xclbin", f"kernelx sm M {D}/silu_mul/build/insts.bin",
        f"xclbin S {D}/gemv_q4/build_share_up/final.xclbin", f"kernelx gsu S {D}/gemv_q4/build_share_up/insts.bin",
        f"xclbin T {D}/gemv_q4/build_share_down/final.xclbin", f"kernelx gsd T {D}/gemv_q4/build_share_down/insts.bin",
        f"xclbin A {D}/moe_combine/build_axpy/final.xclbin", f"kernelx ax A {D}/moe_combine/build_axpy/insts.bin",
        f"xclbin F {D}/moe_combine/build_fin/final.xclbin", f"kernelx fin F {D}/moe_combine/build_fin/insts.bin",
        f"buf xres 8192 {OUT}/xres.bin", f"buf zero 8192 {OUT}/zero.bin", f"buf postln 4096 {OUT}/postln.bin",
        "buf xres1 8192", "buf xm 4096",
        f"buf rw 1048576 {OUT}/router_w.bin", "buf rout 4096",
        f"buf sgw 4096 {OUT}/sgw.bin",
    ]
    for k in range(NE):
        cfg += [f"buf wu{k} 655360 {OUT}/w_up{k}.bin", f"buf wg{k} 655360 {OUT}/w_gate{k}.bin",
                f"buf wd{k} 655360 {OUT}/w_down{k}.bin", f"buf e{k} 4096 {OUT}/e{k}.bin",
                f"buf u{k} 2048", f"buf g{k} 2048", f"buf h{k} 1024", f"buf y{k} 8192"]
    cfg += [f"buf wsu 655360 {OUT}/w_su.bin", f"buf wsg 655360 {OUT}/w_sg.bin", f"buf wsd 655360 {OUT}/w_sd.bin",
            "buf su 2048", "buf sgv 2048", "buf hs 1024", "buf shared 8192",
            "buf accA 8192", "buf accB 8192", "buf out 8192",
            "run ln xres zero postln xres1 xm",
            "run rt xm rw rout"]
    for k in range(NE):
        src, dst = ("accA", "accB") if k % 2 == 0 else ("accB", "accA")
        cfg += [f"run gexp wu{k} xm u{k}", f"run gexp wg{k} xm g{k}", f"run sm g{k} u{k} h{k}",
                f"run gdown wd{k} h{k} y{k}", f"run ax rout y{k} {src} e{k} {dst}"]
    last = "accB" if (NE - 1) % 2 == 0 else "accA"
    cfg += ["run gsu wsu xm su", "run gsu wsg xm sgv", "run sm sgv su hs", "run gsd wsd hs shared",
            f"run fin {last} xres1 shared xm sgw out",
            f"dump rout {OUT}/y_rout.bin 4096", f"dump out {OUT}/y_out.bin 8192",
            f"dump xm {OUT}/y_xm.bin 4096", f"dump y0 {OUT}/y_y0.bin 8192", ""]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"top8={top.tolist()} w8={np.round(w8, 4).tolist()}")
    print(f"out_ref[:4]={out_ref[:4]} replica(fp32 xm/h)[:4]={out_rep[:4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
