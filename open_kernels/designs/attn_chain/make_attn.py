r"""Full-attention layer (layer 2 of the captured 3LiF decode step, position 11)
as a host-driven chain of open kernels:
  ln -> gemv(q) gemv(gate) gemv(k) gemv(v) -> attn -> gemv(o) -> ln(+residual, post-attn norm)
Input residual = the replica's state after layers 0 and 1 (linear+MoE twice);
KV cache = the captured C:/caps/m0c/000902.bo (11 prefill rows).
Reference: fp64 mirror of decode_step.py attn_decode with the kernels' bf16
roundings (xn, k'/v' cache rows, og). Run from tools/kernel-interp (model).

    PHLEGM_KERNEL_INTERP=<phlegm>/tools/kernel-interp MODEL_Q4NX=<model_3LiF.q4nx> \
    OPEN_KERNELS_CAPS=<captures> python make_attn.py
    run_kernel run.cfg ; python compare_attn.py

This is phlegm's step-by-step chain harness (it imports phlegm's kernel-interp);
the in-repo successor is open_kernels/model/. Paths in run.cfg are relative to
this directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402

KI = FX.kernel_interp()
sys.path.insert(0, str(KI))
os.environ["MODEL_Q4NX"] = FX.model_q4nx()
os.chdir(KI)
import decode_step as DS  # noqa: E402
from q4nx import bf16_to_f32  # noqa: E402

CAP = FX.caps("m0c/000902.bo").parent
POOL2 = FX.caps("m0d/000123.bo")
PACK2 = FX.caps("m0d/000124.bo")
SIDE2 = FX.caps("m0d/000125.bo")
D = ".."                                   # designs/, relative to this cfg
OUT = "."
TOK, POS, LAYER = 248068, 11, 2
NH, KVH, HD = 16, 2, 256


def bf(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float64)


def wr(name, arr):
    a = np.ascontiguousarray(arr)
    (HERE / name).write_bytes(a.tobytes())
    return a.nbytes


def rope1(t_, p):
    h = 32
    freqs = (1e7) ** (-np.arange(h) / h)
    ang = p * freqs
    C, Sn = np.cos(ang), np.sin(ang)
    y = t_.copy()
    x1, x2 = t_[..., :h], t_[..., h:64]
    y[..., :h] = x1 * C - x2 * Sn
    y[..., h:64] = x2 * C + x1 * Sn
    return y, C, Sn


def main() -> int:
    m = DS.m
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    x = bf16_to_f32(np.frombuffer(m.mm[base + TOK * 4096: base + (TOK + 1) * 4096], dtype=np.uint16)).astype(np.float64)
    cs0, S0 = DS.load_linear_state(str(CAP / "000898.bo"))
    cs1, S1 = DS.load_linear_state(str(CAP / "000900.bo"))
    kvb = bf16_to_f32(np.fromfile(CAP / "000902.bo", dtype=np.uint16))
    kcache = kvb[:POS * 512].reshape(POS, KVH, HD).astype(np.float64)
    vcache = kvb[536576:536576 + POS * 512].reshape(POS, KVH, HD).astype(np.float64)
    x, _, _ = DS.linear_decode(0, x, cs0, S0)
    x = DS.moe_decode(0, x)
    x, _, _ = DS.linear_decode(1, x, cs1, S1)
    x_res = DS.moe_decode(1, x)                      # attention-layer input

    # ---- replica (fp32 activations) for scale
    x_rep, _, _ = DS.attn_decode(LAYER, x_res.copy(), kcache, vcache, POS)

    # ---- bf16-faithful reference
    ln_w = m.bf16(f"model.layer.{LAYER}.input_layernorm.weight")
    xn = bf(DS.F.rms(x_res) * ln_w)
    Wqg = DS.F.dequant_std(f"model.layer.{LAYER}.self_attn.q_proj.weight", 8192, 2048).astype(np.float64)
    Wq, Wg = Wqg[:4096], Wqg[4096:]
    Wk = DS.F.dequant_std(f"model.layer.{LAYER}.self_attn.k_proj.weight", 512, 2048).astype(np.float64)
    Wv = DS.F.dequant_std(f"model.layer.{LAYER}.self_attn.v_proj.weight", 512, 2048).astype(np.float64)
    Wo = DS.F.dequant_std(f"model.layer.{LAYER}.self_attn.o_proj.weight", 2048, 4096).astype(np.float64)
    qn = m.bf16(f"model.layer.{LAYER}.self_attn.q_norm.weight").astype(np.float64)
    kn = m.bf16(f"model.layer.{LAYER}.self_attn.k_norm.weight").astype(np.float64)
    q = DS.F.rms((xn @ Wq.T).reshape(NH, HD)) * qn
    g = (xn @ Wg.T).reshape(NH, HD)
    k = DS.F.rms((xn @ Wk.T).reshape(KVH, HD)) * kn
    v = (xn @ Wv.T).reshape(KVH, HD)
    q, C, Sn = rope1(q, POS)
    k, _, _ = rope1(k, POS)
    k_new, v_new = bf(k), bf(v)                     # cache rows (bf16)
    K = np.concatenate([kcache, k_new[None]], 0)
    V = np.concatenate([vcache, v_new[None]], 0)
    o = np.zeros((NH, HD))
    for h in range(NH):
        s = (K[:, h // 8] @ q[h]) / 16.0
        a = np.exp(s - s.max()); a /= a.sum()
        o[h] = a @ V[:, h // 8]
    og = bf((o * (1 / (1 + np.exp(-g)))).reshape(4096))
    x_out = x_res + og @ Wo.T
    postln = m.bf16(f"model.layer.{LAYER}.post_attention_layernorm.weight")
    xm = DS.F.rms(x_out) * postln

    # ---- buffers
    pool = np.memmap(POOL2, np.uint8, "r")
    pack = np.fromfile(PACK2, np.uint8)
    side = np.fromfile(SIDE2, np.uint8)
    wr("xres.bin", x_res.astype(np.float32)); wr("zero.bin", np.zeros(2048, np.float32))
    wr("lnw.bin", pack[0:4096]); wr("postln.bin", pack[4096:8192])
    wr("w_q.bin", np.asarray(pool[505_282_560:505_282_560 + 5_242_880]))
    wr("w_k.bin", np.asarray(pool[510_525_440:510_525_440 + 655_360]))
    wr("w_v.bin", np.asarray(pool[511_180_800:511_180_800 + 655_360]))
    wr("w_gate.bin", np.asarray(pool[511_836_160:511_836_160 + 5_242_880]))
    wr("w_o.bin", np.asarray(pool[517_079_040:517_079_040 + 5_242_880]))
    meta = np.zeros(2048, np.uint8)                 # attn.h's two elements: [qn | kn], the position record
    meta[0:512] = side[128:640]                     # q_norm (effective) bf16[256]
    meta[512:1024] = side[640:1152]                 # k_norm
    meta[1024:1032] = np.array([POS, POS], np.int32).view(np.uint8)   # pos, nf (= the static row fills)
    meta[1536:1664] = C.astype(np.float32).view(np.uint8)
    meta[1664:1792] = Sn.astype(np.float32).view(np.uint8)
    wr("meta.bin", meta)
    wr("kv.bin", np.fromfile(CAP / "000902.bo", np.uint8))
    wr("ref_knew.bin", k_new.astype(np.float32)); wr("ref_vnew.bin", v_new.astype(np.float32))
    wr("ref_og.bin", og.astype(np.float32)); wr("ref_xres.bin", x_out.astype(np.float32))
    wr("ref_xm.bin", xm.astype(np.float32)); wr("ref_xres_replica.bin", x_rep.astype(np.float32))

    cfg = [
        "device",
        f"xclbin L {D}/ln/build/final.xclbin", f"kernelx ln L {D}/ln/build/insts.bin",
        f"xclbin Z {D}/gemv_q4/build_z/final.xclbin", f"kernelx g4k Z {D}/gemv_q4/build_z/insts.bin",
        f"xclbin S {D}/gemv_q4/build_share_up/final.xclbin", f"kernelx g512 S {D}/gemv_q4/build_share_up/insts.bin",
        f"xclbin A {D}/attn/build/final.xclbin", f"kernelx at A {D}/attn/build/insts.bin",
        f"xclbin O {D}/gemv_q4/build_out/final.xclbin", f"kernelx go O {D}/gemv_q4/build_out/insts.bin",
        f"xclbin H {D}/gemv_q4/build_z_hi/final.xclbin", f"kernelx g4kh H {D}/gemv_q4/build_z_hi/insts.bin",
        f"xclbin I {D}/gemv_q4/build_512_hi/final.xclbin", f"kernelx g512h I {D}/gemv_q4/build_512_hi/insts.bin",
        f"buf xres 8192 {OUT}/xres.bin", f"buf zero 8192 {OUT}/zero.bin",
        f"buf lnw 4096 {OUT}/lnw.bin", f"buf postln 4096 {OUT}/postln.bin",
        "buf xres1 8192", "buf xn 4096",
        f"buf wq 5242880 {OUT}/w_q.bin", f"buf wg 5242880 {OUT}/w_gate.bin",
        f"buf wk 655360 {OUT}/w_k.bin", f"buf wv 655360 {OUT}/w_v.bin", f"buf wo 5242880 {OUT}/w_o.bin",
        "buf qg 32768", "buf kvn 4096",
        f"buf meta 2048 {OUT}/meta.bin", f"buf kv 3145728 {OUT}/kv.bin",
        "buf kvnew 2048", "buf og 8192", "buf out 8192", "buf xres2 8192", "buf xm 4096",
        "run ln xres zero lnw xres1 xn",
        "run g4k wq xn qg", "run g4kh wg xn qg", "run g512 wk xn kvn", "run g512h wv xn kvn",
        "run at meta qg kvn kv kvnew og",
        "run go wo og out",
        "run ln xres1 out postln xres2 xm",
        f"dump kvnew {OUT}/y_kvnew.bin 2048", f"dump og {OUT}/y_og.bin 8192",
        f"dump xres2 {OUT}/y_xres.bin 8192", f"dump xm {OUT}/y_xm.bin 4096", "",
    ]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"x_out[:4]={x_out[:4]} replica[:4]={x_rep[:4]} og absmax={np.abs(og).max():.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
