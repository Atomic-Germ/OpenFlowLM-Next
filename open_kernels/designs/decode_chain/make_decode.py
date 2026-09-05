r"""The whole decode step of the captured 3LiF model (token 248068 at position 11)
as ONE driver config of open kernels: L0 linear+MoE, L1 linear+MoE, L2 full
attention+MoE, final norm, lm_head. Logits are compared with FLM's own captured
logits (C:/caps/m0c/000905.bo) and with the CPU replica (decode_step.py).

Routing: each MoE block's 8 experts are sliced by the host from the routing the
mirrored (bf16-faithful) math predicts; if a previous NPU run left y_rout{l}.bin
with a different selection, that selection is used instead (and reported).

    cd tools/kernel-interp && python .../decode_chain/make_decode.py
    open-qwen-npu npu designs/decode_chain/run.cfg ; python compare_decode.py
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
from q4nx import bf16_to_f32  # noqa: E402

CAPC = FX.caps("m0c/000905.bo").parent
CAPD = FX.caps("m0d/000118.bo").parent
POOLF = {0: CAPD / "blob_536870912_836fd8e49f35a0b6.bin", 1: CAPD / "000120.bo", 2: CAPD / "000123.bo"}
PACKF = {0: CAPD / "000118.bo", 1: CAPD / "000121.bo", 2: CAPD / "000124.bo"}
SIDEF = {0: CAPD / "000119.bo", 1: CAPD / "000122.bo", 2: CAPD / "000125.bo"}
STATEF = {0: CAPC / "000898.bo", 1: CAPC / "000900.bo"}
D = ".."                                   # designs/, relative to this cfg
OUT = "."
TOK, POS = 248068, 11
S = 163_840
NE = 8


def bf(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float64)


def wr(name, arr):
    a = np.ascontiguousarray(arr)
    (HERE / name).write_bytes(a.tobytes())


def silu(x):
    return x / (1 + np.exp(-x))


def routing(m, l, x_res):
    """Mirrored routing (bf16 xm) for layer l given the residual after attention."""
    postln = m.bf16(f"model.layer.{l}.post_attention_layernorm.weight")
    xm = bf(DS.F.rms(x_res) * postln)
    lg = xm @ m.bf16(f"model.layer.{l}.moe_router.weight").astype(np.float64)
    p = np.exp(lg - lg.max()); p /= p.sum()
    top = np.argsort(-p, kind="stable")[:NE]
    prev = HERE / f"y_rout{l}.bin"
    if prev.is_file():
        got = np.fromfile(prev, np.float32)[256:264].view(np.int32)
        if got.tolist() != top.tolist():
            print(f"layer {l}: NPU routing {got.tolist()} != predicted {top.tolist()}; using the NPU's")
            top = got.astype(np.int64)
    return top


def main() -> int:
    m = DS.m
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    x = bf16_to_f32(np.frombuffer(m.mm[base + TOK * 4096: base + (TOK + 1) * 4096], dtype=np.uint16)).astype(np.float64)
    cs0, S0 = DS.load_linear_state(str(STATEF[0]))
    cs1, S1 = DS.load_linear_state(str(STATEF[1]))
    kvb = bf16_to_f32(np.fromfile(CAPC / "000902.bo", dtype=np.uint16))
    kcache = kvb[:POS * 512].reshape(POS, 2, 256).astype(np.float64)
    vcache = kvb[536576:536576 + POS * 512].reshape(POS, 2, 256).astype(np.float64)

    # ---- replica chain (fp32 activations) + mirrored routing per layer
    x0, _, _ = DS.linear_decode(0, x.copy(), cs0.copy(), S0.copy()); top = {0: routing(m, 0, x0)}
    r0 = DS.moe_decode(0, x0)
    x1, _, _ = DS.linear_decode(1, r0.copy(), cs1.copy(), S1.copy()); top[1] = routing(m, 1, x1)
    r1 = DS.moe_decode(1, x1)
    x2, _, _ = DS.attn_decode(2, r1.copy(), kcache, vcache, POS); top[2] = routing(m, 2, x2)
    r2 = DS.moe_decode(2, x2)
    normw = m.bf16("model.norm.weight")
    hn = (DS.F.rms(r2) * normw).astype(np.float32)
    logits_rep = DS.lm_head_odd(hn)
    wr("ref_logits_replica.bin", logits_rep.astype(np.float32))
    for l, r in ((0, r0), (1, r1), (2, r2)):
        wr(f"ref_res{l}.bin", r.astype(np.float32))

    # ---- buffers
    wr("xres0.bin", x.astype(np.float32))
    wr("zero.bin", np.zeros(2048, np.float32))
    wr("normw.bin", normw.astype(np.float32).astype(bfloat16))
    for k in range(NE):
        eb = np.zeros(1024, np.int32); eb[0] = k
        wr(f"e{k}.bin", eb)
    pools = {l: np.memmap(POOLF[l], np.uint8, "r") for l in range(3)}
    packs = {l: np.fromfile(PACKF[l], np.uint8) for l in range(3)}
    sides = {l: np.fromfile(SIDEF[l], np.uint8) for l in range(3)}
    for l in range(3):
        pk, sd, pl = packs[l], sides[l], pools[l]
        wr(f"lnw{l}.bin", pk[0:4096]); wr(f"postln{l}.bin", pk[4096:8192])
        wr(f"sgw{l}.bin", pk[8192:12288]); wr(f"rw{l}.bin", pk[12288:12288 + 1048576])
        for k, e in enumerate(top[l]):
            e = int(e)
            (HERE / f"wu{l}{k}.bin").write_bytes(b"".join(bytes(pl[(8 * e + 2 * s) * S:(8 * e + 2 * s + 1) * S]) for s in range(4)))
            (HERE / f"wg{l}{k}.bin").write_bytes(b"".join(bytes(pl[(8 * e + 2 * s + 1) * S:(8 * e + 2 * s + 2) * S]) for s in range(4)))
            wr(f"wd{l}{k}.bin", np.asarray(pl[335_544_320 + e * 655_360:335_544_320 + (e + 1) * 655_360]))
        wr(f"wsu{l}.bin", np.asarray(pl[503_316_480:503_316_480 + 655_360]))
        wr(f"wsg{l}.bin", np.asarray(pl[503_971_840:503_971_840 + 655_360]))
        wr(f"wsd{l}.bin", np.asarray(pl[504_627_200:504_627_200 + 655_360]))
        if l < 2:
            wr(f"wqkv{l}.bin", np.asarray(pl[505_282_560:505_282_560 + 10_485_760]))
            wr(f"wz{l}.bin", np.asarray(pl[515_768_320:515_768_320 + 5_242_880]))
            wr(f"wout{l}.bin", sd[328_192:328_192 + 10_485_760])
            st = np.fromfile(STATEF[l], np.uint8)
            wr(f"state{l}.bin", st[:49152]); wr(f"sin{l}.bin", st[49152:49152 + 2097152])
            nwp = np.zeros(2048, bfloat16); nwp[:128] = sd[65536:65536 + 256].view(bfloat16)
            wr(f"nw{l}.bin", nwp)
            side = np.zeros(335872, np.uint8)
            side[4096:4096 + 131072] = sd[66048:66048 + 131072]
            side[135168:135168 + 131072] = sd[197120:197120 + 131072]
            small = np.zeros(1024, np.float32)
            small[:32] = sd[65792:65792 + 128].view(np.float32); small[32:64] = sd[65920:65920 + 128].view(np.float32)
            side[266240:266240 + 4096] = small.view(np.uint8)
            convw = sd[0:65536].view(bfloat16).reshape(4, 8192)
            side[270336:270336 + 65536] = convw.reshape(4, 8, 1024).transpose(1, 0, 2).reshape(-1).view(np.uint8)
            wr(f"side{l}.bin", side)
        else:
            wr("wq.bin", np.asarray(pl[505_282_560:505_282_560 + 5_242_880]))
            wr("wk.bin", np.asarray(pl[510_525_440:510_525_440 + 655_360]))
            wr("wv.bin", np.asarray(pl[511_180_800:511_180_800 + 655_360]))
            wr("wgate.bin", np.asarray(pl[511_836_160:511_836_160 + 5_242_880]))
            wr("wo.bin", np.asarray(pl[517_079_040:517_079_040 + 5_242_880]))
            freqs = (1e7) ** (-np.arange(32) / 32)
            meta = np.zeros(2048, np.uint8)
            meta[0:4] = np.array([POS], np.int32).view(np.uint8)
            meta[512:1024] = sd[128:640]; meta[1024:1536] = sd[640:1152]
            meta[1536:1664] = np.cos(POS * freqs).astype(np.float32).view(np.uint8)
            meta[1664:1792] = np.sin(POS * freqs).astype(np.float32).view(np.uint8)
            wr("meta.bin", meta)
            wr("kv.bin", np.fromfile(CAPC / "000902.bo", np.uint8))

    # ---- config
    X = [("L", "ln", "ln/build"), ("Q", "gqkv", "gemv_q4/build_qkv"), ("Z", "gz", "gemv_q4/build_z"),
         ("G", "glue", "dn_glue/build"), ("N", "dn", "deltanet/build"), ("P", "post", "dn_post/build"),
         ("O", "gout", "gemv_q4/build_out"), ("R", "rt", "router/build"), ("U", "gexp", "gemv_q4/build_exp_up"),
         ("W", "gdown", "gemv_q4/build_exp_down"), ("M", "sm", "silu_mul/build"), ("S", "gsu", "gemv_q4/build_share_up"),
         ("T", "gsd", "gemv_q4/build_share_down"), ("A", "ax", "moe_combine/build_axpy"), ("F", "fin", "moe_combine/build_fin"),
         ("H", "g4kh", "gemv_q4/build_z_hi"), ("I", "g512h", "gemv_q4/build_512_hi"), ("X", "at", "attn/build"),
         ("K", "lm", "lm_head_q8/build_full")]
    cfg = ["device"]
    for tag, kn, path in X:
        cfg += [f"xclbin {tag} {D}/{path}/final.xclbin", f"kernelx {kn} {tag} {D}/{path}/insts.bin"]
    cfg += [f"buf xres0 8192 {OUT}/xres0.bin", f"buf zero 8192 {OUT}/zero.bin", f"buf normw 4096 {OUT}/normw.bin",
            "buf accA 8192", "buf accB 8192", f"buf lmpool 542113792 {FX.caps_cfg('m0d/000127.bo')}",
            "buf xresf 8192", "buf hn 4096", "buf logits 993280"]
    cfg += [f"buf e{k} 4096 {OUT}/e{k}.bin" for k in range(NE)]
    runs = []
    for l in range(3):
        cfg += [f"buf lnw{l} 4096 {OUT}/lnw{l}.bin", f"buf postln{l} 4096 {OUT}/postln{l}.bin",
                f"buf sgw{l} 4096 {OUT}/sgw{l}.bin", f"buf rw{l} 1048576 {OUT}/rw{l}.bin",
                f"buf xa{l} 8192", f"buf xn{l} 4096", f"buf out{l} 8192", f"buf xb{l} 8192", f"buf xm{l} 4096",
                f"buf rout{l} 4096", f"buf xc{l} 8192",
                f"buf wsu{l} 655360 {OUT}/wsu{l}.bin", f"buf wsg{l} 655360 {OUT}/wsg{l}.bin", f"buf wsd{l} 655360 {OUT}/wsd{l}.bin",
                f"buf su{l} 2048", f"buf sgv{l} 2048", f"buf hs{l} 1024", f"buf sh{l} 8192"]
        for k in range(NE):
            cfg += [f"buf wu{l}{k} 655360 {OUT}/wu{l}{k}.bin", f"buf wg{l}{k} 655360 {OUT}/wg{l}{k}.bin",
                    f"buf wd{l}{k} 655360 {OUT}/wd{l}{k}.bin", f"buf u{l}{k} 2048", f"buf g{l}{k} 2048",
                    f"buf h{l}{k} 1024", f"buf y{l}{k} 8192"]
        xin = "xres0" if l == 0 else f"xc{l - 1}"
        if l < 2:
            cfg += [f"buf wqkv{l} 10485760 {OUT}/wqkv{l}.bin", f"buf wz{l} 5242880 {OUT}/wz{l}.bin",
                    f"buf wout{l} 10485760 {OUT}/wout{l}.bin", f"buf side{l} 335872 {OUT}/side{l}.bin",
                    f"buf state{l} 49152 {OUT}/state{l}.bin", f"buf nstate{l} 49152", f"buf vec{l} 65536",
                    f"buf sin{l} 2097152 {OUT}/sin{l}.bin", f"buf sout{l} 2097152", f"buf o{l} 16384",
                    f"buf nw{l} 4096 {OUT}/nw{l}.bin", f"buf qkv{l} 32768", f"buf z{l} 16384", f"buf og{l} 8192"]
            runs += [f"run ln {xin} zero lnw{l} xa{l} xn{l}",
                     f"run gqkv wqkv{l} xn{l} qkv{l}", f"run gz wz{l} xn{l} z{l}",
                     f"dump xn{l} {OUT}/y_xn{l}.bin 4096", f"load side{l} {OUT}/y_xn{l}.bin",
                     f"run glue side{l} qkv{l} state{l} nstate{l} vec{l}",
                     f"run dn sin{l} vec{l} sout{l} o{l}",
                     f"run post o{l} z{l} nw{l} og{l}",
                     f"run gout wout{l} og{l} out{l}",
                     f"run ln xa{l} out{l} postln{l} xb{l} xm{l}"]
        else:
            cfg += [f"buf wq 5242880 {OUT}/wq.bin", f"buf wgate 5242880 {OUT}/wgate.bin", f"buf wk 655360 {OUT}/wk.bin",
                    f"buf wv 655360 {OUT}/wv.bin", f"buf wo 5242880 {OUT}/wo.bin", "buf qg 32768", "buf kvn 4096",
                    f"buf meta 2048 {OUT}/meta.bin", f"buf kv 3145728 {OUT}/kv.bin", "buf kvnew 2048", "buf og2 8192"]
            runs += [f"run ln {xin} zero lnw{l} xa{l} xn{l}",
                     f"run gqkv_dummy" if False else f"run gz wq xn{l} qg", f"run g4kh wgate xn{l} qg",
                     f"run gsu wk xn{l} kvn", f"run g512h wv xn{l} kvn",
                     "run at meta qg kvn kv kvnew og2",
                     "run gout wo og2 out2",
                     f"run ln xa{l} out2 postln{l} xb{l} xm{l}"]
        # MoE block
        runs += [f"run rt xm{l} rw{l} rout{l}"]
        for k in range(NE):
            src, dst = ("accA", "accB") if k % 2 == 0 else ("accB", "accA")
            runs += [f"run gexp wu{l}{k} xm{l} u{l}{k}", f"run gexp wg{l}{k} xm{l} g{l}{k}", f"run sm g{l}{k} u{l}{k} h{l}{k}",
                     f"run gdown wd{l}{k} h{l}{k} y{l}{k}", f"run ax rout{l} y{l}{k} {src} e{k} {dst}"]
        last = "accB" if (NE - 1) % 2 == 0 else "accA"
        runs += [f"run gsu wsu{l} xm{l} su{l}", f"run gsu wsg{l} xm{l} sgv{l}", f"run sm sgv{l} su{l} hs{l}",
                 f"run gsd wsd{l} hs{l} sh{l}", f"run fin {last} xb{l} sh{l} xm{l} sgw{l} xc{l}",
                 f"dump rout{l} {OUT}/y_rout{l}.bin 4096", f"dump xc{l} {OUT}/y_res{l}.bin 8192"]
    runs += ["run ln xc2 zero normw xresf hn", "run lm lmpool hn logits", f"dump logits {OUT}/y_logits.bin 993280", ""]
    (HERE / "run.cfg").write_text("\n".join(cfg + runs), newline="\n")
    cap = np.fromfile(CAPC / "000905.bo", np.float32)[:124160]
    print(f"routing: L0 {top[0].tolist()} L1 {top[1].tolist()} L2 {top[2].tolist()}")
    print(f"replica vs FLM capture: corr {np.corrcoef(logits_rep, cap)[0, 1]:.5f}; "
          f"argmax vocab replica {2 * int(logits_rep.argmax()) + 1} capture {2 * int(cap.argmax()) + 1}")
    print(f"{len([r for r in runs if r.startswith('run ')])} runs, {len(X)} xclbin contexts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
