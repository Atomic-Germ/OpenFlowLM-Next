r"""Full linear-attention layer (decode, layer 0 of the captured 3LiF decode block)
as a host-driven chain of the open kernels, with the fp64 CPU replica
(tools/kernel-interp/decode_step.py linear_decode) as the reference.

Inputs: embed(token 248068) as the residual, states from C:/caps/m0c/000898.bo,
weights from the captured L0 pool/pack/side (C:/caps/m0d/000117..119 = blob).
Chain:  ln -> gemv(qkv), gemv(z) -> glue -> dn_step -> post -> gemv(out) -> ln(+residual, post-attn norm)

    MODEL_Q4NX=/mnt/c/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model_3LiF.q4nx python make_chain.py
    open-qwen-npu npu designs/layer_chain/run.cfg ; python compare_chain.py
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
# full_forward.py loads prompt_token_ids.npy from the cwd at import; it is the
# captured prompt (boundary_manifest.json). Create it if missing and run from there.
_ids = KI / "prompt_token_ids.npy"
if not _ids.is_file():
    import json
    _man = json.load(open(FX.caps("pf_t11_full/boundary_manifest.json")))
    np.save(_ids, np.array(_man["prefill_token_ids"], dtype=np.int64))
os.chdir(KI)
import decode_step as DS  # noqa: E402  (loads the model)
from q4nx import bf16_to_f32  # noqa: E402

POOL = FX.caps("m0d/blob_536870912_836fd8e49f35a0b6.bin")
PACK = FX.caps("m0d/000118.bo")
SIDE = FX.caps("m0d/000119.bo")
STATE = FX.caps("m0c/000898.bo")
TOK = 248068
D = ".."                                   # designs/, relative to this cfg
OUT = "."


def wr(name: str, arr: np.ndarray) -> int:
    (HERE / name).write_bytes(arr.tobytes())
    return arr.nbytes


def main() -> int:
    m = DS.m
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    x = bf16_to_f32(np.frombuffer(m.mm[base + TOK * 4096: base + (TOK + 1) * 4096], dtype=np.uint16)).astype(np.float64)
    cs0, S0 = DS.load_linear_state(str(STATE))

    # ---- reference
    x_ref, cs_ref, S_ref = DS.linear_decode(0, x.copy(), cs0.copy(), S0.copy())
    postln = m.bf16("model.layer.0.post_attention_layernorm.weight")
    xm_ref = DS.F.rms(x_ref) * postln
    ln_w = m.bf16("model.layer.0.input_layernorm.weight")
    xn_ref = DS.F.rms(x) * ln_w

    # ---- inputs
    pool = np.memmap(POOL, np.uint8, "r")
    side_raw = np.fromfile(SIDE, np.uint8)
    pack = np.fromfile(PACK, np.uint8)
    st = np.fromfile(STATE, np.uint8)
    sizes = {}
    sizes["xres"] = wr("x_res.bin", x.astype(np.float32))
    sizes["zero"] = wr("zero.bin", np.zeros(2048, np.float32))
    sizes["lnw"] = wr("lnw.bin", pack[0:4096])
    sizes["postln"] = wr("postln.bin", pack[4096:8192])
    sizes["wqkv"] = wr("w_qkv.bin", np.asarray(pool[505_282_560:505_282_560 + 10_485_760]))
    sizes["wz"] = wr("w_z.bin", np.asarray(pool[515_768_320:515_768_320 + 5_242_880]))
    sizes["wout"] = wr("w_out.bin", side_raw[328_192:328_192 + 10_485_760])
    sizes["state"] = wr("state.bin", st[:49152])
    sizes["sin"] = wr("s_in.bin", st[49152:49152 + 32 * 128 * 128 * 4])
    nwp = np.zeros(2048, bfloat16)
    nwp[:128] = side_raw[65536:65536 + 256].view(bfloat16)
    sizes["nw"] = wr("nw.bin", nwp)
    # glue side blob (dn_glue layout, xn slot filled at runtime by `load`)
    Wa = side_raw[66048:66048 + 131072]
    Wb = side_raw[197120:197120 + 131072]
    A = side_raw[65792:65792 + 128].view(np.float32)
    dtb = side_raw[65920:65920 + 128].view(np.float32)
    convw = side_raw[0:65536].view(bfloat16).reshape(4, 8192)
    side = np.zeros(335872, np.uint8)
    side[4096:4096 + 131072] = Wa
    side[135168:135168 + 131072] = Wb
    small = np.zeros(1024, np.float32)
    small[:32] = A
    small[32:64] = dtb
    side[266240:266240 + 4096] = small.view(np.uint8)
    side[270336:270336 + 65536] = convw.reshape(4, 8, 1024).transpose(1, 0, 2).reshape(-1).view(np.uint8)
    sizes["side"] = wr("side_glue.bin", side)

    # ---- references out
    wr("ref_xres.bin", x_ref.astype(np.float32))
    wr("ref_xm.bin", xm_ref.astype(np.float32))
    wr("ref_xn.bin", xn_ref.astype(np.float32))
    wr("ref_S.bin", S_ref.astype(np.float32))
    wr("ref_cs.bin", cs_ref.astype(np.float32))

    cfg = [
        "device",
        f"xclbin L {D}/ln/build/final.xclbin", f"kernelx ln L {D}/ln/build/insts.bin",
        f"xclbin Q {D}/gemv_q4/build_qkv/final.xclbin", f"kernelx gqkv Q {D}/gemv_q4/build_qkv/insts.bin",
        f"xclbin Z {D}/gemv_q4/build_z/final.xclbin", f"kernelx gz Z {D}/gemv_q4/build_z/insts.bin",
        f"xclbin G {D}/dn_glue/build/final.xclbin", f"kernelx glue G {D}/dn_glue/build/insts.bin",
        f"xclbin N {D}/deltanet/build/final.xclbin", f"kernelx dn N {D}/deltanet/build/insts.bin",
        f"xclbin P {D}/dn_post/build/final.xclbin", f"kernelx post P {D}/dn_post/build/insts.bin",
        f"xclbin O {D}/gemv_q4/build_out/final.xclbin", f"kernelx gout O {D}/gemv_q4/build_out/insts.bin",
        f"buf xres {sizes['xres']} {OUT}/x_res.bin",
        f"buf zero {sizes['zero']} {OUT}/zero.bin",
        f"buf lnw {sizes['lnw']} {OUT}/lnw.bin",
        f"buf postln {sizes['postln']} {OUT}/postln.bin",
        "buf xres1 8192", "buf xn 4096",
        f"buf wqkv {sizes['wqkv']} {OUT}/w_qkv.bin",
        f"buf wz {sizes['wz']} {OUT}/w_z.bin",
        f"buf wout {sizes['wout']} {OUT}/w_out.bin",
        "buf qkv 32768", "buf z 16384",
        f"buf side {sizes['side']} {OUT}/side_glue.bin",
        f"buf state {sizes['state']} {OUT}/state.bin",
        "buf nstate 49152", "buf vec 65536",
        f"buf sin {sizes['sin']} {OUT}/s_in.bin",
        "buf sout 2097152", "buf o 16384",
        f"buf nw {sizes['nw']} {OUT}/nw.bin",
        "buf og 8192", "buf out 8192", "buf xres2 8192", "buf xm 4096",
        "run ln xres zero lnw xres1 xn",
        "run gqkv wqkv xn qkv",
        "run gz wz xn z",
        f"dump xn {OUT}/y_xn.bin 4096",
        f"load side {OUT}/y_xn.bin",
        "run glue side qkv state nstate vec",
        "run dn sin vec sout o",
        "run post o z nw og",
        "run gout wout og out",
        "run ln xres1 out postln xres2 xm",
        f"dump xres2 {OUT}/y_xres.bin 8192",
        f"dump xm {OUT}/y_xm.bin 4096",
        f"dump sout {OUT}/y_S.bin 2097152",
        f"dump nstate {OUT}/y_nstate.bin 49152",
        f"dump qkv {OUT}/y_qkv.bin 32768",
        f"dump o {OUT}/y_o.bin 16384",
        "",
    ]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"x_ref[:4]={x_ref[:4]} xm_ref[:4]={xm_ref[:4]} S_ref absmax={np.abs(S_ref).max():.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
