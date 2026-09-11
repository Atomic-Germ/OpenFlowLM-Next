"""The ORIGINAL hand-written packers, frozen as the oracle for the plan-driven
interpreter (recipes/pack.py). These were open_kernels/model/pools.py
(build_layer_pool / build_pack / build_side / build_lmhead_pool) and
model/make_decode.py's layer_consts, byte-verified against pools captured
from OFLM's own engine (phlegm's tools/kernel-interp/build_pools.py). Do not
"improve" them: they are the definition of correct.

Traces: OPEN-PACK-PLAN (canonical spec: specs/open-engine/spec.md)
"""
from __future__ import annotations

import numpy as np

CH = 5120
S = 163840
POOL_BYTES = 536_870_912
LMHEAD_POOL_BYTES = 542_113_792

# layout.py's constants as they were on 2026-09-05 (designs/layer_x/layout.py, hand-written)
C_LNW, C_SIDE, C_NW, C_POSTLN, C_RW, C_SGW, C_WOUT = 0, 4096, 335872, 339968, 344064, 1392640, 1396736
C_BYTES = C_WOUT + 10_485_760
GLUE_SIDE_BYTES = 331776
CA_LNW, CA_POSTLN, CA_META, CA_RW, CA_SGW = 0, 4096, 8192, 10240, 1058816
CA_BYTES = CA_SGW + 4096


def std_perm(nch, out_dim, in_dim):
    ncol = in_dim // 256
    per_band = in_dim // 128
    kgroups = max(1, in_dim // 1024)
    c = np.arange(nch)
    rows0 = 64 * (c // per_band) + 32 * (c % 2)
    cols0 = (1024 * ((c // 8) % kgroups) + 256 * ((c // 2) % 4)) % in_dim
    return (rows0 // 32) * ncol + cols0 // 256


def permute_chunks(raw, perm):
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, CH)[perm].reshape(-1)


def down_perm():
    c = np.arange(128)
    rt = 4 * (c // 8) + (c % 4)
    return 2 * rt + (c // 4) % 2


def stripe_transpose():
    c = np.arange(32)
    return 8 * (c % 4) + c // 4


def build_layer_pool(m, layer, full_attn, out=None):
    pool = np.zeros(POOL_BYTES, dtype=np.uint8) if out is None else out
    if out is not None:
        pool[:] = 0
    up = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.up_exps_proj.weight"), dtype=np.uint8)
    gt = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.gate_exps_proj.weight"), dtype=np.uint8)
    dn = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.down_exps_proj.weight"), dtype=np.uint8)
    tp = stripe_transpose()
    for e in range(256):
        for k in range(4):
            src = (4 * e + k) * S
            pool[(8 * e + 2 * k) * S:(8 * e + 2 * k + 1) * S] = up[src:src + S].reshape(32, CH)[tp].reshape(-1)
            pool[(8 * e + 2 * k + 1) * S:(8 * e + 2 * k + 2) * S] = gt[src:src + S].reshape(32, CH)[tp].reshape(-1)
    dp = down_perm()
    for e in range(256):
        seg = dn[e * 655360:(e + 1) * 655360].reshape(128, CH)[dp].reshape(-1)
        pool[335544320 + e * 655360: 335544320 + (e + 1) * 655360] = seg
    p128 = std_perm(128, 512, 2048)
    pool[503316480:503316480 + 655360] = permute_chunks(
        m.raw(f"model.layer.{layer}.mlp.share_up_exps_proj.weight"), p128)
    pool[503971840:503971840 + 655360] = permute_chunks(
        m.raw(f"model.layer.{layer}.mlp.share_gate_exps_proj.weight"), p128)
    pool[504627200:504627200 + 655360] = permute_chunks(
        m.raw(f"model.layer.{layer}.mlp.share_down_exps_proj.weight"), std_perm(128, 2048, 512))
    if full_attn:
        qg = np.frombuffer(m.raw(f"model.layer.{layer}.self_attn.q_proj.weight"), dtype=np.uint8).reshape(-1, CH)
        p1024 = std_perm(1024, 4096, 2048)
        pool[505282560:505282560 + 5242880] = qg[:1024][p1024].reshape(-1)
        pool[510525440:510525440 + 655360] = permute_chunks(
            m.raw(f"model.layer.{layer}.self_attn.k_proj.weight"), std_perm(128, 512, 2048))
        pool[511180800:511180800 + 655360] = permute_chunks(
            m.raw(f"model.layer.{layer}.self_attn.v_proj.weight"), std_perm(128, 512, 2048))
        pool[511836160:511836160 + 5242880] = qg[1024:][p1024].reshape(-1)
        pool[517079040:517079040 + 5242880] = permute_chunks(
            m.raw(f"model.layer.{layer}.self_attn.o_proj.weight"), std_perm(1024, 2048, 4096))
    else:
        pool[505282560:505282560 + 10485760] = permute_chunks(
            m.raw(f"model.layer.{layer}.linear_attn.qkv_proj.weight"), std_perm(2048, 8192, 2048))
        pool[515768320:515768320 + 5242880] = permute_chunks(
            m.raw(f"model.layer.{layer}.self_attn.gate_proj.weight"), std_perm(1024, 4096, 2048))
    return pool


def build_pack(m, layer):
    pk = np.zeros(2097152, dtype=np.uint8)
    for off, name in ((0, "input_layernorm.weight"), (4096, "post_attention_layernorm.weight"),
                      (8192, "shared_expert_gate.weight"), (12288, "moe_router.weight")):
        b = np.frombuffer(m.raw(f"model.layer.{layer}.{name}"), dtype=np.uint8)
        pk[off:off + len(b)] = b
    return pk


def build_side(m, layer, full_attn):
    side = np.zeros(6291456, dtype=np.uint8)

    def put(off, name):
        b = np.frombuffer(m.raw(f"model.layer.{layer}.{name}"), dtype=np.uint8)
        side[off:off + len(b)] = b

    if full_attn:
        put(128, "self_attn.q_norm.weight")
        put(640, "self_attn.k_norm.weight")
    else:
        put(0, "linear_attn.ssm_conv1d.weight")       # 65536
        put(65536, "linear_attn.ssm_norm.weight")     # 256
        put(65792, "linear_attn.ssm_a")               # 128 (f32)
        put(65920, "linear_attn.ssm_dt.bias")         # 128
        put(66048, "linear_attn.ssm_alpha_proj.weight")   # 131072
        put(197120, "linear_attn.ssm_beta_proj.weight")   # 131072
        side[328192:328192 + 5242880] = permute_chunks(
            m.raw(f"model.layer.{layer}.linear_attn.ssm_out_proj.weight"), std_perm(1024, 2048, 4096))
    return side


def build_lmhead_pool(m):
    raw = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
    k = np.arange(raw.shape[0])
    s, r = k // 32, k % 32
    perm = (4 * s + r % 4) * 8 + r // 4
    out = np.zeros(LMHEAD_POOL_BYTES, dtype=np.uint8)
    out[:raw.shape[0] * 8704] = raw[perm].reshape(-1)
    return out


def layer_consts(m, layer, full_attn):
    """make_decode.py's layer_consts as it was: the consts BO from build_pack + build_side."""
    pack = np.frombuffer(build_pack(m, layer), np.uint8)
    side = np.frombuffer(build_side(m, layer, full_attn), np.uint8)
    lnw, postln = pack[0:4096], pack[4096:8192]
    sgw, rw = pack[8192:12288], pack[12288:12288 + 1048576]
    if full_attn:
        c = np.zeros(CA_BYTES, np.uint8)
        c[CA_LNW:CA_LNW + 4096] = lnw
        c[CA_POSTLN:CA_POSTLN + 4096] = postln
        c[CA_META:CA_META + 512] = side[128:640]
        c[CA_META + 512:CA_META + 1024] = side[640:1152]
        c[CA_RW:CA_RW + 1048576] = rw
        c[CA_SGW:CA_SGW + 4096] = sgw
        return c
    sb = np.zeros(4096 + GLUE_SIDE_BYTES, np.uint8)
    sb[4096:4096 + 131072] = side[66048:66048 + 131072]
    sb[135168:135168 + 131072] = side[197120:197120 + 131072]
    small = np.zeros(1024, np.float32)
    small[:32] = side[65792:65792 + 128].view(np.float32)
    small[32:64] = side[65920:65920 + 128].view(np.float32)
    sb[266240:266240 + 4096] = small.view(np.uint8)
    convw = side[0:65536].view(np.uint16).reshape(4, 8192)
    sb[270336:270336 + 65536] = convw.reshape(4, 8, 1024).transpose(1, 0, 2).reshape(-1).view(np.uint8)
    nw = np.zeros(2048, np.uint16)
    nw[:128] = side[65536:65536 + 256].view(np.uint16)
    c = np.zeros(C_BYTES, np.uint8)
    c[C_LNW:C_LNW + 4096] = lnw
    c[C_SIDE:C_SIDE + GLUE_SIDE_BYTES] = sb[4096:]
    c[C_NW:C_NW + 4096] = nw.view(np.uint8)
    c[C_POSTLN:C_POSTLN + 4096] = postln
    c[C_RW:C_RW + 1048576] = rw
    c[C_SGW:C_SGW + 4096] = sgw
    c[C_WOUT:C_WOUT + 5242880] = side[328192:328192 + 5242880]
    return c


def ptab(max_ctx=4096):
    """layout.py's ptab() as it was (theta 1e7, rotary 64 of 256)."""
    t = np.zeros((max_ctx, 1024), np.uint8)
    p = np.arange(max_ctx)
    t[:, :8] = np.stack([p, np.maximum(p, 1)], 1).astype(np.int32).view(np.uint8)
    ang = p[:, None] * ((1e7) ** (-np.arange(32) / 32))[None, :]
    t[:, 512:640] = np.cos(ang).astype(np.float32).view(np.uint8)
    t[:, 640:768] = np.sin(ang).astype(np.float32).view(np.uint8)
    return t.reshape(-1)
