"""Pack a layer's weights out of the `.q4nx` container into the NPU pool layout
the kernels read.

The chunk bytes are copied verbatim -- nothing is dequantized or requantized.
All that changes is the ORDER of the 5120-byte chunks, because the kernels
stream weights in the order the AIE array wants them rather than the file's
raster order:

  standard matmul tensor [out, in]: pool chunk c covers
     rows0 = 64*(c//per_band) + 32*(c%2)          per_band = in//128
     cols0 = 1024*((c//8) % max(1, in//1024)) + 256*((c//2)%4)
     file chunk f covers rows0 = 32*(f//ncol), cols0 = 256*(f%ncol)
     -> std_perm() matches the two by (rows0, cols0).
  expert up/gate: interleaved 160 KB stripes [up_k | gate_k] x4 per expert,
     each stripe internally transposed pool_c = 4*(f%8) + (f//8)
  expert / shared down [2048, 512]: pool_c = 8*(rt//4) + 4*cg + rt%4, f = 2rt+cg
  lm_head q8: its own 128-row supertile order, 32-chunk bands.

Layer pool (512 MB), byte offsets:
  0           routed experts' up/gate stripes
  335544320   routed experts' down slices (640 KB each)
  503316480   share_up   503971840  share_gate   504627200  share_down
  505282560   qkv (linear layer) / q (attention layer)
  510525440   k        511180800  v        511836160  gate
  515768320   z-gate (linear layer)        517079040  o (attention layer)

Vendored from phlegm's tools/kernel-interp/build_pools.py, which verified every
law byte-for-byte against pools captured from FLM's own engine.
"""
from __future__ import annotations

import numpy as np

CH = 5120
S = 163840
POOL_BYTES = 536_870_912
LMHEAD_POOL_BYTES = 542_113_792


def std_perm(nch, out_dim, in_dim):
    """pool chunk index -> file chunk index, for a standard matmul tensor."""
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
    """The 512 MB weight pool of one layer. `out` may be a preallocated buffer."""
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
        # q_proj is the fused [q 4096 | gate 4096]; the pool splits the halves.
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
    """[input_layernorm @0][post_attention_layernorm @4096][shared_expert_gate @8192][router @12288]"""
    pk = np.zeros(2097152, dtype=np.uint8)
    for off, name in ((0, "input_layernorm.weight"), (4096, "post_attention_layernorm.weight"),
                      (8192, "shared_expert_gate.weight"), (12288, "moe_router.weight")):
        b = np.frombuffer(m.raw(f"model.layer.{layer}.{name}"), dtype=np.uint8)
        pk[off:off + len(b)] = b
    return pk


def build_side(m, layer, full_attn):
    """The per-layer small weights: q/k norms for an attention layer; the
    DeltaNet glue's conv / norm / a / dt_bias / alpha / beta plus the permuted
    out_proj for a linear one."""
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
    """The q8 lm_head pool: 128-row supertile order, pool chunk k <- file chunk
    (4*(k//32) + (k%4))*8 + ((k%32)//4)."""
    raw = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
    k = np.arange(raw.shape[0])
    s, r = k // 32, k % 32
    perm = (4 * s + r % 4) * 8 + r // 4
    out = np.zeros(LMHEAD_POOL_BYTES, dtype=np.uint8)
    out[:raw.shape[0] * 8704] = raw[perm].reshape(-1)
    return out
