"""Byte layouts for the whole-layer designs lx (linear attention + MoE) and ax
(full attention + MoE), their tests and make_27b.py. Phase 2 "whole-layer
context" (.claude/plans/open-kernels-phase2-whole-layer.md) and dynamic KV
length (.claude/plans/open-kernels-phase2-dynamic-kv.md).

consts, linear layer (C_BYTES = 11_882_496):
  [lnw bf16 2048][glue side minus xn: Wa Wb small convw (331776)][nw (4 KB elem)][postln bf16 2048]
  [router W bf16 2048x256 (1 MB)][sgw bf16 2048][out_proj q4 pool-order, RS=2 (10 MB)]
consts, attention layer (CA_BYTES = 1_062_912):
  [lnw][postln][meta: qn bf16 256 @0 | kn bf16 256 @512 (1 KB used of the 2 KB slot)][router W (1 MB)][sgw]
act, linear layer (A_BYTES = 190_464): the DDR bounce between stages
  [xn bf16 2048][qkv f32 8192][z f32 4096][vec f32 32x512 (DeltaNet in)][o f32 4096 (DeltaNet out)]
  [og bf16 4096][out f32 2048][res f32 2048 + 12 KB pad (residual after attention; the MoE header
  reads 10 KB slices)][xm bf16 2048][rout f32 1024 + pad][hp f32 8x64 + pad (the expert hidden parts)]
act, attention layer (AA_BYTES = 98_304):
  [xn][qg f32 8192][kvn f32 1024][og bf16 4096][out f32 2048][unused 2 KB][res + pad][xm][rout + pad][hp + pad]
kv, one BO per attention layer (KV_BYTES = MAX_CTX x KV_ROW): row t = [K_t bf16 512 | V_t bf16 512] (2 KB).
  The layer reads rows [0, max(pos, 1)) as ONE linear fill and writes row pos; the driver's `attnpos`
  patches the fill's length and the two offsets in the ax0 instruction stream once per token.
ptab, the position record table shared by every attention layer (PTAB_BYTES = MAX_CTX x PTAB_ROW):
  row p = [i32 pos @0 | i32 nf = max(pos, 1) @4 | cos f32 32 @512 | sin f32 32 @640], attn.h's second
  meta element (`ptab()` builds the whole table; nothing is host-written per token).
The MoE header per core is three 10 KB w-stream elements: [rout | junk] from act, [sgw | junk] from
consts, [xres slice c (f32 256) | junk] from act[res + c*1024].
pool offsets (pools.rs): qkv, z for linear layers; q, k, v, gate, o for attention layers; the expert
stripes / down slices / shared expert for both (the routed slots are patched by the driver's moeroute2).
"""
C_LNW, C_SIDE, C_NW, C_POSTLN, C_RW, C_SGW, C_WOUT = 0, 4096, 335872, 339968, 344064, 1392640, 1396736
C_BYTES = C_WOUT + 10_485_760
GLUE_SIDE_BYTES = 331776
CA_LNW, CA_POSTLN, CA_META, CA_RW, CA_SGW = 0, 4096, 8192, 10240, 1058816
CA_BYTES = CA_SGW + 4096

A_XN, A_QKV, A_Z, A_VEC, A_O, A_OG, A_OUT = 0, 4096, 36864, 53248, 118784, 135168, 143360
A_RES, A_XM, A_ROUT, A_HP = 151552, 172032, 176128, 186368
A_BYTES = A_HP + 4096
AA_XN, AA_QG, AA_KVN, AA_OG, AA_OUT = 0, 4096, 36864, 40960, 49152
AA_RES, AA_XM, AA_ROUT, AA_HP = 59392, 79872, 83968, 94208
AA_BYTES = AA_HP + 4096

# state BO (linear layers): [conv state bf16 3x8192 (48 KB)][S: 32 heads x 140 rows x 512 B, rows
# 128..139 zero]; S is updated in place by the layer (DeltaNet on the main cores, dnx.h).
S_ROWS, S_HEAD_BYTES = 140, 140 * 512
STATE_S_OFF = 49152
STATE_BYTES = STATE_S_OFF + 32 * S_HEAD_BYTES
POOL_QKV, POOL_Z = 505_282_560, 515_768_320
POOL_Q, POOL_K, POOL_V, POOL_GATE, POOL_O = 505_282_560, 510_525_440, 511_180_800, 511_836_160, 517_079_040
POOL_DOWN, POOL_SHARE_UP, POOL_SHARE_GATE, POOL_SHARE_DOWN = 335_544_320, 503_316_480, 503_971_840, 504_627_200
POOL_BYTES = 536_870_912

# the KV cache and the position record table (mirrored in npu-engine decode.rs: ATTN_*)
KV_ROW, PTAB_ROW, MAX_CTX = 2048, 1024, 4096
KV_BYTES, PTAB_BYTES = MAX_CTX * KV_ROW, MAX_CTX * PTAB_ROW


def ptab(max_ctx: int = MAX_CTX):
    """The position record table as bytes: row p = [pos | nf = max(p, 1) | cos | sin] for the
    partial RoPE (rotary dim 64 of 256, theta 1e7, the same freqs as decode_step.py's rope1)."""
    import numpy as np

    t = np.zeros((max_ctx, PTAB_ROW), np.uint8)
    p = np.arange(max_ctx)
    t[:, :8] = np.stack([p, np.maximum(p, 1)], 1).astype(np.int32).view(np.uint8)
    ang = p[:, None] * ((1e7) ** (-np.arange(32) / 32))[None, :]
    t[:, 512:640] = np.cos(ang).astype(np.float32).view(np.uint8)
    t[:, 640:768] = np.sin(ang).astype(np.float32).view(np.uint8)
    return t.reshape(-1)
