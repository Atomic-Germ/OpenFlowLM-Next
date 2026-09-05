"""Byte layouts shared by lin_a / lin_c, their tests and make_27b.py.

consts (per layer, 344064 B):
  [lnw bf16 2048][Wa bf16 2048x32][Wb][small f32: A[32] dt_bias[32] pad][convw [8 tiles][4][1024] bf16]
  [nw bf16 128 (4 KB elem)][postln bf16 2048]
  -- elements 1..81 (offset 4096, 331776 B) are exactly dn_glue's side blob minus its xn slot.
act (per layer scratch, the DDR bounce between stages, 69632 B):
  [xn bf16 2048][qkv f32 8192][z f32 4096][og bf16 4096][out f32 2048]
hdr (lin_c output == moe_experts header, 20480 B):
  [xm bf16 2048][rout f32 1024 (router)][sgw bf16 2048 (constant)][xres f32 2048]
"""
C_LNW, C_WA, C_WB, C_SMALL, C_CONVW, C_NW, C_POSTLN, C_BYTES = 0, 4096, 135168, 266240, 270336, 335872, 339968, 344064
GLUE_SIDE_OFF, GLUE_SIDE_BYTES = 4096, 331776
A_XN, A_QKV, A_Z, A_OG, A_OUT, A_BYTES = 0, 4096, 36864, 53248, 61440, 69632
H_XM, H_ROUT, H_SGW, H_XRES, H_BYTES = 0, 4096, 8192, 12288, 20480
POOL_QKV, POOL_Z, POOL_BYTES = 505_282_560, 515_768_320, 536_870_912
