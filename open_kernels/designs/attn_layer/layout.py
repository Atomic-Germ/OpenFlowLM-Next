"""Byte layouts for attn_l (the fused full-attention layer), its test and make_27b.py.

consts (per layer, 10240 B):  [lnw bf16 2048][postln bf16 2048][meta 2048: attn.h's record]
act (per layer scratch, 59392 B):
  [xn bf16 2048][qg f32 8192 = q | gate][kvn f32 1024 = k | v][og bf16 4096][out f32 2048][kvnew bf16 1024]
  (the GEMV x fill of xn reads 8 KB from 0: xn plus the first 4 KB of qg, unread)
hdr: the MoE header, as lin_layer/layout.py (xm at 0, residual at 12288).
pool offsets (pools.rs, full-attention layers): q, k, v, gate, o.
"""
CA_LNW, CA_POSTLN, CA_META, CA_BYTES = 0, 4096, 8192, 10240
AA_XN, AA_QG, AA_KVN, AA_OG, AA_OUT, AA_KVNEW, AA_BYTES = 0, 4096, 36864, 40960, 49152, 57344, 59392
POOL_Q, POOL_K, POOL_V, POOL_GATE, POOL_O = 505_282_560, 510_525_440, 511_180_800, 511_836_160, 517_079_040
KV_BYTES, KV_V_OFF = 3_145_728, 1_073_152
# mirrored from lin_layer/layout.py (both modules are named `layout`; keep in sync)
H_XM, H_ROUT, H_SGW, H_XRES, H_BYTES = 0, 4096, 8192, 12288, 20480
POOL_BYTES = 536_870_912
