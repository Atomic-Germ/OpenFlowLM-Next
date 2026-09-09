"""Byte layouts for the whole-layer designs lx (linear attention + MoE) and ax
(full attention + MoE), their tests and model/make_decode.py -- every number
derived from the ModelSpec by the family recipe (open_kernels/recipes/qwen36moe.py,
`layout()`), which is also what writes the driver's manifest.json. The spec is
OPEN_KERNELS_SPEC, else the checked-in 27B (recipes/load.py).

consts, linear layer (C_BYTES):
  [lnw bf16 HID][glue side minus xn: Wa Wb small convw][nw (4 KB elem)][postln bf16 HID]
  [router W bf16 HID x E][sgw bf16 HID][out_proj q4 pool-order, RS=2]
consts, attention layer (CA_BYTES):
  [lnw][postln][meta: qn bf16 HD @0 | kn bf16 HD @HD*2 (1 KB used of the 2 KB slot)][router W][sgw]
act, linear layer (A_BYTES): the DDR bounce between stages
  [xn bf16 HID][qkv f32][z f32][vec f32 heads x 512 (DeltaNet in)][o f32 (DeltaNet out)]
  [og bf16][out f32 HID][res f32 HID + pad (residual after attention; the MoE header
  reads 10 KB slices)][xm bf16 HID][rout f32 + pad][hp f32 cores x FF/cores + pad (the expert hidden parts)]
act, attention layer (AA_BYTES):
  [xn][qg f32][kvn f32][og bf16][out f32][unused 2 KB][res + pad][xm][rout + pad][hp + pad]
kv, one BO per attention layer (KV_BYTES = MAX_CTX x KV_ROW): row t = [K_t bf16 | V_t bf16].
  The layer reads rows [0, max(pos, 1)) as ONE linear fill and writes row pos; the driver's `attnpos`
  patches the fill's length and the two offsets in the ax0 instruction stream once per token.
ptab, the position record table shared by every attention layer (PTAB_BYTES = MAX_CTX x PTAB_ROW):
  row p = [i32 pos @0 | i32 nf = max(pos, 1) @4 | cos f32 @512 | sin f32 @640], attn.h's second
  meta element (`ptab()` builds the whole table; nothing is host-written per token).
The MoE header per core is three 10 KB w-stream elements: [rout | junk] from act, [sgw | junk] from
consts, [xres slice c | junk] from act[res + c*slice].
pool offsets (recipes/qwen36moe.py, manifest.json `layout.moe` for the driver): qkv, z for linear
layers; q, k, v, gate, o for attention layers; the expert stripes / down slices / shared expert for
both (the routed slots are patched by the driver's moeroute2).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # open_kernels/
from recipes.load import current_recipe  # noqa: E402

R = current_recipe()
SPEC = R.spec
_L = R.layout

C_LNW, C_SIDE, C_NW, C_POSTLN, C_RW, C_SGW, C_WOUT = (_L.C_LNW, _L.C_SIDE, _L.C_NW, _L.C_POSTLN, _L.C_RW, _L.C_SGW,
                                                     _L.C_WOUT)
C_BYTES = _L.C_BYTES
GLUE_SIDE_BYTES = _L.GLUE_SIDE_BYTES
SIDE_ALPHA, SIDE_BETA, SIDE_SMALL, SIDE_CONV = _L.SIDE_ALPHA, _L.SIDE_BETA, _L.SIDE_SMALL, _L.SIDE_CONV
CA_LNW, CA_POSTLN, CA_META, CA_RW, CA_SGW = _L.CA_LNW, _L.CA_POSTLN, _L.CA_META, _L.CA_RW, _L.CA_SGW
CA_BYTES = _L.CA_BYTES

A_XN, A_QKV, A_Z, A_VEC, A_O, A_OG, A_OUT = _L.A_XN, _L.A_QKV, _L.A_Z, _L.A_VEC, _L.A_O, _L.A_OG, _L.A_OUT
A_RES, A_XM, A_ROUT, A_HP = _L.A_RES, _L.A_XM, _L.A_ROUT, _L.A_HP
A_BYTES = _L.A_BYTES
AA_XN, AA_QG, AA_KVN, AA_OG, AA_OUT = _L.AA_XN, _L.AA_QG, _L.AA_KVN, _L.AA_OG, _L.AA_OUT
AA_RES, AA_XM, AA_ROUT, AA_HP = _L.AA_RES, _L.AA_XM, _L.AA_ROUT, _L.AA_HP
AA_BYTES = _L.AA_BYTES

# ffn="dense" (the Qwen3.5 composition, recipes/qwen35.py): the FFN's two extra act stages
# (h = act(gate) * up, out2 = the block output) and its pool block, shared by both layer
# types; and the norm / attention helpers' element sizes, which at HID 2048 are ELEM and
# HEAD_BYTES -- the MoE designs' hardwired ones.
A_H, A_OUT2 = _L.A_H, _L.A_OUT2
AA_H, AA_OUT2 = _L.AA_H, _L.AA_OUT2
POOL_FFN_UP, POOL_FFN_GATE, POOL_FFN_DOWN = _L.POOL_FFN_UP, _L.POOL_FFN_GATE, _L.POOL_FFN_DOWN
ELN, E_A = _L.ELN, _L.E_A

# state BO (linear layers): [conv state bf16 (taps-1) x NCH][S: heads x S_ROWS rows x dim f32,
# rows dim..S_ROWS-1 zero]; S is updated in place by the layer (DeltaNet on the main cores, dnx.h).
S_ROWS, S_HEAD_BYTES = _L.S_ROWS, _L.S_HEAD_BYTES
STATE_S_OFF = _L.STATE_S_OFF
STATE_BYTES = _L.STATE_BYTES
POOL_QKV, POOL_Z = _L.POOL_QKV, _L.POOL_Z
POOL_Q, POOL_K, POOL_V, POOL_GATE, POOL_O = _L.POOL_Q, _L.POOL_K, _L.POOL_V, _L.POOL_GATE, _L.POOL_O
POOL_DOWN, POOL_SHARE_UP, POOL_SHARE_GATE, POOL_SHARE_DOWN = (_L.POOL_DOWN, _L.POOL_SHARE_UP, _L.POOL_SHARE_GATE,
                                                              _L.POOL_SHARE_DOWN)
POOL_BYTES = _L.POOL_BYTES

# the KV cache and the position record table (the driver reads them from manifest.json)
KV_ROW, PTAB_ROW, MAX_CTX = _L.KV_ROW, _L.PTAB_ROW, _L.MAX_CTX
KV_BYTES, PTAB_BYTES = _L.KV_BYTES, _L.PTAB_BYTES
LMHEAD_POOL_BYTES = _L.LMHEAD_POOL_BYTES


def ptab(max_ctx: int = MAX_CTX):
    """The position record table as bytes: row p = [pos | nf = max(p, 1) | cos | sin] for the
    partial RoPE (rotary dim / theta from the spec, the same freqs as replica.py's rope)."""
    from recipes.pack import ptab as _ptab
    return _ptab(max_ctx, SPEC.rotary_dim, SPEC.rope_theta, PTAB_ROW)
