"""The dense L-row pass (designs/dxl/dxl.py): L consecutive positions through one dense
layer in one dispatch, the weights streamed once for all L.

It reads the SAME per-layer buffers as dx (pool, consts, kv, ptab), so a kernel set
that has dx can add dxl without a second copy of anything; what it adds is its own
L-row activation scratch (`act`) and L-row residual (`xres`), laid out here.

Every act region is L rows of the width dx keeps one of, row-major ([row][width]),
so a row is where an attention core or the next projection expects it."""
from __future__ import annotations

from dataclasses import dataclass

from .catalogue import OpRangeError
from .dense import ELEM, geometry, layout as dense_layout, qkv_bias
from .spec import ModelSpec

BAND_ROWS = 64
KS_BF16 = 1024      # a bf16 activation slice: one 2 KB x element per token
KS_F32 = 512        # an fp32 one (the down projection's h)
XE = 2048           # bytes of one x element
L1_BUDGET = 60 * 1024
STACK = 0x1800
CHUNK = 5120


def tab_bytes(k: int) -> int:
    return 2 * k + k // 4


@dataclass(frozen=True)
class DxlLayout:
    L: int
    # act regions (bytes), each [L][width]
    AD_XN: int
    AD_Q: int
    AD_K: int
    AD_V: int
    AD_OG: int
    AD_OUT: int
    AD_RES: int
    AD_XM: int
    AD_H: int
    AD_OUT2: int
    AD_JUNK: int
    AD_BYTES: int
    XN_ROW: int      # bytes per row of each region
    Q_ROW: int
    KV_ROW_ACT: int
    OG_ROW: int
    HID_ROW: int
    XM_ROW: int
    H_ROW: int
    XRES_FLOATS: int
    # main-core tiling: bands per tile (resident accumulators)
    BT: int          # q / o / down tiles; k and v are one tile of KV_PC each
    BP: int          # up | gate band pairs per tile


def _main_l1(L: int, bt: int, bp: int, kv_pc: int = 2) -> int:
    ye = L * BAND_ROWS * 4
    return (STACK + 2 * CHUNK + 2 * XE + L * tab_bytes(KS_BF16)
            + (max(bt, bp, kv_pc) + bp) * ye + 2 * ye)


def layout(spec: ModelSpec, L: int) -> DxlLayout:
    if L % 4 or not 4 <= L <= 8:
        raise OpRangeError(f"dxl: L={L}; the tile runs four tokens per mmul, so L is 4 or 8")
    if spec.sandwich_norms or qkv_bias(spec) or spec.attn_gate or spec.q8_roles or spec.has_local:
        raise OpRangeError("dxl: only the plain dense layer (no sandwich norms, qkv bias, gate, q8 or "
                           "sliding window) has an L-row pass yet")
    G, D = geometry(spec), dense_layout(spec)
    hid, ff, qw, kvw = spec.hidden, spec.intermediate, G.QW, G.KVW
    for what, k, ks in (("hidden", hid, KS_BF16), ("q width", qw, KS_BF16), ("intermediate", ff, KS_F32)):
        if k % ks:
            raise OpRangeError(f"dxl: {what} {k} is not a multiple of its slice {ks}")
    bt = next(b for b in (8, 4, 2, 1) if G.Q_PC % b == 0 and G.O_PC % b == 0 and G.DOWN_PC % b == 0
              and _main_l1(L, b, 1, G.KV_PC) <= L1_BUDGET)
    bp = next(b for b in (6, 4, 3, 2, 1) if G.UP_PC % b == 0 and _main_l1(L, bt, b, G.KV_PC) <= L1_BUDGET)
    rows = {"xn": hid * 2, "q": qw * 4, "k": kvw * 4, "v": kvw * 4, "og": qw * 2, "out": hid * 4,
            "res": hid * 4, "xm": hid * 2, "h": ff * 4, "out2": hid * 4}
    a, off = {}, 0
    for name, row in rows.items():
        a[name] = off
        off += L * row
    a["junk"] = off
    off += D.ELN
    return DxlLayout(
        L=L, AD_XN=a["xn"], AD_Q=a["q"], AD_K=a["k"], AD_V=a["v"], AD_OG=a["og"], AD_OUT=a["out"],
        AD_RES=a["res"], AD_XM=a["xm"], AD_H=a["h"], AD_OUT2=a["out2"], AD_JUNK=a["junk"],
        AD_BYTES=-(-off // ELEM) * ELEM,
        XN_ROW=rows["xn"], Q_ROW=rows["q"], KV_ROW_ACT=rows["k"], OG_ROW=rows["og"], HID_ROW=hid * 4,
        XM_ROW=rows["xm"], H_ROW=rows["h"], XRES_FLOATS=L * hid, BT=bt, BP=bp)
