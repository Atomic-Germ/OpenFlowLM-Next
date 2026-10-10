"""The dense L-row pass's layout: dx's per-layer buffers reused, and both passes' job tables so verify and draft share an xclbin."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .catalogue import OpRangeError
from .dense import ELEM, geometry, layout as dense_layout, qkv_bias
from .spec import ModelSpec

BAND_ROWS = 64
KS_BF16 = 1024      # a bf16 activation slice: one 2 KB x element per token
KS_BF16_HALF = 512  # where K is an odd multiple of 512: half an element, the other half over-read and unused
KS_F32 = 512        # an fp32 one (the down projection's h)
XE = 2048           # bytes of one x element
L1_BUDGET = 60 * 1024
STACK = 0x1800
CHUNK = 5120
Z_ROW = 2048        # one row of z: 512 fp32, one x element
LORA_K = 256        # the B tile's width
JMAX = 32           # jobs a table holds
# a job's fields, as the cores' table stores them (dxl_job.cc reads the same order)
FIELDS = ("S", "BT", "CPS", "NDRAIN", "KS", "MODE", "OFF", "B0", "S0", "KT", "DMODE", "G0")
NF = len(FIELDS)
PREP_BF16, PREP_F32, PREP_Z = 0, 1, 2          # MODE: the slice's activation format
OUT_COPY, OUT_ACT = 0, 1                       # DMODE: a band as is, or silu(gate) * up
LORA_TENSORS = ("a_qkv", "a_o", "a_gu", "a_d", "b_q", "b_k", "b_v", "b_o", "b_g", "b_u", "b_d")


def tab_bytes(k: int) -> int:
    return 2 * k + k // 4


def ks_bf16(k: int) -> int:
    return KS_BF16 if k % KS_BF16 == 0 else KS_BF16_HALF


def head_act_bytes(spec: ModelSpec, L: int) -> int:
    """lmhl's act: L bf16 rows of the hidden, padded so the last row's last whole x element is in the buffer."""
    return L * spec.hidden * 2 + (-spec.hidden * 2) % XE


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
    AD_ZQKV: int
    AD_ZO: int
    AD_ZGU: int
    AD_ZD: int
    AD_ZERO: int     # one x element nothing writes: the seed row's z
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
    BTQ: int         # q, o and down tiles; k and v are one tile of KV_PC each
    BTO: int
    BTD: int
    BP: int          # up | gate band pairs per tile
    YB: int          # accumulator bands before the gate bands
    KS_H: int        # the bf16 slices over the hidden and over the q width
    KS_Q: int
    TAB_K: int       # the widest slice, which sizes the cores' slice tables
    # the LoRA pool (per layer): tensor -> (offset, rows, cols)
    LORA: dict = field(default_factory=dict)
    LORA_BYTES: int = 0


def _main_l1(L: int, bt: int, bp: int, kv_pc: int = 2, tab_k: int = KS_BF16) -> int:
    ye = L * BAND_ROWS * 4
    return (STACK + 2 * CHUNK + 2 * XE + L * tab_bytes(tab_k)
            + (max(bt, bp, kv_pc) + bp) * ye + 2 * ye + 2 * JMAX * NF * 4)


def _band_bytes(cols: int) -> int:
    return 2 * (cols // 256) * CHUNK


def lora_shapes(spec: ModelSpec, n_cores: int) -> dict[str, tuple[int, int]]:
    G = geometry(spec)
    hid, ff, qw, kvw = spec.hidden, spec.intermediate, G.QW, G.KVW
    a = n_cores * BAND_ROWS
    return {"a_qkv": (a, hid), "a_o": (a, qw), "a_gu": (a, hid), "a_d": (a, ff),
            "b_q": (qw, LORA_K), "b_k": (kvw, LORA_K), "b_v": (kvw, LORA_K), "b_o": (hid, LORA_K),
            "b_g": (ff, LORA_K), "b_u": (ff, LORA_K), "b_d": (hid, LORA_K)}


def layout(spec: ModelSpec, L: int) -> DxlLayout:
    if L % 4 or not 4 <= L <= 8:
        raise OpRangeError(f"dxl: L={L}; the tile runs four tokens per mmul, so L is 4 or 8")
    if spec.sandwich_norms or qkv_bias(spec) or spec.attn_gate or spec.q8_roles or spec.has_local:
        raise OpRangeError("dxl: only the plain dense layer (no sandwich norms, qkv bias, gate, q8 or "
                           "sliding window) has an L-row pass yet")
    G, D = geometry(spec), dense_layout(spec)
    hid, ff, qw, kvw = spec.hidden, spec.intermediate, G.QW, G.KVW
    for what, k, ks in (("hidden", hid, KS_BF16_HALF), ("q width", qw, KS_BF16_HALF), ("intermediate", ff, KS_F32)):
        if k % ks:
            raise OpRangeError(f"dxl: {what} {k} is not a multiple of its slice {ks}")
    ks_h, ks_q = ks_bf16(hid), ks_bf16(qw)
    tab_k = max(ks_h, ks_q, KS_F32)

    def tile(n: int, cap: int, fits) -> int:
        return next(b for b in range(min(n, cap), 0, -1) if n % b == 0 and fits(b))

    # each projection's own largest tile: Granite's 5 bands a core must not fall to tiles of 1 for K2's 8
    btq, bto, btd = (tile(n, 8, lambda b: _main_l1(L, b, 1, G.KV_PC, tab_k) <= L1_BUDGET)
                     for n in (G.Q_PC, G.O_PC, G.DOWN_PC))
    bt = max(btq, bto, btd)
    bp = tile(G.UP_PC, 6, lambda b: _main_l1(L, bt, b, G.KV_PC, tab_k) <= L1_BUDGET)
    rows = {"xn": hid * 2, "q": qw * 4, "k": kvw * 4, "v": kvw * 4, "og": qw * 2, "out": hid * 4,
            "res": hid * 4, "xm": hid * 2, "h": ff * 4, "out2": hid * 4,
            "zqkv": Z_ROW, "zo": Z_ROW, "zgu": Z_ROW, "zd": Z_ROW}
    a, off = {}, 0
    for name, row in rows.items():
        a[name] = off
        off += L * row
    a["zero"] = off
    off += XE
    a["junk"] = off
    off += D.ELN
    lora, loff = {}, 0
    for name, (r, c) in lora_shapes(spec, G.N_CORES).items():
        lora[name] = (loff, r, c)
        loff += (r // BAND_ROWS) * _band_bytes(c)
    return DxlLayout(
        L=L, AD_XN=a["xn"], AD_Q=a["q"], AD_K=a["k"], AD_V=a["v"], AD_OG=a["og"], AD_OUT=a["out"],
        AD_RES=a["res"], AD_XM=a["xm"], AD_H=a["h"], AD_OUT2=a["out2"], AD_ZQKV=a["zqkv"], AD_ZO=a["zo"],
        AD_ZGU=a["zgu"], AD_ZD=a["zd"], AD_ZERO=a["zero"], AD_JUNK=a["junk"],
        AD_BYTES=-(-off // ELEM) * ELEM,
        XN_ROW=rows["xn"], Q_ROW=rows["q"], KV_ROW_ACT=rows["k"], OG_ROW=rows["og"], HID_ROW=hid * 4,
        XM_ROW=rows["xm"], H_ROW=rows["h"], XRES_FLOATS=L * hid, BTQ=btq, BTO=bto, BTD=btd, BP=bp,
        YB=max(bt, bp, G.KV_PC), KS_H=ks_h, KS_Q=ks_q, TAB_K=tab_k,
        LORA=lora, LORA_BYTES=-(-loff // (1 << 20)) * (1 << 20))


@dataclass(frozen=True)
class Job:
    """One pass of the main cores over `bt` bands: `core` is its table row, the rest place its DMA."""
    stage: str
    core: dict
    wbuf: str
    woff: int
    wpc: int
    t: int
    bb: int
    x: tuple
    y: tuple | None


def jobs(spec: ModelSpec, L: int, draft: bool) -> list[Job]:
    """The main cores' jobs for one layer, in the order they run them."""
    G, D, X = geometry(spec), dense_layout(spec), layout(spec, L)
    hid, ff, qw = spec.hidden, spec.intermediate, G.QW
    out: list[Job] = []
    lt = 1 if draft else 0                 # the LoRA k-tile every base band gains in a draft

    def core(S, BT, KS, mode, ndrain=0, off=0, b0=0, s0=0, kt=0, dmode=OUT_COPY, g0=0):
        return {"S": S, "BT": BT, "CPS": 2 * KS // 256, "NDRAIN": ndrain, "KS": KS, "MODE": mode, "OFF": off,
                "B0": b0, "S0": s0, "KT": kt, "DMODE": dmode, "G0": g0}

    def a_job(stage, name, xoff, xrow, K, KS, f32, zoff):
        loff, _, c = X.LORA[name]
        out.append(Job(stage, core(K // KS, 1, KS, PREP_F32 if f32 else PREP_BF16, 1, kt=K // 256), "lora",
                       loff, 1, 0, _band_bytes(c), ("act", xoff, xrow, K, KS, f32), (zoff, Z_ROW)))

    def proj(stage, pool_off, nb, bt, K, KS, f32, xoff, xrow, yoff, yrow, lora=None, b0=0, drain=True,
             dmode=OUT_COPY, g0=0):
        """nb bands per core in tiles of bt: the base slices, then (draft) the LoRA tile."""
        for t in range(nb // bt):
            kt = K // 256 + lt
            dn = bt if drain else 0
            base = core(K // KS, bt, KS, PREP_F32 if f32 else PREP_BF16, 0 if lora else dn, b0=b0, kt=kt,
                        dmode=dmode, g0=g0)
            y = (yoff, yrow) if (drain and not lora) else None
            out.append(Job(stage, base, "pool", pool_off, nb, t, _band_bytes(K), ("act", xoff, xrow, K, KS, f32), y))
            if lora:
                name, zoff, zcol = lora
                loff, _, c = X.LORA[name]
                lj = core(1, bt, LORA_K, PREP_Z, dn, off=zcol, b0=b0, s0=K // 256, kt=kt, dmode=dmode, g0=g0)
                out.append(Job(stage, lj, "lora", loff, nb, t, _band_bytes(c), ("z", zoff),
                               (yoff, yrow) if drain else None))

    BP, YB, KH, KQ = X.BP, X.YB, X.KS_H, X.KS_Q
    # q | k | v
    if draft:
        a_job("qkv", "a_qkv", X.AD_XN, X.XN_ROW, hid, KH, False, X.AD_ZQKV)
    proj("qkv", D.POOL_Q, G.Q_PC, X.BTQ, hid, KH, False, X.AD_XN, X.XN_ROW, X.AD_Q, X.Q_ROW,
         ("b_q", X.AD_ZQKV, 0) if draft else None)
    proj("qkv", D.POOL_K, G.KV_PC, G.KV_PC, hid, KH, False, X.AD_XN, X.XN_ROW, X.AD_K, X.KV_ROW_ACT,
         ("b_k", X.AD_ZQKV, 0) if draft else None)
    proj("qkv", D.POOL_V, G.KV_PC, G.KV_PC, hid, KH, False, X.AD_XN, X.XN_ROW, X.AD_V, X.KV_ROW_ACT,
         ("b_v", X.AD_ZQKV, LORA_K) if draft else None)
    # o
    if draft:
        a_job("o", "a_o", X.AD_OG, X.OG_ROW, qw, KQ, False, X.AD_ZO)
    proj("o", D.POOL_O, G.O_PC, X.BTO, qw, KQ, False, X.AD_OG, X.OG_ROW, X.AD_OUT, X.HID_ROW,
         ("b_o", X.AD_ZO, 0) if draft else None)
    # up | gate per tile of BP pairs: up into bands 0.., gate into YB.., then h = silu(gate) * up
    if draft:
        a_job("ug", "a_gu", X.AD_XM, X.XM_ROW, hid, KH, False, X.AD_ZGU)
    for t in range(G.UP_PC // BP):
        for which, pool_off, name, zcol in (("up", D.POOL_UP, "b_u", 0), ("gate", D.POOL_GATE, "b_g", 0)):
            gate = which == "gate"
            kt = hid // 256 + lt
            base = core(hid // KH, BP, KH, PREP_BF16, BP if (gate and not draft) else 0,
                        b0=YB if gate else 0, kt=kt, dmode=OUT_ACT, g0=YB)
            out.append(Job("ug", base, "pool", pool_off, G.UP_PC, t, _band_bytes(hid),
                           ("act", X.AD_XM, X.XM_ROW, hid, KH, False),
                           (X.AD_H, X.H_ROW) if (gate and not draft) else None))
            if draft:
                loff, _, c = X.LORA[name]
                lj = core(1, BP, LORA_K, PREP_Z, BP if gate else 0, off=zcol, b0=YB if gate else 0,
                          s0=hid // 256, kt=kt, dmode=OUT_ACT, g0=YB)
                out.append(Job("ug", lj, "lora", loff, G.UP_PC, t, _band_bytes(c), ("z", X.AD_ZGU),
                               (X.AD_H, X.H_ROW) if gate else None))
    # down
    if draft:
        a_job("down", "a_d", X.AD_H, X.H_ROW, ff, KS_F32, True, X.AD_ZD)
    proj("down", D.POOL_DOWN, G.DOWN_PC, X.BTD, ff, KS_F32, True, X.AD_H, X.H_ROW, X.AD_OUT2, X.HID_ROW,
         ("b_d", X.AD_ZD, 0) if draft else None)
    if len(out) > JMAX:
        raise OpRangeError(f"dxl: {len(out)} jobs, the table holds {JMAX}")
    return out


def job_table(spec: ModelSpec, L: int) -> np.ndarray:
    """int32 [2][1 + JMAX * NF]: per mode (0 verify, 1 draft) the job count, then each job's FIELDS."""
    t = np.zeros((2, 1 + JMAX * NF), np.int32)
    for mode, draft in ((0, False), (1, True)):
        js = jobs(spec, L, draft)
        t[mode, 0] = len(js)
        for j, job in enumerate(js):
            t[mode, 1 + j * NF:1 + (j + 1) * NF] = [job.core[f] for f in FIELDS]
    return t


def lora_pack_plan(spec: ModelSpec, L: int) -> list[dict]:
    """The LoRA pool's pack ops: each uno.q4nx tensor as a std_perm band region."""
    X = layout(spec, L)
    return [{"op": "std_perm", "tensor": f"model.layers.{{l}}.uno.{name}.weight", "dst": off,
             "nch": (r // 32) * (c // 256), "in_dim": c}
            for name, (off, r, c) in X.LORA.items()]


@dataclass(frozen=True)
class HeadLayout:
    """lmhl2's split: every core the same BPC bands of the (padded) head pool, in tiles of BT."""
    CORES: int
    BPC: int
    BT: int
    ROW_FLOATS: int     # logits row stride: every core's bands, the padding ones included
    OUT_FLOATS: int     # L logits rows, then CORES argmax elements of L * 64


def head_layout(spec: ModelSpec, L: int) -> HeadLayout:
    from .dense import cores_for, lm_rows
    n = cores_for(spec)
    bands = lm_rows(spec) // BAND_ROWS
    bpc = -(-bands // n)                    # the head pool is padded to n * bpc bands (dense.layout)
    ye = L * BAND_ROWS * 4
    fixed = 0x1000 + L * tab_bytes(KS_BF16) + 2 * 2 * CHUNK + 2 * XE + 2 * ye + 2 * 64
    bt = max(b for b in range(1, bpc + 1) if bpc % b == 0 and fixed + b * ye <= L1_BUDGET)
    row = n * bpc * BAND_ROWS
    return HeadLayout(CORES=n, BPC=bpc, BT=bt, ROW_FLOATS=row, OUT_FLOATS=L * row + n * L * 64)
