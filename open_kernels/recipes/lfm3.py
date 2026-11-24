"""LiquidAI LFM3 (OPEN-FAMILY-LFM3; LFM3-1.7B) on the open engine, from the GGUF.

LFM2's hybrid -- most layers a gated short depthwise causal conv, the rest GQA attention -- with a dense FFN
on every layer, a sandwich norm on every block output, a sigmoid output gate on attention and two attention
kinds (full: theta 1e6 and the first half of the head rotated; sliding: theta 1e4, every pair, 256 rows). A
layer is two dispatches (designs/lfm3/lg.py), both over ONE pool:

  mixer   one of two programs over POOL_MIX, the same offset in every layer's pool:
    lg_conv  ln -> B | C | u (in_proj's three row slices) -> the conv core (7 taps; bx = B u; conv =
             sum_k s_k w_k + bx w_6, the LAST tap on this token; y = C conv; the state slides) -> out ->
             t = post_operator_norm(out); res = x + t; xm = ffn_norm(res). State: the `linear` kind, 6 x
             hidden f32.
    lg_att   (attnpos) ln -> q | gate | k | v -> per-head RMSNorm of q, k -> RoPE (the position record: the
             full layers' table has frequency 0 on the second half of the pairs, which rotates by nothing)
             -> causal attention (the sliding layers' kernel entry carries the 256-row window) -> * the
             sigmoid gate (attn.h ATTN_GATE) -> o -> the same close.
  lg_mlp   the dense SiLU MLP as 5 unweighted slots of 1536 (gate | up per band, h = silu(gate) * up, the
           down bands summed into each core's accumulator) -> t = post_ffn_norm(down); xres = res + t.

Then the final norm (`ln`, output_norm) and the q8 head (`lm_head_q8`, f32 scales over the tied token table).

Weights. The GGUF is derived from LiquidAI's QAD Q4_0 checkpoint (model/lfm3_gguf_make.py): token_embd is
re-encoded Q6_K -> Q8_0, and each attention layer's per-head gate row is repeated head_dim times so the gate
is a q-width projection like any other (attn.h's ATTN_GATE applies it per element). Every Q4_0 scale in the
file is a bf16 value, so the q4_1 pool law (5120-B chunks, d and m = -8 d as bf16; fl_q4) holds every matrix
exactly. Norms and q / k norms go to bf16 in the consts; the conv taps stay f32 (tap-major).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import BAND_ROWS, ELEM, MB, roundup, tab_bytes
from .spec import DENSE_LOCAL, FULL, SHORT_CONV, ModelSpec

CH = 5120                 # a q4_1 chunk: 32 rows x 256 K, bf16 d | m planes (gemv_q4.h)
CH_Q40 = 4608             # a Q4_0-native chunk: the bf16 d plane and the codes (m = -8 d on the core; the default)


def q40_on() -> bool:
    """Q4_0-native pools (opt-in with LFM3_Q40=1; the default is the q4_1 law): the QAD Q4_0's scales are bf16
    values, so 4608-B chunks (no m plane) hold every matrix exactly in 10 % fewer bytes than the q4_1 law -- but
    they need the q4_0 pool law in every reader (the pack ops, gemm_q4_prefill's FORMATS, the engine's chunk
    table), and this tree's readers hold the q4_1 law. The q4_1 path re-encodes the Q4_0 source exactly
    (m = -8 d as bf16), so it is the same arithmetic at 10 % more bytes."""
    return os.environ.get("LFM3_Q40") == "1"


def chunk() -> int:
    """The pool's chunk law in bytes."""
    return CH_Q40 if q40_on() else CH
PER_CALL = 2              # chunks per w element (10240 B)
N_CORES = 8
ATTN_CORES = 4            # attnknobs.attn_cores(16): the largest divisor of 16 heads <= 6
STACK = 0x1800
LM_CORES = 8
LM_BAND_ROWS = 128        # lm_head_q8: 128-row bands
LM_TILE = 9216            # lm_head_q8.h's f32-scale chunk: 32 rows x 256 K
LM_TILE_BF16 = 8704       # ... its bf16-scale chunk (the default; LFM3_HEAD=f32 for the other)


def head_bf16() -> bool:
    """The head's chunks with bf16 scales: lm_head_q8.h's default 8704-B law, which this tree's
    `lmhead_q8` pack op writes. The f32-scale 9216-B chunk is not implemented here (the kernel rounds
    its scales to bf16 RNE on the core anyway, so the logits are bit-identical)."""
    return True


def head_tile() -> int:
    return LM_TILE_BF16 if head_bf16() else LM_TILE
TAPS = 7                  # conv_L_cache; the conv core holds taps - 1 rows of state
SCW = 128                 # channels in one conv-core element (a block of TAPS rows is TAPS x SCW f32)
SLOT = 1536               # the dense MLP's slot width: 7680 = 5 x 1536 (a whole number of 64-row bands over 8 cores)
EMBED = "token_embd.weight"
MIX_CONV, MIX_ATT = "conv", "att"


def band_bytes(K: int) -> int:
    """64 rows x K: K/256 k-tiles x 2 thirty-two-row chunks of chunk() bytes."""
    return K // 128 * chunk()


@dataclass(frozen=True)
class Geometry:
    HID: int; FF: int; DFF: int
    NH: int; KVH: int; HD: int; ROT: int; QW: int; KVW: int
    N_CORES: int; PER_CALL: int; CALL_BYTES: int
    Q_PC: int; KV_PC: int; O_PC: int; SC_PC: int; DOWN_PC: int
    E_A: int; HPE: int; HPO: int; Q_AIN_ELEMS: int; K_AIN_ELEMS: int
    XN_ELEMS: int; XM_ELEMS: int; OG_ELEMS: int
    TAB_BYTES: int; MS_U: int; MS_G: int; MS_FLOATS: int
    EPS: float; SLOTS: int
    ACORES: int; NHL: int; VEXP: int; MLS: int; RB: int
    PTAB_ELEMS: int; TAPS: int; SCW: int; SC_ELEMS: int; WINDOW: int


@dataclass(frozen=True)
class Ffn:
    """The MLP's slot law: SLOTS kernel slots of width FF from a pool of as many slot-shaped regions at byte 0
    (gate | up per band, then the down slices)."""
    FF: int; UP_PC: int; SLOTS: int; POOL_SLOTS: int; H_ELEMS: int; H_STRIDE: int
    UP_BYTES: int; EXPERT_UG: int; POOL_DOWN: int


@dataclass(frozen=True)
class Layout:
    CD_LNW: int; CD_POSTOP: int; CD_FFNN: int; CD_POSTFFN: int; CD_META: int; CD_CONV: int; CD_BYTES: int
    AD_XN: int; AD_Q: int; AD_GATE: int; AD_KVN: int; AD_OG: int; AD_B: int; AD_C: int; AD_U: int; AD_Y: int
    AD_OUT: int; AD_T: int; AD_RES: int; AD_XM: int; AD_H: int; AD_OUT2: int; AD_T2: int; AD_BYTES: int
    POOL_MIX: int; POOL_B: int; POOL_C: int; POOL_U: int; POOL_OUT: int
    POOL_Q: int; POOL_GATE: int; POOL_K: int; POOL_V: int; POOL_O: int; POOL_BYTES: int
    STATE_BYTES: int
    KV_ROW: int; PTAB_ROW: int; MAX_CTX: int; KV_BYTES: int; PTAB_BYTES: int
    LM_ROWS: int; LMHEAD_BANDS: int; LMHEAD_BAND_BYTES: int; LMHEAD_POOL_BYTES: int
    ELN: int


@dataclass(frozen=True)
class Lfm3Recipe:
    spec: ModelSpec
    layout: Layout
    geo: Geometry
    max_ctx: int
    kind: str = "lfm3"


def _check(spec: ModelSpec) -> None:
    if spec.family != "lfm3":
        raise OpRangeError(f"lfm3 recipe given a {spec.family!r} spec")
    bad = sorted(set(spec.layer_types) - {SHORT_CONV, FULL, DENSE_LOCAL})
    if bad:
        raise OpRangeError(f"lfm3: layer types {bad} are not short_conv / full_attention / sliding")
    if spec.conv_kernel != TAPS:
        raise OpRangeError(f"lfm3: conv_kernel {spec.conv_kernel} is not {TAPS} (the conv core holds "
                           f"{TAPS - 1} rows of state)")
    if spec.activation != "silu" or not spec.sandwich_norms or not spec.attn_gate or not spec.qk_norm:
        raise OpRangeError("lfm3: expected SiLU, sandwich norms, the attention gate and q/k norms")
    if spec.num_experts or spec.shared_expert_intermediate or spec.rope_scaling:
        raise OpRangeError("lfm3: a dense model, default RoPE")
    if spec.rotary_dim != spec.head_dim or (DENSE_LOCAL in spec.layer_types and spec.sliding_window <= 0):
        raise OpRangeError("lfm3: a full-head RoPE table (the partial rotation is zero frequencies) and a window")
    if not spec.intermediate or spec.intermediate % SLOT:
        raise OpRangeError(f"lfm3: the MLP {spec.intermediate} is not whole {SLOT}-wide slots")
    if spec.num_kv_heads % 2 or spec.num_heads % spec.num_kv_heads or spec.num_heads % ATTN_CORES:
        raise OpRangeError(f"lfm3: {spec.num_heads} q / {spec.num_kv_heads} kv heads do not split "
                           f"over the attention elements and {ATTN_CORES} cores")
    qw, kvw = spec.num_heads * spec.head_dim, spec.num_kv_heads * spec.head_dim
    for what, w in (("hidden", spec.hidden), ("q width", qw), ("kv width", kvw), ("mlp slot", SLOT)):
        if w % (BAND_ROWS * N_CORES) or w % 256:
            raise OpRangeError(f"lfm3: {what} {w} is not whole 64-row bands over {N_CORES} cores and "
                               f"256-wide k-tiles")
    if spec.hidden % SCW:
        raise OpRangeError(f"lfm3: hidden {spec.hidden} is not whole {SCW}-channel conv elements")
    if 4 * spec.head_dim > 2 * kvw:
        raise OpRangeError("lfm3: qn | kn do not fit one meta element")


def geometry(spec: ModelSpec) -> Geometry:
    _check(spec)
    hid = spec.hidden
    nh, kvh, hd = spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    e_a = kvw * 2
    ptab_row = max(1024, e_a)
    nhl = nh // ATTN_CORES
    return Geometry(
        HID=hid, FF=SLOT, DFF=spec.intermediate,
        NH=nh, KVH=kvh, HD=hd, ROT=spec.rotary_dim, QW=qw, KVW=kvw,
        N_CORES=N_CORES, PER_CALL=PER_CALL, CALL_BYTES=PER_CALL * chunk(),
        Q_PC=qw // BAND_ROWS // N_CORES, KV_PC=kvw // BAND_ROWS // N_CORES, O_PC=hid // BAND_ROWS // N_CORES,
        SC_PC=hid // BAND_ROWS // N_CORES, DOWN_PC=hid // BAND_ROWS // N_CORES,
        E_A=e_a, HPE=e_a // (hd * 4), HPO=e_a // (hd * 2), Q_AIN_ELEMS=nh // (e_a // (hd * 4)),
        K_AIN_ELEMS=kvh // (e_a // (hd * 4)),
        XN_ELEMS=roundup(hid * 2, ELEM) // ELEM, XM_ELEMS=roundup(hid * 2, ELEM) // ELEM,
        OG_ELEMS=roundup(qw * 2, ELEM) // ELEM,
        TAB_BYTES=tab_bytes(max(hid, qw, SLOT)), MS_U=0, MS_G=BAND_ROWS, MS_FLOATS=3 * BAND_ROWS,   # u | g | a down band
        EPS=spec.norm_eps, SLOTS=spec.intermediate // SLOT,
        # the same attention point as HY V3 / LFM2-MoE (RB 4, the vector softmax, 4 cores) at head dim 128
        ACORES=ATTN_CORES, NHL=nhl, VEXP=1, MLS=roundup(nhl, 32), RB=4,
        PTAB_ELEMS=ptab_row // e_a, TAPS=TAPS, SCW=SCW, SC_ELEMS=hid // SCW, WINDOW=spec.sliding_window,
    )


def ffn(spec: ModelSpec) -> Ffn:
    G = geometry(spec)
    up = (G.FF // BAND_ROWS) * band_bytes(G.HID)         # one slot's gate (= up) projection
    down = (G.HID // BAND_ROWS) * band_bytes(G.FF)
    if up != down:
        raise OpRangeError("lfm3: a slot's gate and down must be the same size")
    return Ffn(FF=G.FF, UP_PC=G.FF // BAND_ROWS // N_CORES, SLOTS=G.SLOTS, POOL_SLOTS=G.SLOTS,
               H_ELEMS=roundup(G.FF * 4, ELEM) // ELEM, H_STRIDE=roundup(G.FF * 4, ELEM),
               UP_BYTES=up, EXPERT_UG=2 * up, POOL_DOWN=G.SLOTS * 2 * up)


def lm_rows(spec: ModelSpec) -> int:
    return roundup(spec.vocab, LM_BAND_ROWS)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    G = geometry(spec)
    F = ffn(spec)
    eln = G.HID * 2
    cd = {"lnw": 0, "postop": eln, "ffnn": 2 * eln, "postffn": 3 * eln, "meta": 4 * eln,
          "conv": roundup(4 * eln + G.E_A, ELEM)}
    cd_bytes = roundup(cd["conv"] + G.TAPS * G.HID * 4, ELEM)
    a: dict[str, int] = {}
    off = 0
    for name, size in (("xn", G.XN_ELEMS * ELEM), ("q", G.QW * 4), ("gate", G.QW * 4), ("kvn", 2 * G.KVW * 4),
                       ("og", G.OG_ELEMS * ELEM), ("b", G.HID * 4), ("c", G.HID * 4), ("u", G.HID * 4),
                       ("y", G.XN_ELEMS * ELEM), ("out", G.HID * 4), ("t", G.HID * 4), ("res", G.HID * 4),
                       ("xm", G.XM_ELEMS * ELEM), ("h", F.SLOTS * F.H_STRIDE), ("out2", G.HID * 4),
                       ("t2", G.HID * 4)):
        a[name] = off
        off += roundup(size, ELEM)
    ad_bytes = roundup(off, ELEM)
    # the mixer after the MLP's slots, at the same offset in every layer's pool: one conv build and one
    # attention build serve every layer
    mix = F.POOL_DOWN + F.POOL_SLOTS * F.UP_BYTES
    sq = (G.HID // BAND_ROWS) * band_bytes(G.HID)       # one [hidden, hidden] projection
    qb = (G.QW // BAND_ROWS) * band_bytes(G.HID)
    kb = (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    p = {"b": mix, "c": mix + sq, "u": mix + 2 * sq, "out": mix + 3 * sq, "q": mix}
    p["gate"] = p["q"] + qb
    p["k"] = p["gate"] + qb
    p["v"] = p["k"] + kb
    p["o"] = p["v"] + kb
    end = max(mix + 4 * sq, p["o"] + (G.HID // BAND_ROWS) * band_bytes(G.QW))
    pool_bytes = roundup(end, MB)
    if pool_bytes >= 1 << 32:
        raise OpRangeError("lfm3: a layer pool past 4 GiB does not fit a 32-bit fill offset")
    rows = lm_rows(spec)
    bands = rows // LM_BAND_ROWS
    lm_band = 4 * (G.HID // 256) * head_tile()
    kv_row = 2 * G.E_A
    ptab_row = G.PTAB_ELEMS * G.E_A
    return Layout(
        CD_LNW=cd["lnw"], CD_POSTOP=cd["postop"], CD_FFNN=cd["ffnn"], CD_POSTFFN=cd["postffn"],
        CD_META=cd["meta"], CD_CONV=cd["conv"], CD_BYTES=cd_bytes,
        AD_XN=a["xn"], AD_Q=a["q"], AD_GATE=a["gate"], AD_KVN=a["kvn"], AD_OG=a["og"], AD_B=a["b"], AD_C=a["c"],
        AD_U=a["u"], AD_Y=a["y"], AD_OUT=a["out"], AD_T=a["t"], AD_RES=a["res"], AD_XM=a["xm"], AD_H=a["h"],
        AD_OUT2=a["out2"], AD_T2=a["t2"], AD_BYTES=ad_bytes,
        POOL_MIX=mix, POOL_B=p["b"], POOL_C=p["c"], POOL_U=p["u"], POOL_OUT=p["out"],
        POOL_Q=p["q"], POOL_GATE=p["gate"], POOL_K=p["k"], POOL_V=p["v"], POOL_O=p["o"], POOL_BYTES=pool_bytes,
        STATE_BYTES=(G.TAPS - 1) * G.HID * 4,
        KV_ROW=kv_row, PTAB_ROW=ptab_row, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * ptab_row,
        LM_ROWS=rows, LMHEAD_BANDS=bands, LMHEAD_BAND_BYTES=lm_band,
        LMHEAD_POOL_BYTES=roundup(roundup(bands, LM_CORES) * lm_band, MB),
        ELN=eln,
    )


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> Lfm3Recipe:
    return Lfm3Recipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def mixer_of(layer_type: str) -> str:
    return MIX_CONV if layer_type == SHORT_CONV else MIX_ATT


# The attention gate (designs/attn/probe_dense.py) lifts this family's SHIPPED
# worker out of the design it builds: LFM3's `attn_core` (designs/lfm3/lg.py),
# which is the dense `_attn` plus the per-element output gate stream. The probe
# binds the body by name, so a family only has to point at its own source.
ATTN_WORKER = ("designs/lfm3/lg.py", "attn_core")


def attn_flags(G: Geometry, probe_env=()) -> list[str]:
    """attn.h for LFM3: the q / k norm before the rotation, the per-element output gate, no sink, the vector
    softmax over 4 cores of 4 heads, RB 4. ROT is the whole head: the full layers' half rotation is a table
    whose second half of the pairs has frequency 0."""
    flags = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=1",
             "-DATTN_QKNORM=1", "-DATTN_QKNORM_POST=0", f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}",
             f"-DATTN_NHL={G.NHL}", "-DATTN_INTFP=1"]   # INTFP: attn.h's integer scalar routines (bit-identical;
    # the gate's sigmoid and the block kernel share one 16 KB program memory)
    if G.RB > 1:
        flags.append(f"-DATTN_RB={G.RB}")
    if G.PTAB_ELEMS > 1:
        flags.append("-DATTN_PTAB_SPLIT=1")
    for k, v in probe_env:
        flags.append(f"-D{k}={v}")
    return flags


def ln_flags(G: Geometry) -> list[str]:
    return [f"-DLN_N={G.HID}", f"-DLN_EPS={G.EPS:g}f"]


PROBE_VARS = ("ATTN_NULL", "LG_NULL")


def probe_env() -> dict[str, str]:
    return {k: os.environ[k] for k in PROBE_VARS if os.environ.get(k)}


def gguf_quant(spec: ModelSpec) -> str:
    """The law the GGUF view's pools hold: q4_1 (the QAD Q4_0's bf16 scales exactly: m = -8 d)."""
    return "q4_1"


def finalize_gguf_manifest(spec: ModelSpec, manifest: dict) -> None:
    """The manifest IS the GGUF view (there is no q4nx container): core.cpp reads a model.gguf through the
    nested `gguf` section, so the section is the same manifest, nested once."""
    manifest["layers"] = list(spec.layer_types)
    if "gguf" not in manifest:
        manifest["gguf"] = copy.deepcopy(manifest)


def _q4_op(tensor: str, dst: int, rows: int, in_dim: int, row0: int = 0, col0: int = 0, experts: int = 1,
           row_step: int = 0, col_step: int = 0, expert_bytes: int = 0, band_stride: int = 0) -> dict:
    """runtime pack op fl_q4 (flint-runtime.patch): for e < experts, the rows x in_dim block at
    (row0 + e row_step, col0 + e col_step) of a Q4_0 / Q4_1 GGUF tensor as 64-row bands of q4_1 chunks
    (std_perm_gguf's byte law, scale and min bf16 RNE), band b at dst + e expert_bytes + b band_stride."""
    if rows % BAND_ROWS or in_dim % 256 or col0 % 32 or col_step % 32:
        raise OpRangeError(f"lfm3: {tensor}: {rows} x {in_dim} at column {col0} is not whole bands and k-tiles")
    return {"op": "fl_q4", "tensor": tensor, "dst": dst, "rows": rows, "in_dim": in_dim, "row0": row0,
            "chunk0": col0, "experts": experts, "n_total": row_step, "cols": col_step,
            "expert_bytes": expert_bytes, "stripe_bytes": band_stride}


def _layer_plan(spec: ModelSpec, mixer: str) -> dict:
    G, L, F = geometry(spec), layout(spec), ffn(spec)
    b = "blk.{l}."
    hid, bbh = G.HID, band_bytes(G.HID)
    ug = dict(expert_bytes=F.EXPERT_UG, band_stride=2 * bbh)
    n = F.POOL_SLOTS
    ffn_ops = [   # the MLP's rows (gate, up) and columns (down) in SLOT-wide slots
        _q4_op(b + "ffn_gate.weight", 0, F.FF, hid, experts=n, row_step=F.FF, **ug),
        _q4_op(b + "ffn_up.weight", bbh, F.FF, hid, experts=n, row_step=F.FF, **ug),
        _q4_op(b + "ffn_down.weight", F.POOL_DOWN, hid, F.FF, experts=n, col_step=F.FF, expert_bytes=F.UP_BYTES),
    ]
    consts = [
        {"op": "put", "tensor": b + "attn_norm.weight", "dst": L.CD_LNW, "cap": L.ELN},
        {"op": "put", "tensor": b + "post_attention_norm.weight", "dst": L.CD_POSTOP, "cap": L.ELN},
        {"op": "put", "tensor": b + "ffn_norm.weight", "dst": L.CD_FFNN, "cap": L.ELN},
        {"op": "put", "tensor": b + "post_ffw_norm.weight", "dst": L.CD_POSTFFN, "cap": L.ELN},
    ]
    if mixer == MIX_CONV:
        ip = b + "shortconv.in_proj.weight"      # [3 hidden, hidden]: B, C, u down the rows
        mix = [_q4_op(ip, L.POOL_B, hid, hid, row0=0), _q4_op(ip, L.POOL_C, hid, hid, row0=hid),
               _q4_op(ip, L.POOL_U, hid, hid, row0=2 * hid),
               _q4_op(b + "shortconv.out_proj.weight", L.POOL_OUT, hid, hid)]
        # The GGUF stores the taps F32 [hidden, taps]; the core wants them TAP-major (consecutive channels of
        # one tap contiguous), kept f32: [hidden, taps] -> [taps, hidden], 4-byte elements.
        consts.append({"op": "transpose", "tensor": b + "shortconv.conv.weight", "dst": L.CD_CONV,
                       "rows": hid, "cols": G.TAPS, "elem": 4})
    else:
        mix = [_q4_op(b + "attn_q.weight", L.POOL_Q, G.QW, hid),
               _q4_op(b + "attn_gate.weight", L.POOL_GATE, G.QW, hid),     # the derived GGUF's expanded gate
               _q4_op(b + "attn_k.weight", L.POOL_K, G.KVW, hid), _q4_op(b + "attn_v.weight", L.POOL_V, G.KVW, hid),
               _q4_op(b + "attn_output.weight", L.POOL_O, hid, G.QW)]
        consts += [
            {"op": "put", "tensor": b + "attn_q_norm.weight", "dst": L.CD_META, "cap": 2 * G.HD},
            {"op": "put", "tensor": b + "attn_k_norm.weight", "dst": L.CD_META + 2 * G.HD, "cap": 2 * G.HD},
        ]
    return {"pool": ffn_ops + mix, "consts": consts}


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    out = {}
    for lt in spec.layer_types:
        out.setdefault(lt, _layer_plan(spec, mixer_of(lt)))
    return {
        "pool_bytes": L.POOL_BYTES, "chunk_bytes": chunk(),
        "layer_types": out,
        # the head is tied: the (derived) Q8_0 token table, through lm_head_q8's f32-scale chunks
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "lmhead_q8", "tensor": EMBED, "dst": 0, "chunk_bytes": head_tile(),
                             "rows": L.LM_ROWS, "in_dim": G.HID}]},
        "embed": {"tensor": EMBED, "dim": G.HID},
        "norm": {"tensor": "output_norm.weight", "bytes": L.ELN},
    }


CONV_ARGS = ["pool", "xres", "consts", "state", "act"]
ATT_ARGS = ["pool", "xres", "consts", "state", "act"]      # + the position table: ptab / ptab_local
MLP_ARGS = ["pool", "xres", "consts", "act"]


def rope_table(spec: ModelSpec, local: bool) -> list[float]:
    """The inverse frequency of each rotary pair i < head_dim / 2 (HF Lfm3RotaryEmbedding, zero-tail): the
    first int(partial * hd) // 2 pairs at theta^(-2i/hd), the rest 0 (cos 1, sin 0: unrotated)."""
    hd = spec.head_dim
    partial = 1.0 if local else float(spec.extra.get("partial_rotary_factor", 1.0))
    theta = spec.rope_local_theta if local else spec.rope_theta
    n = int(partial * hd) // 2
    return [theta ** (-2.0 * i / hd) for i in range(n)] + [0.0] * (hd // 2 - n)


# ---- block prefill (the LFM2.5-24B-A2B route, recipes/lfm2moe.py gemm_route, kind `lfm`; .plans/lfm3-native.md "block prefill")
def block_on() -> bool:
    """The block prefill route's kernels and manifest keys.

    The engine refuses `kind:"lfm"` block prefill for a GGUF-direct model (core.cpp's
    `enable_gguf_block` accepts only dense kinds) and falls back to step prefill, so
    shipping the route to a GGUF set builds kernels nothing runs. LFM3 is GGUF-direct
    only; the block route stays in the tree behind LFM3_BLOCK=1 for a future engine
    that implements the hybrid kind. Block prefill: not implemented here.
    """
    return os.environ.get("LFM3_BLOCK") == "1"


def gemm_route(spec: ModelSpec, max_ctx: int = 4096) -> dict | None:
    """Per layer type the `gemm_block` (kind `lfm`) the driver reads, plus the contexts / kernels / globals / builds the
    route adds. None under LFM3_BLOCK=0. A layer is the mixer's fused input GEMM (conv in_proj B | C | u; attention
    q | gate | k | v with the derived GGUF's expanded gate), the host's middle (the 7-tap conv with its 6-row state, or
    attention over the block with the layer type's RoPE table and window, the sigmoid gate on the output), the out GEMM, the
    sandwich norm and residual, then the dense MLP and its sandwich norm. The MLP reads the decode pool in place: slot e's
    gate | up bands (alternating 64-row bands, EXPERT_UG bytes at e * EXPERT_UG) are one GEMM weight of 2 x 1536 rows and its
    down bands (UP_BYTES at POOL_DOWN + e * UP_BYTES) another of K = 1536; the host sums the slots' down outputs."""
    if not block_on():
        return None
    from .qwen36moe import GEMM_T
    L, G, F, T = layout(spec, max_ctx), geometry(spec), ffn(spec), GEMM_T
    hid, qw, kvw = G.HID, G.QW, G.KVW
    shapes: set[tuple[int, int]] = set()

    def ctx(N: int, K: int) -> str:
        if N % 256 or K % 256:
            raise OpRangeError(f"lfm3 gemm route: [{N}, {K}] is not a multiple of 256 in both dims")
        shapes.add((N, K))
        return f"gemm_n{N}_k{K}"

    def run(N: int, K: int, w: str) -> dict:
        return {"op": "run", "kernel": ctx(N, K), "args": [w, f"gemm_x_k{K}", f"gemm_y_n{N}"]}

    sq = (hid // BAND_ROWS) * band_bytes(hid)
    qb = (qw // BAND_ROWS) * band_bytes(hid)
    kb = (kvw // BAND_ROWS) * band_bytes(hid)
    out: dict = {"layer_types": {}, "contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    ffn_w, ffn_prog = {}, []
    for e in range(F.SLOTS):
        ffn_w[f"gug{e}_w"] = {"from": "pool", "range": [e * F.EXPERT_UG, F.EXPERT_UG]}
        ffn_w[f"gdn{e}_w"] = {"from": "pool", "range": [F.POOL_DOWN + e * F.UP_BYTES, F.UP_BYTES]}
        ffn_prog += [run(2 * F.FF, hid, f"gug{e}_w"), run(hid, F.FF, f"gdn{e}_w")]
    for lt in sorted(set(spec.layer_types)):
        gb: dict = {"kind": "lfm", "mixer": mixer_of(lt), "t": T, "eps": spec.norm_eps, "sandwich": True,
                    "cd_ln": L.CD_LNW, "cd_ffn": L.CD_FFNN, "cd_postop": L.CD_POSTOP, "cd_postffn": L.CD_POSTFFN,
                    "ff": F.FF, "ffn_slots": F.SLOTS, "ffn_program": ffn_prog, "ffn_weights": ffn_w}
        if lt == SHORT_CONV:
            gb.update(program=[run(3 * hid, hid, "gin_w"), run(hid, hid, "gout_w")],
                      weights={"gin_w": {"from": "pool", "range": [L.POOL_B, 3 * sq]},
                               "gout_w": {"from": "pool", "range": [L.POOL_OUT, sq]}},
                      taps=G.TAPS, cd_conv=L.CD_CONV)
        else:
            local = lt == DENSE_LOCAL
            gb.update(program=[run(2 * qw + 2 * kvw, hid, "gqkv_w"), run(hid, qw, "go_w")],
                      weights={"gqkv_w": {"from": "pool", "range": [L.POOL_Q, 2 * qb + 2 * kb]},
                               "go_w": {"from": "pool", "range": [L.POOL_O, (hid // BAND_ROWS) * band_bytes(qw)]}},
                      gate=True, qw=qw, kvw=kvw, nh=G.NH, kvh=G.KVH, hd=G.HD, rot=spec.rotary_dim,
                      cd_qnorm=L.CD_META, cd_knorm=L.CD_META + 2 * G.HD,
                      window=spec.sliding_window if local else 0, rope_inv_freq=rope_table(spec, local))
        out["layer_types"][lt] = gb
    for N, K in sorted(shapes):
        name = f"gemm_n{N}_k{K}"
        out["contexts"].setdefault("gemm", f"{name}/final.xclbin")
        out["kernels"][name] = {"context": "gemm", "insts": f"{name}/insts.bin", "build": name}
        out["globals"][f"gemm_x_k{K}"] = K * T * 2
        out["globals"][f"gemm_y_n{N}"] = max(out["globals"].get(f"gemm_y_n{N}", 0), N * T * 4)
        out["builds"][name] = {"design": "gemm_q4_prefill/gemm_q4_prefill.py",
                               "build_dir": f"gemm_q4_prefill/build_n{N}_k{K}_t{T}",
                               "env": {"GQP_N": str(N), "GQP_K": str(K), "GQP_T": str(T)}}
    return out


def fused_on() -> bool:
    """One dispatch a layer (designs/lfm3/lg.py LG_PART=convmlp | attmlp: the mixer's program then the MLP's over the
    same fifos): on by default (LFM3_FUSED=0 for the mixer and the MLP as two dispatches, which also turns the shared
    image off)."""
    return os.environ.get("LFM3_FUSED") != "0"


def shared_on() -> bool:
    """The fused layers as ONE image for both mixers (LG_PART=shared, two instruction streams LG_STREAM=conv | att whose
    xclbins agree apart from build stamps; only the conv build's is loaded): conv and attention layers never change
    hardware context. On by default with fusion (LFM3_SHARED=0 for a fused image a mixer)."""
    return os.environ.get("LFM3_SHARED") != "0" and fused_on()


SHARED_BUILDS = {"conv": "lg_shared_conv", "att": "lg_shared_att"}


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    out = _programs(spec, max_ctx)
    if not fused_on():
        return out
    # fold the MLP into the mixer's dispatch: the kernels / contexts are renamed, the layer program is one step
    ren = {"lg_conv": "lg_convmlp", "lg_att": "lg_attmlp", "lg_att_local": "lg_attmlp_local"}
    ctx_ren = {"lg_conv": "lg_convmlp", "lg_att": "lg_attmlp"}
    out["contexts"] = {ctx_ren.get(k, k): v.replace(f"{k}/", f"{ctx_ren.get(k, k)}/") if k in ctx_ren else v
                       for k, v in out["contexts"].items() if k != "lg_mlp"}
    ks = {}
    for k, v in out["kernels"].items():
        if k == "lg_mlp":
            continue
        if k in ren:
            v = dict(v)
            v["context"] = ctx_ren.get(v["context"], v["context"])
            v["insts"] = v["insts"].replace(v["build"] + "/", ctx_ren[v["build"]] + "/")
            v["build"] = ctx_ren[v["build"]]
        ks[ren.get(k, k)] = v
    out["kernels"] = ks
    for lt, e in out["layer_types"].items():
        step = dict(e["program"][0])
        step["kernel"] = ren[step["kernel"]]
        e["program"] = [step]
    if not shared_on():
        return out
    # one image: the conv stream's xclbin is the context of every layer; both streams take the attention's six
    # buffer arguments (a conv layer passes its state BO in the kv slot and the full layers' ptab, which it never reads)
    for name in ("lg_convmlp", "lg_attmlp"):
        out["contexts"].pop(name, None)
    out["contexts"]["lg_shared"] = f"{SHARED_BUILDS['conv']}/final.xclbin"
    for k, v in out["kernels"].items():
        if v.get("build") in ("lg_convmlp", "lg_attmlp"):
            b = SHARED_BUILDS["conv" if v["build"] == "lg_convmlp" else "att"]
            v.update(context="lg_shared", insts=f"{b}/insts.bin", build=b)
    if SHORT_CONV in out["layer_types"]:
        if "ptab" not in out["globals"]:
            raise OpRangeError("lfm3 shared image: a conv layer needs a ptab global for its sixth argument")
        out["layer_types"][SHORT_CONV]["program"][0]["args"] = CONV_ARGS + ["ptab"]
    return out


def _programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    check_buffer_args("lg_conv", CONV_ARGS)
    check_buffer_args("lg_att", ATT_ARGS + ["ptab"])
    check_buffer_args("lg_mlp", MLP_ARGS)
    out = {
        "contexts": {"lg_mlp": "lg_mlp/final.xclbin", "ln": "ln/final.xclbin", "lm": "lm_head_q8/final.xclbin"},
        "kernels": {
            "lg_mlp": {"context": "lg_mlp", "insts": "lg_mlp/insts.bin", "build": "lg_mlp"},
            "ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
            "lm": {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"},
        },
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": G.HID * 4, "zero": G.HID * 4, "normw": G.HID * 2, "xresf": G.HID * 4, "hn": G.HID * 2,
                    "logits": L.LM_ROWS * 4, "lmpool": L.LMHEAD_POOL_BYTES},
    }
    if SHORT_CONV in spec.layer_types:
        out["contexts"]["lg_conv"] = "lg_conv/final.xclbin"
        out["kernels"]["lg_conv"] = {"context": "lg_conv", "insts": "lg_conv/insts.bin", "build": "lg_conv"}
    if FULL in spec.layer_types or DENSE_LOCAL in spec.layer_types:
        out["contexts"]["lg_att"] = "lg_att/final.xclbin"
    mlp = {"op": "run", "kernel": "lg_mlp", "args": MLP_ARGS}
    for lt in sorted(set(spec.layer_types)):
        if lt == SHORT_CONV:
            first = {"op": "run", "kernel": "lg_conv", "args": CONV_ARGS}
            state = {"kind": "linear", "bytes": L.STATE_BYTES}
        else:
            local = lt == DENSE_LOCAL
            kn, tab = ("lg_att_local", "ptab_local") if local else ("lg_att", "ptab")
            window = spec.sliding_window if local else 0
            out["kernels"][kn] = {"context": "lg_att", "insts": "lg_att/insts.bin", "patch": "attnpos",
                                  "build": "lg_att", "window": window}
            out["globals"][tab] = {"per_row": L.PTAB_ROW, "inv_freq": rope_table(spec, local), "window": window}
            first = {"op": "run", "kernel": kn, "args": ATT_ARGS + [tab]}
            state = {"kind": "kv", "row": L.KV_ROW}
        out["layer_types"][lt] = {"buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": state},
                                  "program": [first, mlp]}
    gr = gemm_route(spec, max_ctx)
    if gr:
        for lt, gb in gr["layer_types"].items():
            out["layer_types"][lt]["gemm_block"] = gb
        for k in ("contexts", "kernels"):
            out[k].update(gr[k])
        for k, v in gr["globals"].items():
            out["globals"][k] = max(out["globals"].get(k, 0), v)
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    out = _fused_builds(spec)
    if head_bf16():
        b = out["lm_head_q8"]
        b["build_dir"] = b["build_dir"].replace("build_gguf_", "build_gguf_bf16s_")
        b["env"] = {k: v for k, v in b["env"].items() if k != "LMHEAD_SCALES_F32"}
    if q40_on():
        # the Q4_0-native chunk law: its own build directories, and the block route's GEMMs read the pool as q4_0
        for name, b in out.items():
            if b["design"] == "lfm3/lg.py":
                b["build_dir"] += "_q40"
                b["env"] = {**b["env"], "LFM3_Q40": "1"}
            elif b["design"] == "gemm_q4_prefill/gemm_q4_prefill.py":
                b["build_dir"] += "_q40"
                b["env"] = {**b["env"], "GQP_FMT": "q4_0"}
    return out


def _fused_builds(spec: ModelSpec) -> dict[str, dict]:
    out = _builds(spec)
    if not fused_on():
        return out
    G = geometry(spec)
    out.pop("lg_mlp")
    if "lg_conv" in out:
        out.pop("lg_conv")
        out["lg_convmlp"] = {"design": "lfm3/lg.py", "build_dir": f"lfm3/build_convmlp_h{G.HID}_t{G.TAPS}",
                             "env": {"LG_PART": "convmlp"}}
    if "lg_att" in out:
        out.pop("lg_att")
        out["lg_attmlp"] = {"design": "lfm3/lg.py", "build_dir": f"lfm3/build_attmlp_h{G.HID}_d{G.HD}",
                            "env": {"LG_PART": "attmlp"}}
    if shared_on():
        out.pop("lg_convmlp", None)
        out.pop("lg_attmlp", None)
        for stream, name in SHARED_BUILDS.items():
            out[name] = {"design": "lfm3/lg.py", "build_dir": f"lfm3/build_shared_h{G.HID}_d{G.HD}_{stream}",
                         "env": {"LG_PART": "shared", "LG_STREAM": stream}}
    return out


def _builds(spec: ModelSpec) -> dict[str, dict]:
    G, L = geometry(spec), layout(spec)
    out = {
        "lg_mlp": {"design": "lfm3/lg.py", "build_dir": f"lfm3/build_mlp_h{G.HID}_f{G.FF}_s{G.SLOTS}",
                   "env": {"LG_PART": "mlp"}},
        "ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{G.HID}_{G.EPS:g}",
               "env": {"LN_N": str(G.HID), "LN_EPS": f"{G.EPS:g}"}},
        "lm_head_q8": {"design": "lm_head_q8/lm_head_q8.py",
                       "build_dir": f"lm_head_q8/build_gguf_n{L.LM_ROWS}_k{G.HID}_c{LM_CORES}",
                       "env": {"LMHEAD_N": str(L.LM_ROWS), "LMHEAD_K": str(G.HID), "LMHEAD_CORES": str(LM_CORES),
                               "LMHEAD_SCALES_F32": "1"}},
    }
    if SHORT_CONV in spec.layer_types:
        out["lg_conv"] = {"design": "lfm3/lg.py", "build_dir": f"lfm3/build_conv_h{G.HID}_t{G.TAPS}",
                          "env": {"LG_PART": "conv"}}
    if FULL in spec.layer_types or DENSE_LOCAL in spec.layer_types:
        out["lg_att"] = {"design": "lfm3/lg.py", "build_dir": f"lfm3/build_att_h{G.HID}_d{G.HD}",
                         "env": {"LG_PART": "att"}}
    gr = gemm_route(spec)
    if gr:
        out.update(gr["builds"])
    return out


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    return {"hidden": G.HID, "vocab": L.LM_ROWS, "real_vocab": spec.real_vocab, "chunk_bytes": chunk(),
            "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": head_tile(),
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": rope_table(spec, False)}


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "intermediate_size": spec.intermediate, "conv_L_cache": spec.conv_kernel,
            "sliding_window": spec.sliding_window, "tie_word_embeddings": True, "use_sandwich_norm": True}


GEN_KERNELS = "designs/lfm3/gen_kernels.py"
KERNEL_SOURCES = [
    "designs/lfm3/*.py", "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h",
    "designs/attn/*.cc", "designs/attn/*.h", "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py",
    "designs/lin_layer/ln_nr.cc", "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.h", "designs/lm_head_q8/*.cc",
    "include/vecmath.h", "include/scalar_fp.h", "ironutil.py", "build_design.py",
]
