"""LiquidAI LFM2-MoE (OPEN-FAMILY-LFM2MOE; LFM2.5-24B-A2B) on the open engine, from the GGUF.

LFM2's hybrid -- most layers a short depthwise causal conv, every fourth GQA attention -- with the FFN of
every layer past the first two replaced by routed experts. A layer is two dispatches with the router on the
host between them:

  mixer   one of two programs (designs/lfm2moe/lm.py) over ONE pool offset, POOL_MIX:
    lm_conv  (LM_PART=conv; the conv layers): ln -> B | C | u (in_proj's three row slices) -> the conv core
             (bx = B u; conv = s0 w0 + s1 w1 + bx w2, the LAST tap on this token; y = C conv; the state
             slides) -> out -> res = x + out; xm = ffn_norm(res). State: the `linear` kind, 2 x hidden f32.
    lm_att   (LM_PART=att; attnpos; the attention layers): ln -> q | k | v -> per-head RMSNorm q, k ->
             NEOX RoPE -> causal attention (no sink, no window) -> o -> res = x + o; xm = ffn_norm(res)
  hy_route (host; hyv3-runtime.patch, reused unchanged -- it IS this router): sigmoid scores of the F32
           router [64, 2048] over xm; the top 4 of score + exp_probs_b (the selection bias); their
           UNBIASED scores renormalised times routed_scaling_factor; no shared expert. The weights go into
           act beside xm, the four experts into lm_moe's fills (moeroute2's patch).
  lm_moe   (LM_PART=moe; moeroute2): per slot, gate | up -> w_e * silu(gate) * up -> h_e; down sums the
           four slots in each core's accumulator; xres = res + down
The two leading dense layers (conv layers in LFM2.5-24B-A2B) run the same lm_conv build and `lm_dense`: the
11776-wide MLP as 23 slots of 512 through the same expert program, unweighted, no route. Then the final norm
(`ln`, token_embd_norm) and the q8 head (`lm_head_q8`, f32 scales over the tied Q8_0 token_embd).

Weights. The GGUF is our own pure Q4_0 from LiquidAI's BF16 GGUF (llama quantize --pure, the tied embedding
Q8_0; .plans/lfm2moe-native.md): from a BF16 source every Q4_0 scale is a block max / -8, a bf16 value, so
the q4_1 pool law (5120-B chunks, d and m = -8 d as bf16; fl_q4) holds every matrix EXACTLY. The embedding
stays on the host (embed_row, Q8_0 exactly); norms and q / k norms go to bf16 in the consts, the conv taps
stay f32 (tap-major, the `transpose` op over the GGUF's F32 [hidden, 3]).

Widths: hidden 2048 and the expert width 1536 are whole 64-row bands over 8 cores (1536 = 24 bands = 3 a
core) and whole 256-wide k-tiles, so a slot's gate and its down are the same size (moeroute2's law) with no
padding and no pair law (cf. HY V3's 768).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import BAND_ROWS, ELEM, GEMM_T, MB, roundup, tab_bytes
from .spec import FULL, SHORT_CONV, ModelSpec

CH = 5120                 # a q4_1 chunk: 32 rows x 256 K, bf16 d | m planes (gemv_q4.h)
PER_CALL = 2              # chunks per w element (10240 B)
N_CORES = 8
WV = 16                   # floats of the routing-weight record beside xm (hy_route writes f32[16])
ATTN_CORES = 4            # attnknobs.attn_cores(32): the largest divisor of 32 heads <= 6
STACK = 0x1800
LM_CORES = 8
LM_BAND_ROWS = 128        # lm_head_q8: 128-row bands
LM_TILE = 9216            # lm_head_q8.h's f32-scale chunk: 32 rows x 256 K
TAPS = 3                  # conv_L_cache; the conv core holds taps - 1 rows of state
SCW = 512                 # channels in one conv-core element (cx.py's width)
DENSE_SLOT = 512          # the dense MLP's slot width: 11776 = 23 x 512 (not whole 1536s)
EMBED = "token_embd.weight"
# the layer's mixer x its FFN form
MIX_CONV, MIX_ATT = "conv", "att"


def band_bytes(K: int) -> int:
    """64 rows x K: K/256 k-tiles x 2 thirty-two-row chunks of CH bytes."""
    return K // 128 * CH


@dataclass(frozen=True)
class Geometry:
    HID: int; FF: int; DFF: int
    NH: int; KVH: int; HD: int; ROT: int; QW: int; KVW: int
    N_CORES: int; PER_CALL: int; CALL_BYTES: int
    Q_PC: int; KV_PC: int; O_PC: int; SC_PC: int; DOWN_PC: int
    E_A: int; HPE: int; HPO: int; Q_AIN_ELEMS: int; K_AIN_ELEMS: int
    XN_ELEMS: int; XM_ELEMS: int; OG_ELEMS: int
    TAB_BYTES: int; MS_U: int; MS_G: int; MS_FLOATS: int
    EPS: float; TOPK: int; EXPERTS: int; SLOTS: int; DENSE_SLOTS: int
    ACORES: int; NHL: int; VEXP: int; MLS: int; RB: int
    PTAB_ELEMS: int; W_OFF: int; TAPS: int; SCW: int; SC_ELEMS: int


@dataclass(frozen=True)
class Ffn:
    """One FFN form's slot law: `slots` kernel slots of width `ff` from a pool of `pool_slots`
    expert-shaped regions at byte 0 (gate | up per band, then the down slices)."""
    FF: int; UP_PC: int; SLOTS: int; POOL_SLOTS: int; H_ELEMS: int; H_STRIDE: int
    UP_BYTES: int; EXPERT_UG: int; POOL_DOWN: int; WEIGHTED: bool


@dataclass(frozen=True)
class Layout:
    CD_LNW: int; CD_POSTLN: int; CD_META: int; CD_CONV: int; CD_BYTES: int
    AD_XN: int; AD_Q: int; AD_KVN: int; AD_OG: int; AD_B: int; AD_C: int; AD_U: int; AD_Y: int
    AD_OUT: int; AD_RES: int; AD_XM: int; AD_W: int; AD_H: int; AD_OUT2: int; AD_JUNK: int; AD_BYTES: int
    POOL_MIX: int; POOL_B: int; POOL_C: int; POOL_U: int; POOL_OUT: int
    POOL_Q: int; POOL_K: int; POOL_V: int; POOL_O: int; POOL_BYTES: int
    STATE_BYTES: int
    KV_ROW: int; PTAB_ROW: int; MAX_CTX: int; KV_BYTES: int; PTAB_BYTES: int
    LM_ROWS: int; LMHEAD_BANDS: int; LMHEAD_BAND_BYTES: int; LMHEAD_POOL_BYTES: int
    ELN: int


@dataclass(frozen=True)
class Lfm2MoeRecipe:
    spec: ModelSpec
    layout: Layout
    geo: Geometry
    max_ctx: int
    kind: str = "lfm2moe"


def router_scale(spec: ModelSpec) -> float:
    """routed_scaling_factor (host arithmetic: spec.extra, outside the kernel hash)."""
    return float(spec.extra.get("router_scale", 1.0))


def padded_ff(real: int) -> int:
    """The compiled expert width: the real MoE intermediate rounded up to whole
    64-row bands over the cores (and whole 256-wide k-tiles, which 64*8 already is).

    An expert width need not be a whole number of bands -- LFM2.5-8B-A1B's is
    1792 = 3.5 bands. The pool and the kernel still need a whole number, so the
    compiled width is roundup(1792, 512) = 2048: four bands. The pack op reads
    ONLY the real 1792 rows/columns from the GGUF (via `rows`/`in_dim`) and writes
    them into the padded 2048-row slot; the remaining bytes stay zero (pools.cpp
    zeroes every pool before apply), so the padded lanes contribute nothing to
    the GEMV. Nothing here is tied to a particular size: any width with whole
    k-tiles can be padded this way.
    """
    return roundup(real, BAND_ROWS * N_CORES)


def _check(spec: ModelSpec) -> None:
    if spec.family != "lfm2moe":
        raise OpRangeError(f"lfm2moe recipe given a {spec.family!r} spec")
    bad = sorted(set(spec.layer_types) - {SHORT_CONV, FULL})
    if bad:
        raise OpRangeError(f"lfm2moe: layer types {bad} are neither short_conv nor full_attention")
    if spec.conv_kernel != TAPS:
        raise OpRangeError(f"lfm2moe: conv_kernel {spec.conv_kernel} is not {TAPS} (the conv core holds "
                           f"{TAPS - 1} rows of state)")
    if spec.activation != "silu" or not spec.num_experts or not spec.moe_intermediate:
        raise OpRangeError("lfm2moe: expected routed SiLU experts")
    if spec.shared_expert_intermediate:
        raise OpRangeError("lfm2moe: a shared expert is not part of this family")
    if spec.experts_per_tok > WV:
        raise OpRangeError(f"lfm2moe: top-{spec.experts_per_tok} exceeds the {WV}-entry weight record")
    if spec.dense_lead and (not spec.intermediate or spec.intermediate % DENSE_SLOT):
        raise OpRangeError(f"lfm2moe: the dense MLP {spec.intermediate} is not whole {DENSE_SLOT}-wide slots")
    if not spec.qk_norm or spec.attn_gate:
        raise OpRangeError("lfm2moe: q/k norms and no attention gate")
    if spec.rotary_dim != spec.head_dim or spec.rope_scaling or spec.sliding_window:
        raise OpRangeError("lfm2moe: a full-head default RoPE, no window")
    if spec.num_kv_heads % 2 or spec.num_heads % spec.num_kv_heads or spec.num_heads % ATTN_CORES:
        raise OpRangeError(f"lfm2moe: {spec.num_heads} q / {spec.num_kv_heads} kv heads do not split "
                           f"over the attention elements and {ATTN_CORES} cores")
    qw, kvw = spec.num_heads * spec.head_dim, spec.num_kv_heads * spec.head_dim
    for what, w in (("hidden", spec.hidden), ("q width", qw), ("kv width", kvw),
                    ("dense slot", DENSE_SLOT)):
        if w % (BAND_ROWS * N_CORES) or w % 256:
            raise OpRangeError(f"lfm2moe: {what} {w} is not whole 64-row bands over {N_CORES} cores and "
                               f"256-wide k-tiles")
    # The expert width need only be whole 256-wide k-tiles: a partial band is
    # padded to a whole one (padded_ff) when the pool and kernel are compiled,
    # while the pack op reads exactly these real rows from the GGUF.
    if spec.moe_intermediate % 256:
        raise OpRangeError(f"lfm2moe: expert width {spec.moe_intermediate} is not whole 256-wide k-tiles")
    if spec.hidden % SCW:
        raise OpRangeError(f"lfm2moe: hidden {spec.hidden} is not whole {SCW}-channel conv elements")
    if 4 * spec.head_dim > 2 * kvw:
        raise OpRangeError("lfm2moe: qn | kn do not fit one meta element")


def geometry(spec: ModelSpec) -> Geometry:
    _check(spec)
    hid, ff = spec.hidden, padded_ff(spec.moe_intermediate)
    nh, kvh, hd = spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    e_a = kvw * 2
    ptab_row = max(1024, e_a)
    nhl = nh // ATTN_CORES
    xm_bytes = hid * 2
    return Geometry(
        HID=hid, FF=ff, DFF=spec.intermediate,
        NH=nh, KVH=kvh, HD=hd, ROT=spec.rotary_dim, QW=qw, KVW=kvw,
        N_CORES=N_CORES, PER_CALL=PER_CALL, CALL_BYTES=PER_CALL * CH,
        Q_PC=qw // BAND_ROWS // N_CORES, KV_PC=kvw // BAND_ROWS // N_CORES, O_PC=hid // BAND_ROWS // N_CORES,
        SC_PC=hid // BAND_ROWS // N_CORES, DOWN_PC=hid // BAND_ROWS // N_CORES,
        E_A=e_a, HPE=e_a // (hd * 4), HPO=e_a // (hd * 2), Q_AIN_ELEMS=nh // (e_a // (hd * 4)),
        K_AIN_ELEMS=kvh // (e_a // (hd * 4)),
        XN_ELEMS=roundup(hid * 2, ELEM) // ELEM, XM_ELEMS=roundup(xm_bytes + 4 * WV, ELEM) // ELEM,
        OG_ELEMS=roundup(qw * 2, ELEM) // ELEM,
        TAB_BYTES=tab_bytes(max(hid, qw, ff)), MS_U=0, MS_G=BAND_ROWS, MS_FLOATS=3 * BAND_ROWS,   # u | g | a down band
        EPS=spec.norm_eps, TOPK=spec.experts_per_tok, EXPERTS=spec.num_experts,
        SLOTS=spec.experts_per_tok, DENSE_SLOTS=spec.intermediate // DENSE_SLOT if spec.dense_lead else 0,
        # Flint's / HY V3's attention point (RB 4, the vector softmax, 4 cores of 8 heads), no sink, head dim 64
        ACORES=ATTN_CORES, NHL=nhl, VEXP=1, MLS=roundup(nhl, 32), RB=4,
        PTAB_ELEMS=ptab_row // e_a,
        W_OFF=xm_bytes % ELEM,        # the routing weights' byte offset in xm's last x element
        TAPS=TAPS, SCW=SCW, SC_ELEMS=hid // SCW,
    )


def ffn(spec: ModelSpec, dense: bool) -> Ffn:
    """The MoE layers' slot law (the experts; 4 weighted slots) or the dense layers' (the MLP as 23 slots of
    512, unweighted). Both pools put their slots at byte 0 and their down slices after them."""
    G = geometry(spec)
    ff = DENSE_SLOT if dense else G.FF
    slots = G.DENSE_SLOTS if dense else G.SLOTS
    pool_slots = G.DENSE_SLOTS if dense else G.EXPERTS
    up = (ff // BAND_ROWS) * band_bytes(G.HID)         # one slot's gate (= up) projection
    down = (G.HID // BAND_ROWS) * band_bytes(ff)
    if up != down:
        raise OpRangeError("lfm2moe: a slot's gate and down must be the same size (moe2's law)")
    return Ffn(FF=ff, UP_PC=ff // BAND_ROWS // N_CORES, SLOTS=slots, POOL_SLOTS=pool_slots,
               H_ELEMS=roundup(ff * 4, ELEM) // ELEM, H_STRIDE=roundup(ff * 4, ELEM),
               UP_BYTES=up, EXPERT_UG=2 * up, POOL_DOWN=pool_slots * 2 * up, WEIGHTED=not dense)


def lm_rows(spec: ModelSpec) -> int:
    return roundup(spec.vocab, LM_BAND_ROWS)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    G = geometry(spec)
    fm = ffn(spec, dense=False)
    fd = ffn(spec, dense=True) if spec.dense_lead else fm
    eln = G.HID * 2
    cd = {"lnw": 0, "postln": eln, "meta": 2 * eln, "conv": roundup(2 * eln + G.E_A, ELEM)}
    cd_bytes = roundup(cd["conv"] + G.TAPS * G.HID * 4, ELEM)
    a: dict[str, int] = {}
    off = 0
    h_bytes = max(fm.SLOTS * fm.H_STRIDE, fd.SLOTS * fd.H_STRIDE)
    for name, size in (("xn", G.XN_ELEMS * ELEM), ("q", G.QW * 4), ("kvn", 2 * G.KVW * 4), ("og", G.OG_ELEMS * ELEM),
                       ("b", G.HID * 4), ("c", G.HID * 4), ("u", G.HID * 4), ("y", G.XN_ELEMS * ELEM),
                       ("out", G.HID * 4), ("res", G.HID * 4), ("xm", G.XM_ELEMS * ELEM),
                       ("h", h_bytes), ("out2", G.HID * 4), ("junk", eln)):
        a[name] = off
        off += roundup(size, ELEM)
    ad_bytes = roundup(off, ELEM)
    # the mixer, after the MoE pool's experts, at the same offset in every layer's pool (the dense
    # layers' MLP slots sit far below it), so one conv build and one attention build serve all layers
    mix = fm.POOL_DOWN + fm.POOL_SLOTS * fm.UP_BYTES
    if fd.POOL_DOWN + fd.POOL_SLOTS * fd.UP_BYTES > mix:
        raise OpRangeError("lfm2moe: the dense MLP's slots reach past the mixer")
    sq = (G.HID // BAND_ROWS) * band_bytes(G.HID)       # one [hidden, hidden] projection
    p = {"b": mix, "c": mix + sq, "u": mix + 2 * sq, "out": mix + 3 * sq, "q": mix}
    p["k"] = p["q"] + (G.QW // BAND_ROWS) * band_bytes(G.HID)
    p["v"] = p["k"] + (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    p["o"] = p["v"] + (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    end = max(mix + 4 * sq, p["o"] + (G.HID // BAND_ROWS) * band_bytes(G.QW))
    pool_bytes = roundup(end, MB)
    if pool_bytes >= 1 << 32:
        raise OpRangeError("lfm2moe: a layer pool past 4 GiB does not fit a 32-bit fill offset")
    rows = lm_rows(spec)
    bands = rows // LM_BAND_ROWS
    lm_band = 4 * (G.HID // 256) * LM_TILE
    kv_row = 2 * G.E_A
    ptab_row = G.PTAB_ELEMS * G.E_A
    return Layout(
        CD_LNW=cd["lnw"], CD_POSTLN=cd["postln"], CD_META=cd["meta"], CD_CONV=cd["conv"], CD_BYTES=cd_bytes,
        AD_XN=a["xn"], AD_Q=a["q"], AD_KVN=a["kvn"], AD_OG=a["og"], AD_B=a["b"], AD_C=a["c"], AD_U=a["u"],
        AD_Y=a["y"], AD_OUT=a["out"], AD_RES=a["res"], AD_XM=a["xm"],
        AD_W=a["xm"] + (G.XM_ELEMS - 1) * ELEM + G.W_OFF, AD_H=a["h"], AD_OUT2=a["out2"],
        AD_JUNK=a["junk"], AD_BYTES=ad_bytes,
        POOL_MIX=mix, POOL_B=p["b"], POOL_C=p["c"], POOL_U=p["u"], POOL_OUT=p["out"],
        POOL_Q=p["q"], POOL_K=p["k"], POOL_V=p["v"], POOL_O=p["o"], POOL_BYTES=pool_bytes,
        STATE_BYTES=(G.TAPS - 1) * G.HID * 4,
        KV_ROW=kv_row, PTAB_ROW=ptab_row, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * ptab_row,
        LM_ROWS=rows, LMHEAD_BANDS=bands, LMHEAD_BAND_BYTES=lm_band,
        LMHEAD_POOL_BYTES=roundup(roundup(bands, LM_CORES) * lm_band, MB),
        ELN=eln,
    )


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> Lfm2MoeRecipe:
    return Lfm2MoeRecipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def kinds(spec: ModelSpec) -> list[tuple[str, bool]]:
    """Each layer's (mixer, dense FFN?)."""
    return [(MIX_CONV if t == SHORT_CONV else MIX_ATT, i < spec.dense_lead) for i, t in enumerate(spec.layer_types)]


def layer_type_names(spec: ModelSpec) -> list[str]:
    """The manifest's per-layer type names: the spec's own, `_dense` for the dense-MLP layers."""
    return [t + ("_dense" if i < spec.dense_lead else "") for i, t in enumerate(spec.layer_types)]


def attn_flags(G: Geometry, probe_env=()) -> list[str]:
    """attn.h for LFM2-MoE: the q / k norm before the rotation, no gate, no sink, the vector softmax over 4
    cores of 8 heads, RB 4 (HY V3's point at head dim 64)."""
    flags = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
             "-DATTN_QKNORM=1", "-DATTN_QKNORM_POST=0", f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}",
             f"-DATTN_NHL={G.NHL}"]
    if G.RB > 1:
        flags.append(f"-DATTN_RB={G.RB}")
    if G.PTAB_ELEMS > 1:
        flags.append("-DATTN_PTAB_SPLIT=1")
    for k, v in probe_env:
        flags.append(f"-D{k}={v}")
    return flags


def ln_flags(G: Geometry) -> list[str]:
    return [f"-DLN_N={G.HID}", f"-DLN_EPS={G.EPS:g}f"]


PROBE_VARS = ("ATTN_NULL", "LM_NULL")


def probe_env() -> dict[str, str]:
    return {k: os.environ[k] for k in PROBE_VARS if os.environ.get(k)}


def gguf_quant(spec: ModelSpec) -> str:
    """The law the GGUF view's pools hold: q4_1 (our pure Q4_0 exactly: m = -8 d)."""
    return "q4_1"


def finalize_gguf_manifest(spec: ModelSpec, manifest: dict) -> None:
    """The manifest IS the GGUF view (there is no q4nx container): core.cpp reads a model.gguf through the
    nested `gguf` section, so the section is the same manifest, nested once. And the per-layer type names:
    the dense-MLP layers are their own types (`<type>_dense`)."""
    manifest["layers"] = layer_type_names(spec)
    if "gguf" not in manifest:
        manifest["gguf"] = copy.deepcopy(manifest)


def _q4_op(tensor: str, dst: int, rows: int, in_dim: int, row0: int = 0, col0: int = 0, experts: int = 1,
           row_step: int = 0, col_step: int = 0, expert_bytes: int = 0, band_stride: int = 0) -> dict:
    """runtime pack op fl_q4 (flint-runtime.patch): for e < experts, the rows x in_dim block at
    (row0 + e row_step, col0 + e col_step) of a Q4_0 / Q4_1 GGUF tensor (the trailing dims flattened
    into rows) as 64-row bands of q4_1 chunks (std_perm_gguf's byte law, scale and min bf16 RNE),
    band b at dst + e expert_bytes + b band_stride (band_stride 0: the bands contiguous)."""
    if rows % BAND_ROWS or in_dim % 256 or col0 % 32 or col_step % 32:
        raise OpRangeError(f"lfm2moe: {tensor}: {rows} x {in_dim} at column {col0} is not whole bands and k-tiles")
    return {"op": "fl_q4", "tensor": tensor, "dst": dst, "rows": rows, "in_dim": in_dim, "row0": row0,
            "chunk0": col0, "experts": experts, "n_total": row_step, "cols": col_step,
            "expert_bytes": expert_bytes, "stripe_bytes": band_stride}


def _layer_plan(spec: ModelSpec, mixer: str, dense: bool) -> dict:
    G, L, F = geometry(spec), layout(spec), ffn(spec, dense)
    b = "blk.{l}."
    hid, bbh = G.HID, band_bytes(G.HID)
    ug = dict(expert_bytes=F.EXPERT_UG, band_stride=2 * bbh)
    n = F.POOL_SLOTS
    if dense:     # the MLP's rows (gate, up) and columns (down) in 512-wide slots
        ffn_ops = [
            _q4_op(b + "ffn_gate.weight", 0, F.FF, hid, experts=n, row_step=F.FF, **ug),
            _q4_op(b + "ffn_up.weight", bbh, F.FF, hid, experts=n, row_step=F.FF, **ug),
            _q4_op(b + "ffn_down.weight", F.POOL_DOWN, hid, F.FF, experts=n, col_step=F.FF, expert_bytes=F.UP_BYTES),
        ]
    else:
        # The pool slot is padded (padded_ff), but the GGUF holds a real expert
        # width. `rows` / `in_dim` are SOURCE reads: only the real 1792 rows and
        # columns come from the file; the pad lanes stay zero in the destination
        # slot. Expert e still begins at row e*real_ff (gate/up) or e*hid (down)
        # of the flat tensor, so the source strides are the real widths.
        real = spec.moe_intermediate
        ffn_ops = [
            _q4_op(b + "ffn_gate_exps.weight", 0, real, hid, experts=n, row_step=real, **ug),
            _q4_op(b + "ffn_up_exps.weight", bbh, real, hid, experts=n, row_step=real, **ug),
            _q4_op(b + "ffn_down_exps.weight", F.POOL_DOWN, hid, real, experts=n, row_step=hid,
                   expert_bytes=F.UP_BYTES),
        ]
    consts = [
        {"op": "put", "tensor": b + "attn_norm.weight", "dst": L.CD_LNW, "cap": L.ELN},
        {"op": "put", "tensor": b + "ffn_norm.weight", "dst": L.CD_POSTLN, "cap": L.ELN},
    ]
    if mixer == MIX_CONV:
        ip = b + "shortconv.in_proj.weight"      # [3 hidden, hidden]: B, C, u down the rows
        mix = [_q4_op(ip, L.POOL_B, hid, hid, row0=0), _q4_op(ip, L.POOL_C, hid, hid, row0=hid),
               _q4_op(ip, L.POOL_U, hid, hid, row0=2 * hid),
               _q4_op(b + "shortconv.out_proj.weight", L.POOL_OUT, hid, hid)]
        # The GGUF stores the taps F32 [hidden, taps]; the core wants them TAP-major (32 consecutive
        # channels of one tap contiguous), kept f32: [hidden, taps] -> [taps, hidden], 4-byte elements.
        consts.append({"op": "transpose", "tensor": b + "shortconv.conv.weight", "dst": L.CD_CONV,
                       "rows": hid, "cols": G.TAPS, "elem": 4})
    else:
        mix = [_q4_op(b + "attn_q.weight", L.POOL_Q, G.QW, hid), _q4_op(b + "attn_k.weight", L.POOL_K, G.KVW, hid),
               _q4_op(b + "attn_v.weight", L.POOL_V, G.KVW, hid),
               _q4_op(b + "attn_output.weight", L.POOL_O, hid, G.QW)]
        consts += [
            {"op": "put", "tensor": b + "attn_q_norm.weight", "dst": L.CD_META, "cap": 2 * G.HD},
            {"op": "put", "tensor": b + "attn_k_norm.weight", "dst": L.CD_META + 2 * G.HD, "cap": 2 * G.HD},
        ]
    if dense and block_on():
        ffn_ops += _dense_gemm_ops(spec)
    return {"pool": ffn_ops + mix, "consts": consts}


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    out = {}
    for nm, (mixer, dense) in zip(layer_type_names(spec), kinds(spec)):
        out.setdefault(nm, _layer_plan(spec, mixer, dense))
    return {
        "pool_bytes": L.POOL_BYTES, "chunk_bytes": CH,
        "layer_types": out,
        # the head is tied: the Q8_0 token embedding, through lm_head_q8's f32-scale chunks
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "lmhead_q8", "tensor": EMBED, "dst": 0, "chunk_bytes": LM_TILE,
                             "rows": L.LM_ROWS, "in_dim": G.HID}]},
        "embed": {"tensor": EMBED, "dim": G.HID},
        "norm": {"tensor": "token_embd_norm.weight", "bytes": L.ELN},
    }


def route_step(spec: ModelSpec) -> dict:
    """hy_route (hyv3-runtime.patch), unchanged: no shared expert, so the slots are the top k."""
    L, G = layout(spec), geometry(spec)
    return {"op": "hy_route", "kernel": "lm_moe", "a_xm": L.AD_XM, "a_w": L.AD_W, "hidden_real": G.HID,
            "slots": G.SLOTS, "router": "blk.{l}.ffn_gate_inp.weight", "bias": "blk.{l}.exp_probs_b.bias",
            "experts": spec.num_experts, "topk": spec.experts_per_tok, "scale": router_scale(spec), "shared": []}


# ---- block prefill (the Gemma 4 route: commit 8c27569; .plans/lfm2moe-native.md "block prefill")
def block_on() -> bool:
    """The block prefill route's kernels and manifest keys.

    The engine refuses `kind:"lfm"` block prefill for a GGUF-direct model (core.cpp:
    `enable_gguf_block` accepts only dense kinds) and falls back to step prefill, so
    building the token-batched `mbl` stream for this family ships a set nothing runs.
    It also cannot be built at every expert width: mbl holds all of h in one core's
    L1, which overflows once the width is padded to a whole band (LFM2.5-8B-A1B's
    1792 -> 2048). Both facts are family laws, not a per-model choice, so the block
    route is off here. LFM2MOE_BLOCK=1 is the explicit opt-in for a future engine
    that implements the hybrid kind; the shipped sets use step prefill.
    """
    return os.environ.get("LFM2MOE_BLOCK") == "1"


MB_NT = 8             # designs/lfm2moe_batch/mbl.py token slots per expert visit (one 8-token MAC tile)


def moe_batch_slots(spec: ModelSpec) -> list[int]:
    """mbl's dispatch lengths: a binary ladder down from the expert count (an unused slot streams a real expert and
    is wasted, so the rungs decide the padding)."""
    return [s for s in (spec.num_experts >> i for i in range(30)) if s >= 8 and s % 8 == 0]


def _dense_gemm_ops(spec: ModelSpec) -> list[dict]:
    """The leading dense layers' MLP as three CONTIGUOUS GEMM operands (up, gate, down) in the free region of the
    layer's pool between the 512-wide slots the sequential program reads and POOL_MIX. The slots interleave gate
    and up bands, which is no GEMM operand; the extra 3 x 15 MB a dense layer is read once at load."""
    G, L = geometry(spec), layout(spec)
    F = ffn(spec, dense=True)
    b = "blk.{l}."
    ff, hid = spec.intermediate, G.HID
    base = roundup(F.POOL_DOWN + F.POOL_SLOTS * F.UP_BYTES, MB)
    ug = (ff // BAND_ROWS) * band_bytes(hid)
    dn = (hid // BAND_ROWS) * band_bytes(ff)
    if base + 2 * ug + dn > L.POOL_MIX:
        raise OpRangeError("lfm2moe: the dense MLP's GEMM operands do not fit between its slots and the mixer")
    return [_q4_op(b + "ffn_up.weight", base, ff, hid), _q4_op(b + "ffn_gate.weight", base + ug, ff, hid),
            _q4_op(b + "ffn_down.weight", base + 2 * ug, hid, ff)]


def moe_batch(spec: ModelSpec) -> dict:
    """The token-batched expert kernel of the block route (designs/lfm2moe_batch/mbl.py): the `moe_batch` entry of a
    layer type's gemm_block, plus the contexts / kernels / globals / builds it needs. One xclbin (context `mb`),
    one instruction stream per dispatch length; the runtime patches the slots' expert offsets (`moebatch`,
    moe_geometry's pool law: expert_stripes = 2 up_bytes is mbl's EXP_UG). y leaves [rbg][half][row][1024 B]
    (designs/lfm2moe_batch/layout.py), not moe_batch's C-tile order."""
    G, L, F = geometry(spec), layout(spec), ffn(spec, dense=False)
    slots = moe_batch_slots(spec)
    args = ["pool", "mb_x", "mb_h", "mb_y"]
    check_buffer_args("moe_batch", args)
    out: dict = {"contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    for s in slots:
        name = f"mb_s{s}"
        out["contexts"].setdefault("mb", f"mb_s{slots[0]}/final.xclbin")
        out["kernels"][name] = {"context": "mb", "insts": f"{name}/insts.bin", "patch": "moebatch", "build": name}
        out["builds"][name] = {"design": "lfm2moe_batch/mbl.py",
                               "build_dir": f"lfm2moe_batch/build_s{s}_h{G.HID}_f{G.FF}_e{spec.num_experts}",
                               "env": {"MBL_SLOTS": str(s), "MBL_NT": str(MB_NT), "MBL_HID": str(G.HID),
                                       "MBL_FF": str(G.FF), "MBL_EXPERTS": str(spec.num_experts),
                                       "MBL_POOL_DOWN": str(F.POOL_DOWN), "MBL_POOL_BYTES": str(L.POOL_MIX)}}
    out["globals"]["mb_x"] = slots[0] * G.HID * MB_NT * 2
    out["globals"]["mb_h"] = slots[0] * G.FF * MB_NT * 2
    out["globals"]["mb_y"] = slots[0] * G.HID * MB_NT * 4
    out["moe_batch"] = {"kernels": {str(s): f"mb_s{s}" for s in slots}, "args": args, "nt": MB_NT,
                        "y_layout": "rbg_half_row"}
    return out


def gemm_route(spec: ModelSpec, max_ctx: int = 4096) -> dict | None:
    """Per layer type the `gemm_block` (kind `lfm`) the driver reads, plus the contexts / kernels / globals / builds
    the route adds. None under LFM2MOE_BLOCK=0. A layer is the mixer's fused input GEMM (conv in_proj B | C | u,
    attention q | k | v), the host's middle (the conv, or attention over the block), the out GEMM, the host's
    residual / ffn_norm, and then the leading dense layers' two MLP GEMMs or the routed experts (host router, the
    token-batched mbl stream). The q4_1 GEMM reads the pool's bytes as they are: B | C | u and q | k | v are
    contiguous pool ops."""
    if not block_on():
        return None
    L, G, T = layout(spec, max_ctx), geometry(spec), GEMM_T
    hid, qw, kvw = G.HID, G.QW, G.KVW
    shapes: set[tuple[int, int]] = set()

    def ctx(N: int, K: int) -> str:
        if N % 256 or K % 256:
            raise OpRangeError(f"lfm2moe gemm route: [{N}, {K}] is not a multiple of 256 in both dims")
        shapes.add((N, K))
        return f"gemm_n{N}_k{K}"

    def run(N: int, K: int, w: str) -> dict:
        return {"op": "run", "kernel": ctx(N, K), "args": [w, f"gemm_x_k{K}", f"gemm_y_n{N}"]}

    def idx(pool: list[dict], suffix: str, dst: int | None = None) -> int:
        hits = [i for i, o in enumerate(pool) if o["tensor"].endswith(suffix) and (dst is None or o["dst"] == dst)]
        if len(hits) != 1:
            raise OpRangeError(f"lfm2moe gemm route: {len(hits)} pack ops end in {suffix!r}, want one")
        return hits[0]

    def run_of(pool: list[dict], ids: list[int]) -> dict:
        ops = [pool[i] for i in ids]
        for a, b in zip(ops, ops[1:]):       # one byte-contiguous run, as the engine needs
            if b["dst"] != a["dst"] + a["rows"] // BAND_ROWS * band_bytes(a["in_dim"]):
                raise OpRangeError("lfm2moe gemm route: the pool ops of a GEMM weight are not contiguous")
        return {"from": "pool", "ops": ids}

    mbatch = moe_batch(spec)
    out: dict = {"layer_types": {}, "contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    for k in ("contexts", "kernels", "globals", "builds"):
        out[k].update(mbatch[k])
    seen: set[str] = set()
    for nm, (mixer, dense) in zip(layer_type_names(spec), kinds(spec)):
        if nm in seen:
            continue
        seen.add(nm)
        pool = _layer_plan(spec, mixer, dense)["pool"]
        gb: dict = {"kind": "lfm", "mixer": mixer, "t": T, "eps": spec.norm_eps,
                    "cd_ln": L.CD_LNW, "cd_ffn": L.CD_POSTLN}
        if mixer == MIX_CONV:
            ip = "shortconv.in_proj.weight"
            w = {"gin_w": run_of(pool, [idx(pool, ip, L.POOL_B), idx(pool, ip, L.POOL_C), idx(pool, ip, L.POOL_U)]),
                 "gout_w": run_of(pool, [idx(pool, "shortconv.out_proj.weight")])}
            gb.update(program=[run(3 * hid, hid, "gin_w"), run(hid, hid, "gout_w")], weights=w,
                      taps=G.TAPS, cd_conv=L.CD_CONV)
        else:
            w = {"gqkv3_w": run_of(pool, [idx(pool, "attn_q.weight"), idx(pool, "attn_k.weight"),
                                          idx(pool, "attn_v.weight")]),
                 "go_w": run_of(pool, [idx(pool, "attn_output.weight")])}
            gb.update(program=[run(qw + 2 * kvw, hid, "gqkv3_w"), run(hid, qw, "go_w")], weights=w,
                      qw=qw, kvw=kvw, nh=G.NH, kvh=G.KVH, hd=G.HD, rot=spec.rotary_dim,
                      cd_qnorm=L.CD_META, cd_knorm=L.CD_META + 2 * G.HD)
        if dense:
            ff = spec.intermediate
            up, gate, down = _dense_gemm_ops(spec)
            fw = {"gffn_ug_w": run_of(pool, [idx(pool, "ffn_up.weight", up["dst"]), idx(pool, "ffn_gate.weight", gate["dst"])]),
                  "gffn_down_w": run_of(pool, [idx(pool, "ffn_down.weight", down["dst"])])}
            gb.update(ff=ff, ffn_program=[run(2 * ff, hid, "gffn_ug_w"), run(hid, ff, "gffn_down_w")], ffn_weights=fw)
        else:
            r = route_step(spec)
            gb.update(route={"router": r["router"], "bias": r["bias"], "experts": r["experts"], "topk": r["topk"],
                             "scale": r["scale"]}, moe_batch=mbatch["moe_batch"])
        out["layer_types"][nm] = gb
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


CONV_ARGS = ["pool", "xres", "consts", "state", "act"]
ATT_ARGS = ["pool", "xres", "consts", "state", "act", "ptab"]


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    check_buffer_args("lm_conv", CONV_ARGS)
    check_buffer_args("lm_att", ATT_ARGS)
    out = {
        "contexts": {"lm_moe": "lm_moe/final.xclbin", "ln": "ln/final.xclbin", "lm": "lm_head_q8/final.xclbin"},
        "kernels": {
            "lm_moe": {"context": "lm_moe", "insts": "lm_moe/insts.bin", "patch": "moeroute2", "build": "lm_moe"},
            "ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
            "lm": {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"},
        },
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": G.HID * 4, "zero": G.HID * 4, "normw": G.HID * 2, "xresf": G.HID * 4, "hn": G.HID * 2,
                    "logits": L.LM_ROWS * 4, "lmpool": L.LMHEAD_POOL_BYTES},
    }
    ks = kinds(spec)
    if any(m == MIX_CONV for m, _ in ks):
        out["contexts"]["lm_conv"] = "lm_conv/final.xclbin"
        out["kernels"]["lm_conv"] = {"context": "lm_conv", "insts": "lm_conv/insts.bin", "build": "lm_conv"}
    if any(m == MIX_ATT for m, _ in ks):
        out["contexts"]["lm_att"] = "lm_att/final.xclbin"
        out["kernels"]["lm_att"] = {"context": "lm_att", "insts": "lm_att/insts.bin", "patch": "attnpos",
                                    "build": "lm_att", "window": 0}
        out["globals"]["ptab"] = {"per_row": L.PTAB_ROW, "inv_freq": spec.rope_inv_freq(), "window": 0}
    if spec.dense_lead:
        out["contexts"]["lm_dense"] = "lm_dense/final.xclbin"
        out["kernels"]["lm_dense"] = {"context": "lm_dense", "insts": "lm_dense/insts.bin", "build": "lm_dense"}
    for nm, (mixer, dense) in zip(layer_type_names(spec), ks):
        if nm in out["layer_types"]:
            continue
        if mixer == MIX_CONV:
            first = {"op": "run", "kernel": "lm_conv", "args": CONV_ARGS}
            state = {"kind": "linear", "bytes": L.STATE_BYTES}
        else:
            first = {"op": "run", "kernel": "lm_att", "args": ATT_ARGS}
            state = {"kind": "kv", "row": L.KV_ROW}
        ffn_steps = ([{"op": "run", "kernel": "lm_dense", "args": ["pool", "xres", "act"]}] if dense else
                     [route_step(spec), {"op": "run", "kernel": "lm_moe", "args": ["pool", "xres", "act"]}])
        out["layer_types"][nm] = {"buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": state},
                                  "program": [first] + ffn_steps}
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
    G, L = geometry(spec), layout(spec)
    out = {
        "lm_moe": {"design": "lfm2moe/lm.py", "build_dir": f"lfm2moe/build_moe_h{G.HID}_f{G.FF}_s{G.SLOTS}",
                   "env": {"LM_PART": "moe"}},
        "ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{G.HID}_{G.EPS:g}",
               "env": {"LN_N": str(G.HID), "LN_EPS": f"{G.EPS:g}"}},
        "lm_head_q8": {"design": "lm_head_q8/lm_head_q8.py",
                       "build_dir": f"lm_head_q8/build_gguf_n{L.LM_ROWS}_k{G.HID}_c{LM_CORES}",
                       "env": {"LMHEAD_N": str(L.LM_ROWS), "LMHEAD_K": str(G.HID), "LMHEAD_CORES": str(LM_CORES),
                               "LMHEAD_SCALES_F32": "1"}},
    }
    ks = kinds(spec)
    if any(m == MIX_CONV for m, _ in ks):
        out["lm_conv"] = {"design": "lfm2moe/lm.py", "build_dir": f"lfm2moe/build_conv_h{G.HID}_t{G.TAPS}",
                          "env": {"LM_PART": "conv"}}
    if any(m == MIX_ATT for m, _ in ks):
        out["lm_att"] = {"design": "lfm2moe/lm.py", "build_dir": f"lfm2moe/build_att_h{G.HID}_d{G.HD}",
                         "env": {"LM_PART": "att"}}
    if spec.dense_lead:
        out["lm_dense"] = {"design": "lfm2moe/lm.py",
                           "build_dir": f"lfm2moe/build_dense_h{G.HID}_f{DENSE_SLOT}_s{G.DENSE_SLOTS}",
                           "env": {"LM_PART": "dense"}}
    gr = gemm_route(spec)
    if gr:
        out.update(gr["builds"])
    return out


def moe_geometry(spec: ModelSpec) -> dict:
    """stream_patch::MoeGeometry for moeroute2 over lm_moe's fills: slot j is compiled as pool expert j, and
    the pool holds the 64 routed experts from byte 0; nothing past POOL_MIX is an expert."""
    L, F = layout(spec), ffn(spec, dense=False)
    return {"stripe": F.UP_BYTES, "up_bytes": F.UP_BYTES, "down_core": F.UP_BYTES, "pool_down": F.POOL_DOWN,
            "share_up": L.POOL_MIX, "share_gate": L.POOL_MIX, "share_down": L.POOL_MIX,
            "experts": spec.num_experts, "topk": F.SLOTS}


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    return {"hidden": G.HID, "vocab": L.LM_ROWS, "real_vocab": spec.real_vocab, "chunk_bytes": CH,
            "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": LM_TILE,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": spec.rope_inv_freq(), "moe": moe_geometry(spec)}


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "intermediate_size": spec.intermediate, "moe_intermediate_size": spec.moe_intermediate,
            "num_experts": spec.num_experts, "num_experts_per_tok": spec.experts_per_tok,
            "num_dense_layers": spec.dense_lead, "conv_L_cache": spec.conv_kernel,
            "routed_scaling_factor": router_scale(spec), "use_expert_bias": True, "norm_topk_prob": True,
            "tie_word_embeddings": True}


GEN_KERNELS = "designs/lfm2moe/gen_kernels.py"
KERNEL_SOURCES = [
    "designs/lfm2moe/*.py", "designs/lfm2moe_batch/*.py", "designs/lfm2moe_batch/*.cc", "designs/lfm2moe_batch/*.h",
    "designs/gemm_q4_prefill/*.py", "designs/gemm_q4_prefill/*.h", "designs/gemm_q4_prefill/*.cc",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h",
    "designs/attn/*.cc", "designs/attn/*.h", "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py",
    "designs/lin_layer/ln_nr.cc", "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.h", "designs/lm_head_q8/*.cc",
    "include/vecmath.h", "include/scalar_fp.h", "ironutil.py", "build_design.py",
]
