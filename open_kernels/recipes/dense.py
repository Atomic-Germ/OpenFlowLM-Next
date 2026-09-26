"""The dense recipe (Qwen3 dense, Llama 3, Gemma 3, HunYuan dense, Granite, Phi-3): ModelSpec -> everything
designs/dense/dx.py, the packers and the driver need for a GQA + gated-FFN
decoder layer.

    ln -> gemv q | k | v -> attention ([q/k RMSNorm,] RoPE, no gate) -> gemv o
       -> [post-attention ln] -> +residual -> ln -> gemv up | gate -> act(gate) * up -> gemv down
       -> [post-FFN ln] -> +residual

The family differences are spec fields: Qwen3 has q/k RMSNorm and eps 1e-6;
Llama 3 has no q/k norms, eps 1e-5 and the llama3 RoPE frequency scaling
(host side: the position table takes the spec's inverse frequencies); Gemma 3
has GeGLU-tanh, the sandwich norms in brackets above, and two layer types --
`dense_local` (a 1024-row sliding window, its own RoPE theta) and `dense`
(global, linearly scaled RoPE) -- served by ONE design: the window is a
per-token patch of the KV fill (attnpos), the tables are two `ptab` globals.
HunYuan dense is Llama 3's shape with q/k RMSNorm applied AFTER RoPE, which is
a family property (`QKNORM_POST_ROPE`), not a spec field, and one static RoPE
theta the spec builder folds the NTK alpha into. Qwen2.5 is Llama 3's shape with
a per-channel bias on q, k and v (`QKV_BIAS_FAMILIES`, attn.h's ATTN_QKV_BIAS):
the three vectors ride in `consts` as bf16 and stream into the attention core
beside the projections they belong to. Phi-3 rotates only the first
`rotary_dim` dims of a head (96 of 128) and scales cos / sin by longrope's attention
factor, which rides on the position table (`programs`: the ptab global's `scale`).

One xclbin, ONE instruction stream per layer (no routing read, so no part
split). Same 8 main cores as the MoE designs (w / x / y streams), the ln
helper core, the attention helper core; the same six buffer arguments
(pool, xres, consts, kv, act, ptab). The lm_head is a q4_1 GEMV
(designs/lm_head_q4), the final norm the ln design at this width.

Element sizes, all derived: x-stream elements are 4 KB (ELEM), the ln core's
elements are HID*2 bytes (ELN: x as two f32 halves, w / xn as one bf16
element), the attention core's elements are one KV row half (E_A = KVH*HD
bf16), so a q element carries KVH/2 heads and an og element KVH heads. For
the 27B these rules give the sizes ax.py uses (4096 / 4096 / 1024).
"""
from __future__ import annotations

import os

from dataclasses import dataclass

from .catalogue import LIMITS, OpRangeError, check_buffer_args, require
from .qwen36moe import (BAND_ROWS, CHUNK, ELEM, GEMM_T, MB, _op_index, band_bytes, proj_op, q4_chunks,
                        mixed_check, quant_check, require_gemv, role_bytes, roundup, tab_bytes)
from .spec import DENSE, DENSE_LOCAL, ModelSpec


@dataclass(frozen=True)
class DenseLayout:
    # consts: [lnw (ELN)][postln (ELN)][meta: qn bf16 HD @0 | kn @HD*2 (E_A)][sandwich: preffn (ELN)][postffn (ELN)]
    #         [qkv bias, bf16: qb (QW) | kb (KVW) | vb (KVW)]  -- only a family that has one; -1 otherwise
    CD_LNW: int; CD_POSTLN: int; CD_META: int; CD_PREFFN: int; CD_POSTFFN: int
    CD_QB: int; CD_KB: int; CD_VB: int; CD_BYTES: int
    # act: the DDR bounce between stages (T / T2: the sandwich norms' outputs, f32 HID)
    AD_XN: int; AD_Q: int; AD_KVN: int; AD_OG: int; AD_OUT: int; AD_RES: int; AD_XM: int
    AD_H: int; AD_OUT2: int; AD_JUNK: int; AD_T: int; AD_T2: int; AD_BYTES: int
    # pool
    POOL_Q: int; POOL_K: int; POOL_V: int; POOL_O: int; POOL_UP: int; POOL_GATE: int; POOL_DOWN: int; POOL_BYTES: int
    # KV / ptab / lm_head
    KV_ROW: int; PTAB_ROW: int; MAX_CTX: int; KV_BYTES: int; PTAB_BYTES: int
    LMHEAD_POOL_BYTES: int; LMHEAD_BANDS: int; LMHEAD_BAND_BYTES: int
    ELN: int; E_A: int

    def constants(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DenseGeometry:
    N_CORES: int; HID: int; FF: int; NH: int; KVH: int; HD: int; ROT: int; GATE: bool; QKNORM: bool
    QKNORM_POST: bool                                               # the norm weight multiplies after RoPE (HunYuan)
    QKVB: bool                                                      # q/k/v carry a per-channel bias (Qwen2)
    PTAB_ELEMS: int; PTAB_CS_ELEM: int                              # fifo elements per position record; which one holds cos / sin
    EPS: float; ACT: str; SANDWICH: bool; WINDOW: int
    PER_CALL: int; CALL_BYTES: int                                  # chunks per weight element (1 when the table is wide)
    QW: int; KVW: int
    Q_PC: int; KV_PC: int; O_PC: int; UP_PC: int; DOWN_PC: int      # bands per core
    XN_ELEMS: int; XN_BLOCKS: int; OG_ELEMS: int; OG_BLOCKS: int    # x-stream elements / 32-blocks of the activations
    H_ELEMS: int; H_BLOCKS: int; XM_ELEMS: int
    HPE: int; HPO: int                                              # heads per ain element / per og element
    Q_AIN_ELEMS: int; K_AIN_ELEMS: int; OG_AOUT_ELEMS: int
    TAB_BYTES: int; KWIDE: int
    MS_U: int; MS_G: int; MS_FLOATS: int                            # the up / gate band scratch
    VEXP: int; MLS: int                                             # batched softmax exponentials; ml stride
    ACORES: int; NHL: int; RB: int                                  # attention cores; heads each owns; rows per call


@dataclass(frozen=True)
class DenseRecipe:
    spec: ModelSpec
    layout: DenseLayout
    geo: DenseGeometry
    max_ctx: int = 4096


# families whose q/k RMSNorm weight multiplies AFTER the rotation (HunYuan's
# query_layernorm(apply_rotary_pos_emb(q))); everyone else norms first.
QKNORM_POST_ROPE = ("hunyuan",)
# families whose q/k/v projections carry a per-channel bias. Like the post-RoPE norm this
# is a family property, not a spec field: every Qwen2 has it, and spec_hash() covers every
# field, so a field would move every shipped model's hash.
# o_proj and the FFN have none on the families listed here. GPT-OSS breaks both halves of
# that - attention_bias covers its o_proj, and all three expert projections carry one - so
# adding it here is not enough on its own; it needs a fourth consts slot (gptoss-bringup.md).
QKV_BIAS_FAMILIES = ("qwen2",)
DENSE_FAMILIES = ("qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2")


def qkv_bias(spec: ModelSpec) -> bool:
    return spec.family in QKV_BIAS_FAMILIES


def _ptab_row(spec: ModelSpec) -> int:
    """One position record: [pos i32 @0][nf i32 @4][cos f32[ROT/2] @512][sin @512 + 2*ROT].
    It is streamed to the attention core through the same fifo as q / k / v, so it is at
    least one element wide -- and, where the element is narrower, a whole number of them."""
    return max(1024, spec.num_kv_heads * spec.head_dim * 2)


def _ptab_check(spec: ModelSpec, e_a: int) -> None:
    """Every family until Qwen2.5-3B had a record exactly one element wide, so the design
    read cos / sin at a fixed offset into the one element it acquired. At 2 kv heads and
    head dim 128 the element is 512 B and the record 1024, which would have desynchronised
    the whole stream by an element. attn.h's ATTN_PTAB_SPLIT covers the case; these are its
    limits."""
    row = _ptab_row(spec)
    if row % e_a:
        raise OpRangeError(f"dense: a {row}-byte position record is not a whole number of "
                           f"{e_a}-byte attention elements")
    if row // e_a > 2:
        raise OpRangeError(f"dense: a {row}-byte position record spans {row // e_a} "
                           f"{e_a}-byte elements; attn.h handles one or two")
    if 512 // e_a != (512 + 4 * spec.rotary_dim - 1) // e_a:
        raise OpRangeError(f"dense: cos / sin straddle two {e_a}-byte attention elements "
                           f"(record offsets 512 to {512 + 4 * spec.rotary_dim})")


def lm_rows(spec: ModelSpec) -> int:
    """lm_head rows: the vocabulary rounded up to a whole 64-row band. Every family
    but HunYuan already publishes a padded vocab_size; HunYuan's 128167 is not one,
    and the head's bands (and the logits buffer) are sized by the rounded count while
    `real_vocab` still bounds the ids."""
    return roundup(spec.vocab, BAND_ROWS)


def cores_for(spec: ModelSpec) -> int:
    """Main cores this spec's widths can be split over, at most LIMITS["n_cols"].

    A GEMV band is BAND_ROWS rows and every width has to divide into whole bands per
    core, so the core count is bounded by the geometry rather than chosen. Gemma 3 12B
    is the case that forced this: hidden 3840 is not a multiple of 64*8, while its
    intermediate (15360), q width (4096) and kv width (2048) all are -- so the model
    was unbuildable at a fixed 8 while 4 serves every one of its widths.

    Fewer cores is measured to be nearly free, which is what makes this a widening and
    not a compromise: on gemma3-4b, 8 cores against 4 is 1.18x at position 0 and
    1.06-1.09x with a context, because the weight stream is DMA-bound and the array is
    one memory client whatever its width (T45 / tasks/0151, established there for the
    encoder GEMM and confirmed here for the decoder GEMV).

    Every spec that divides by 64*8 still gets 8 and rebuilds byte-identical.
    """
    widths = (spec.hidden, spec.intermediate, spec.attn_q_width, spec.attn_kv_width)
    for n in range(LIMITS["n_cols"], 0, -1):
        if all(w % (BAND_ROWS * n) == 0 for w in widths):
            return n
    return 1


def _check(spec: ModelSpec) -> None:
    n = cores_for(spec)
    if spec.family not in DENSE_FAMILIES:
        raise OpRangeError(f"dense recipe given a {spec.family!r} spec")
    if spec.activation not in ("silu", "gelu_tanh"):
        raise OpRangeError(f"dense: activation {spec.activation!r} (silu | gelu_tanh)")
    if spec.has_local and spec.sliding_window <= 0:
        raise OpRangeError("dense: dense_local layers need a positive sliding_window")
    quant_check(spec, "dense")
    mixed_check(spec, "dense", ("attn", "ffn"))
    if spec.quant_of("experts") == "q8" or spec.quant_of("shared") == "q8" or spec.quant_of("linear") == "q8":
        raise OpRangeError("dense: this family has only the 'attn' and 'ffn' roles")
    if not spec.has_dense or spec.has_linear or spec.has_full or spec.intermediate == 0:
        raise OpRangeError("dense: every layer must be a dense layer with an FFN")
    if spec.attn_gate:
        raise OpRangeError("dense: the attention gate is a MoE-family feature; dx.py has no gate elements")
    for what, v in (("hidden", spec.hidden), ("intermediate", spec.intermediate), ("q width", spec.attn_q_width),
                    ("kv width", spec.attn_kv_width)):
        if v % (BAND_ROWS * n):
            raise OpRangeError(f"qwen3: {what} {v} is not a multiple of {BAND_ROWS * n} (64-row bands over {n} cores)")
    if spec.hidden % 256 or spec.intermediate % 256:
        raise OpRangeError("qwen3: hidden and intermediate must be multiples of 256 (one q4 k-tile)")
    if spec.num_kv_heads % 2:
        raise OpRangeError("qwen3: an odd kv-head count does not split into ain elements")
    require("ln", width=spec.hidden)
    require("attn", head_dim=spec.head_dim, num_heads=spec.num_heads, num_kv_heads=spec.num_kv_heads,
            rotary_dim=spec.rotary_dim, rope_theta=spec.rope_theta, qk_norm=spec.qk_norm, attn_gate=spec.attn_gate,
            qk_norm_post_rope=spec.qk_norm and spec.family in QKNORM_POST_ROPE, qkv_bias=qkv_bias(spec))
    pc = per_call(spec)
    require_gemv(spec, "attn", spec.hidden, spec.attn_q_width // n, pc)
    require_gemv(spec, "attn", spec.attn_q_width, spec.hidden // n, pc)
    require_gemv(spec, "ffn", spec.hidden, spec.intermediate // n, pc)
    require_gemv(spec, "ffn", spec.intermediate, spec.hidden // n, pc)
    require("lm_head_q4", K=spec.hidden, vocab=lm_rows(spec))


L1_BUDGET = 60 * 1024      # a main core's 64 KB data memory less IRON's own bookkeeping
STACK = 0x1800


def per_call(spec: ModelSpec) -> int:
    """Chunks per weight element: 2 (10 KB, as the MoE designs) unless the widest activation table
    leaves no room for two of them beside the x elements -- then 1 (Llama 3 8B: K = 14336 -> 32 KB)."""
    wide = max(spec.hidden, spec.attn_q_width, spec.intermediate)
    for pc in (2, 1):
        l1 = tab_bytes(wide) + 2 * pc * CHUNK + 2 * ELEM + 2 * BAND_ROWS * 4 + 2 * BAND_ROWS * 4 + STACK
        if l1 <= L1_BUDGET:
            return pc
    raise OpRangeError(f"dense: a {wide}-wide activation table does not leave room for the streams in a core's L1")


# ---- the fast attention path's knobs live in recipes/attnknobs.py (shared with the
# MoE / Qwen3.5 recipe, which compiles the same attn.h); re-exported here because the
# dense recipe is where they were first exposed and where cache.py and the tests look.
from .attnknobs import (PROBE_VARS, RB_SUPPORTED, FAST_ATTENTION, MAX_ATTN_CORES,  # noqa: E402,F401
                        probe_env, fast_attention, attn_cores, knobs as attn_knobs)


def geometry(spec: ModelSpec) -> DenseGeometry:
    n = cores_for(spec)
    hid, ff, nh, kvh, hd = spec.hidden, spec.intermediate, spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    e_a = kvw * 2
    _ptab_check(spec, e_a)
    hpe = e_a // (hd * 4)                     # q/k/v heads (f32) per ain element = KVH/2
    hpo = e_a // (hd * 2)                     # og heads (bf16) per aout element = KVH
    wide = max(hid, qw, ff)
    pc = per_call(spec)
    # The fast attention path (recipes/attnknobs.py): only for the families measured on
    # it; every other family's artifacts stay byte-identical until it has been.
    A = attn_knobs(spec, nh, hpo)
    vexp, acores, nhl, mls, rb = A.VEXP, A.ACORES, A.NHL, A.MLS, A.RB
    return DenseGeometry(
        N_CORES=n, HID=hid, FF=ff, NH=nh, KVH=kvh, HD=hd, ROT=spec.rotary_dim, GATE=spec.attn_gate,
        QKNORM=spec.qk_norm, QKNORM_POST=spec.qk_norm and spec.family in QKNORM_POST_ROPE,
        QKVB=qkv_bias(spec),
        PTAB_ELEMS=_ptab_row(spec) // e_a, PTAB_CS_ELEM=512 // e_a,
        EPS=spec.norm_eps, ACT=spec.activation, SANDWICH=spec.sandwich_norms,
        WINDOW=spec.sliding_window if spec.has_local else 0, PER_CALL=pc, CALL_BYTES=pc * CHUNK,
        QW=qw, KVW=kvw,
        Q_PC=qw // BAND_ROWS // n, KV_PC=kvw // BAND_ROWS // n, O_PC=hid // BAND_ROWS // n,
        UP_PC=ff // BAND_ROWS // n, DOWN_PC=hid // BAND_ROWS // n,
        XN_ELEMS=roundup(hid * 2, ELEM) // ELEM, XN_BLOCKS=hid // 32,
        OG_ELEMS=roundup(qw * 2, ELEM) // ELEM, OG_BLOCKS=qw // 32,
        H_ELEMS=roundup(ff * 4, ELEM) // ELEM, H_BLOCKS=ff // 32,
        XM_ELEMS=roundup(hid * 2, ELEM) // ELEM,
        HPE=hpe, HPO=hpo, Q_AIN_ELEMS=nh // hpe, K_AIN_ELEMS=kvh // hpe, OG_AOUT_ELEMS=nh // hpo,
        TAB_BYTES=tab_bytes(wide), KWIDE=wide,
        MS_U=0, MS_G=BAND_ROWS, MS_FLOATS=2 * BAND_ROWS,
        VEXP=vexp, MLS=mls, ACORES=acores, NHL=nhl, RB=rb,
    )


def layout(spec: ModelSpec, max_ctx: int = 4096) -> DenseLayout:
    n = cores_for(spec)
    hid, ff = spec.hidden, spec.intermediate
    G = geometry(spec)
    eln, e_a = hid * 2, G.KVW * 2
    if 2 * G.HD * 2 > e_a or 512 + 4 * spec.rotary_dim > _ptab_row(spec):
        raise OpRangeError("dense: qn | kn or the RoPE record do not fit the attention element")
    _ptab_check(spec, e_a)
    # consts
    c = {"lnw": 0, "postln": eln, "meta": 2 * eln, "preffn": 2 * eln + e_a, "postffn": 3 * eln + e_a}
    off = (4 * eln + e_a) if spec.sandwich_norms else (2 * eln + e_a)
    # The bias vectors stream into the attention core beside the projections they belong to,
    # one bias element per q / k / v element. A projection element carries KVH/2 heads as
    # f32 (E_A bytes); the same heads of a bf16 bias are E_A/2, so the two streams run in
    # lockstep for any geometry: QW*2 / (E_A/2) = 2*NH/KVH = Q_AIN_ELEMS, KVW*2 / (E_A/2) = 2.
    if qkv_bias(spec):
        c["qb"], c["kb"], c["vb"] = off, off + G.QW * 2, off + (G.QW + G.KVW) * 2
        off += (G.QW + 2 * G.KVW) * 2
    else:
        c["qb"] = c["kb"] = c["vb"] = -1
    cd_bytes = roundup(off, ELEM)
    # act
    a: dict[str, int] = {}
    off = 0
    for name, size in (("xn", G.XN_ELEMS * ELEM), ("q", G.QW * 4), ("kvn", 2 * G.KVW * 4), ("og", G.QW * 2),
                       ("out", hid * 4), ("res", hid * 4), ("xm", G.XM_ELEMS * ELEM), ("h", G.H_ELEMS * ELEM),
                       ("out2", hid * 4), ("junk", eln), ("t", hid * 4), ("t2", hid * 4)):
        a[name] = off
        off += size
    ad_bytes = roundup(off, ELEM)
    # pool
    p: dict[str, int] = {}
    off = 0
    for name, role, rows, cols in (("q", "attn", G.QW, hid), ("k", "attn", G.KVW, hid),
                                   ("v", "attn", G.KVW, hid), ("o", "attn", hid, G.QW),
                                   ("up", "ffn", ff, hid), ("gate", "ffn", ff, hid),
                                   ("down", "ffn", hid, ff)):
        p[name] = off
        off += role_bytes(spec, role, rows, cols)
    pool_bytes = roundup(off, MB)
    kv_row = 2 * e_a
    ptab_row = _ptab_row(spec)
    band = band_bytes(hid)
    bands = lm_rows(spec) // BAND_ROWS
    return DenseLayout(
        CD_LNW=c["lnw"], CD_POSTLN=c["postln"], CD_META=c["meta"], CD_PREFFN=c["preffn"], CD_POSTFFN=c["postffn"],
        CD_QB=c["qb"], CD_KB=c["kb"], CD_VB=c["vb"], CD_BYTES=cd_bytes,
        AD_XN=a["xn"], AD_Q=a["q"], AD_KVN=a["kvn"], AD_OG=a["og"], AD_OUT=a["out"], AD_RES=a["res"], AD_XM=a["xm"],
        AD_H=a["h"], AD_OUT2=a["out2"], AD_JUNK=a["junk"], AD_T=a["t"], AD_T2=a["t2"], AD_BYTES=ad_bytes,
        POOL_Q=p["q"], POOL_K=p["k"], POOL_V=p["v"], POOL_O=p["o"], POOL_UP=p["up"], POOL_GATE=p["gate"],
        POOL_DOWN=p["down"], POOL_BYTES=pool_bytes,
        KV_ROW=kv_row, PTAB_ROW=ptab_row, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * ptab_row,
        LMHEAD_POOL_BYTES=roundup(roundup(bands, n) * band, MB), LMHEAD_BANDS=bands, LMHEAD_BAND_BYTES=band,
        ELN=eln, E_A=e_a,
    )


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> DenseRecipe:
    _check(spec)
    return DenseRecipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    hid, ff = spec.hidden, spec.intermediate
    pre = "model.layers.{l}."
    one = {
            "pool": [
                proj_op(spec, "attn", pre + "self_attn.q_proj.weight", L.POOL_Q, G.QW, hid, hid),
                proj_op(spec, "attn", pre + "self_attn.k_proj.weight", L.POOL_K, G.KVW, hid, hid),
                proj_op(spec, "attn", pre + "self_attn.v_proj.weight", L.POOL_V, G.KVW, hid, hid),
                proj_op(spec, "attn", pre + "self_attn.o_proj.weight", L.POOL_O, hid, G.QW, G.QW),
                proj_op(spec, "ffn", pre + "mlp.up_proj.weight", L.POOL_UP, ff, hid, hid),
                proj_op(spec, "ffn", pre + "mlp.gate_proj.weight", L.POOL_GATE, ff, hid, hid),
                proj_op(spec, "ffn", pre + "mlp.down_proj.weight", L.POOL_DOWN, hid, ff, ff),
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CD_LNW, "cap": L.ELN},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CD_POSTLN, "cap": L.ELN},
            ] + ([
                {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CD_META, "cap": G.HD * 2},
                {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CD_META + G.HD * 2, "cap": G.HD * 2},
            ] if spec.qk_norm else []) + ([
                {"op": "put", "tensor": pre + "self_attn.q_proj.bias", "dst": L.CD_QB, "cap": G.QW * 2},
                {"op": "put", "tensor": pre + "self_attn.k_proj.bias", "dst": L.CD_KB, "cap": G.KVW * 2},
                {"op": "put", "tensor": pre + "self_attn.v_proj.bias", "dst": L.CD_VB, "cap": G.KVW * 2},
            ] if G.QKVB else []) + ([
                {"op": "put", "tensor": pre + "pre_feedforward_layernorm.weight", "dst": L.CD_PREFFN, "cap": L.ELN},
                {"op": "put", "tensor": pre + "post_feedforward_layernorm.weight", "dst": L.CD_POSTFFN, "cap": L.ELN},
            ] if spec.sandwich_norms else []),
    }
    return {
        "pool_bytes": L.POOL_BYTES, "chunk_bytes": CHUNK,
        "layer_types": {lt: one for lt in sorted(set(spec.layer_types))},
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "std_perm", "tensor": "lm_head.weight", "dst": 0,
                             "nch": q4_chunks(lm_rows(spec), hid), "in_dim": hid}]},
        "embed": {"tensor": "model.embed_tokens.weight", "dim": hid},
        "norm": {"tensor": "model.norm.weight", "bytes": hid * 2},
    }


# ---- the block prefill route (OPEN-PREFILL-BATCH, kind "dense") ---------------------
def gemm_route(spec: ModelSpec, max_ctx: int = 4096) -> dict | None:
    """Per layer type the `gemm_block` the engine reads -- kind "dense": the five-step chain
    (qkv3, o, gate, up, down) with T single-token attention dispatches between the first two
    -- plus the contexts / kernels / globals / builds the route adds. A layer type with its own
    sliding window (Gemma 3's `dense_local`) gets its own attention kernel and position table,
    named in its `gemm_block` (`attn_kernel`, `attn_args`, mirroring `programs()`'s dx/dx_local
    split below); the five GEMM steps, the weight map, the widths and the shared `gact` scratch
    and GEMM contexts are identical across layer types, since `layout()`/`geometry()` don't vary
    by layer type. None means the sequential path, byte for byte what it was, for one of the
    reasons below.

    The block size, the GEMM design and the one-context-per-route rule are the 35B route's
    (`qwen36moe.gemm_route`); what differs is the kind, and that the FFN here is dense rather
    than a set of experts, so the whole layer is five dispatches and a handful of host stages."""
    # A projection streamed at q8 has no route: the GEMM dequantises the q4_1 band law only
    # (OPEN-PREFILL-BATCH), so such a spec keeps exactly the sequential manifest it had.
    if any(spec.quant_of(r) == "q8" for r in Q8_ROLES):
        return None
    # This route only knows the dense layer shape (`dense`, and Gemma 3's sliding-window
    # `dense_local`) -- a hybrid family's short_conv / full_attention / linear_attention layer
    # types (LFM2, the 35B) get no route here; their own recipe, where they have one, covers them.
    if not set(spec.layer_types) <= {DENSE, DENSE_LOCAL}:
        return None
    # The block route's host chain (core.cpp step_gemm_block_layer) knows exactly two
    # (activation, residual/norm chain) pairings: plain SiLU (every family without sandwich
    # norms) and Gemma 3's GeGLU-tanh + sandwich norms. A family combining sandwich norms with
    # a different activation, or vice versa, has no route yet -- fail closed rather than emit
    # one the host chain would compute wrong.
    if spec.sandwich_norms != (spec.activation == "gelu_tanh"):
        return None
    L, G, T = layout(spec, max_ctx), geometry(spec), GEMM_T
    # The attention dispatch (designs/dense/dx_attn.py) has no q/k/v bias stream and acquires
    # a one-element position record, and refuses at build time otherwise -- so Qwen2 (both)
    # keeps the sequential route rather than failing its export on dx_attn.
    if G.QKVB or G.PTAB_ELEMS > 1:
        return None
    hid, ff, qw, kvw = spec.hidden, spec.intermediate, G.QW, G.KVW
    plans = pack_plan(spec)["layer_types"]
    shapes: set[tuple[int, int]] = set()

    def ctx(N: int, K: int) -> str:
        # gemm_q4_prefill tiles N into 256-row blocks (4 rows x 64-row bands) and reads K as
        # 256-wide band groups, so both dims are multiples of 256 or the design will not build.
        if N % 256 or K % 256:
            raise OpRangeError(f"gemm route: [{N}, {K}] is not a multiple of 256 in both dims")
        shapes.add((N, K))
        return f"gemm_n{N}_k{K}"

    def run(N: int, K: int, w: str) -> dict:
        return {"op": "run", "kernel": ctx(N, K), "args": [w, f"gemm_x_k{K}", f"gemm_y_n{N}"]}

    # The five steps in the FIXED order the engine reads them (manifest.hpp's
    # GemmBlockProgram): qkv3, o, gate, up, down. q, k and v are three consecutive pack ops
    # at one pool offset, so the fused input projection is ONE dispatch over their bytes;
    # each FFN projection is one op. Every `ops` list must be byte-contiguous in the pool --
    # the engine builds each weight buffer with a single memcpy and says so if it is not.
    # (Identical for every layer type: pack_plan's pool op order doesn't vary by layer type.)
    program = [run(qw + 2 * kvw, hid, "gqkv3_w"), run(hid, qw, "go_w"),
               run(ff, hid, "ggate_w"), run(ff, hid, "gup_w"), run(hid, ff, "gdown_w")]

    qh = spec.quant_hash()
    sfx = f"_q{qh}" if qh else ""          # a q8 variant is a different kernel set (OPEN-QUANT-Q8)
    out: dict = {"layer_types": {}, "contexts": {"dxa": "dx_attn/final.xclbin"}, "kernels": {}, "globals": {},
                "builds": {"dx_attn": {"design": "dense/dx_attn.py",
                                       "build_dir": f"dense/build_dxa_{spec.family}_h{hid}{sfx}", "env": {}}}}
    for lt in sorted(set(spec.layer_types)):
        local = lt == DENSE_LOCAL
        pool = plans[lt]["pool"]
        weights = {
            "gqkv3_w": {"from": "pool", "ops": [_op_index(pool, "self_attn.q_proj.weight"),
                                                _op_index(pool, "self_attn.k_proj.weight"),
                                                _op_index(pool, "self_attn.v_proj.weight")]},
            "go_w": {"from": "pool", "ops": [_op_index(pool, "self_attn.o_proj.weight")]},
            "ggate_w": {"from": "pool", "ops": [_op_index(pool, "mlp.gate_proj.weight")]},
            "gup_w": {"from": "pool", "ops": [_op_index(pool, "mlp.up_proj.weight")]},
            "gdown_w": {"from": "pool", "ops": [_op_index(pool, "mlp.down_proj.weight")]},
        }
        # The attention half of the layer as its own dispatch (designs/dense/dx_attn.py: dx.py's
        # single-token attention phase and nothing else, same kernels and same placeholder
        # offsets), so the SAME attnpos patch the sequential layer uses drives it once per token
        # of the block. `pool` and `xres` are dummy arguments there -- the engine passes the
        # layer's six in the sequential order and the design ignores the two it does not read.
        # A layer type with a sliding window gets its OWN kernel entry -- same instruction
        # stream, its own window patch -- and its own table, exactly as the sequential path's
        # dx/dx_local split above.
        kn = "dxB_local" if local else "dxB"
        tab = "ptab_local" if local else "ptab"
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        out["kernels"][kn] = {"context": "dxa", "insts": "dx_attn/insts.bin", "patch": "attnpos",
                              "build": "dx_attn", "window": spec.sliding_window if local else 0}
        out["layer_types"][lt] = {"kind": "dense", "t": T, "eps": spec.norm_eps,
                                  "program": program, "weights": weights,
                                  "qw": qw, "kvw": kvw, "ff": ff,
                                  "ad_q": L.AD_Q, "ad_kvn": L.AD_KVN, "ad_og": L.AD_OG,
                                  "attn_kernel": kn, "attn_args": args,
                                  "sandwich": spec.sandwich_norms, "act": spec.activation}
    # The T-wide attention scratch: the engine writes every token's q/k/v into it, then
    # shuttles one token's slice in and out of the layer's own `act` around each attention
    # dispatch (core.cpp `shuttle_buf`), and reads all T og rows back out at the end. Shared
    # across layer types -- AD_BYTES doesn't vary by layer type.
    out["globals"]["gact"] = T * L.AD_BYTES
    # ONE hardware context for every projection shape: the GEMM core program depends on
    # neither N nor K (the band count K/256 reaches each core as a runtime parameter), so
    # each shape is an instruction stream over one xclbin. Changing context costs ~2.5 ms
    # and the shapes alternate, so this is worth more here than anywhere else in the route.
    for N, K in sorted(shapes):
        name = f"gemm_n{N}_k{K}"
        out["contexts"].setdefault("gemm", f"{name}/final.xclbin")
        out["kernels"][name] = {"context": "gemm", "insts": f"{name}/insts.bin", "build": name}
        out["globals"][f"gemm_x_k{K}"] = K * T * 2      # bf16, pre-tiled [K, T]
        out["globals"][f"gemm_y_n{N}"] = N * T * 4      # f32 [N, T]
        out["builds"][name] = {"design": "gemm_q4_prefill/gemm_q4_prefill.py",
                               "build_dir": f"gemm_q4_prefill/build_n{N}_k{K}_t{T}",
                               "env": {"GQP_N": str(N), "GQP_K": str(K), "GQP_T": str(T)}}
    return out


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    """One design serves every dense layer type; a layer type with a sliding window gets its own
    kernel entry (the same instruction stream, its own instruction BO, patched with its window)
    and its own position table (its RoPE frequencies, its window's row counts).

    Phi-3's longrope carries TWO factor lists, chosen by HF per forward call from the running
    sequence length -- a resident table cannot re-select per call, but this engine computes one
    row per token as the context grows (`Core::step`), so the ROW is the right unit to switch
    on: row r took the table a real forward call at sequence length r + 1 would have picked, and
    that decision never moves once made (an earlier row's cached K does not get rewound). Both
    tables are baked (`inv_freq` for rows below `original_max_position_embeddings`,
    `long_inv_freq` at and above it -- `pools.cpp` / `pack.ptab` switch per row against
    `switch_row`) and `max_ctx` plays no part in the choice; every other family's global carries
    neither key, unchanged. The attention scale rides on the table as `scale` for any family that
    has one (Phi-3 only, today)."""
    L, G = layout(spec), geometry(spec)
    sc = spec.rope_scaling
    longrope = bool(sc) and sc.get("rope_type") == "longrope"
    out = {
        "contexts": {"dx": "dx/final.xclbin", "ln": "ln/final.xclbin", "lm": "lm_head_q4/final.xclbin"},
        "kernels": {"ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
                    "lm": {"context": "lm", "insts": "lm_head_q4/insts.bin", "build": "lm_head_q4"}},
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": lm_rows(spec) * 4,
                    "lmpool": L.LMHEAD_POOL_BYTES},
    }
    types = sorted(set(spec.layer_types))
    for lt in types:
        local = lt == DENSE_LOCAL
        kn = "dx_local" if local else "dx"
        tab = "ptab_local" if local else "ptab"
        window = spec.sliding_window if local else 0
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        out["kernels"][kn] = {"context": "dx", "insts": "dx/insts.bin", "patch": "attnpos", "build": "dx", "window": window}
        out["globals"][tab] = {"per_row": L.PTAB_ROW, "inv_freq": spec.rope_inv_freq(local=local), "window": window}
        if longrope:
            orig = int(sc["original_max_position_embeddings"])
            out["globals"][tab]["long_inv_freq"] = spec.rope_inv_freq(local=local, ctx=orig + 1)
            out["globals"][tab]["switch_row"] = orig
        if spec.rope_scale() != 1.0:
            out["globals"][tab]["scale"] = spec.rope_scale()
        out["layer_types"][lt] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": kn, "args": args}],
        }
    r = gemm_route(spec, max_ctx)
    if r:
        for k in ("contexts", "kernels", "globals"):
            out[k].update(r[k])
        for lt, gb in r["layer_types"].items():
            out["layer_types"][lt]["gemm_block"] = gb
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    n = cores_for(spec)
    qh = spec.quant_hash()
    sfx = f"_q{qh}" if qh else ""          # a q8 variant is a different kernel set (OPEN-QUANT-Q8)
    b = {
        "dx": {"design": "dense/dx.py", "build_dir": f"dense/build_{spec.family}_h{spec.hidden}{sfx}", "env": {}},
        "ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{spec.hidden}_{spec.norm_eps:g}", "env": {"LN_N": str(spec.hidden), "LN_EPS": f"{spec.norm_eps:g}"}},
        "lm_head_q4": {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{lm_rows(spec)}",
                       "env": {"LMHEAD_N": str(lm_rows(spec)), "LMHEAD_K": str(spec.hidden), "LMHEAD_CORES": str(n)}},
    }
    r = gemm_route(spec)
    if r:
        b.update(r["builds"])
    return b


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L = layout(spec, max_ctx)
    return {"hidden": spec.hidden, "vocab": lm_rows(spec), "real_vocab": spec.real_vocab,
            "chunk_bytes": CHUNK, "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": spec.rope_inv_freq()}     # the global table's short/base table; each ptab
            # global carries its own (both tables, for a longrope family -- see programs())


def _phi3_raw_scaling(spec: ModelSpec):
    """What a Phi-3 container's config.json literally holds at `rope_scaling`: the verbatim
    sub-object the derivation kept (`extra.rope_scaling_raw`), or -- for a spec loaded from
    JSON without it -- the HF-standard shape rebuilt from the canonical fields. None for a
    plain-RoPE Phi."""
    if spec.rope_scaling is None:
        return None
    raw = spec.extra.get("rope_scaling_raw")
    if raw is not None:
        return raw
    return {"type": "longrope", "short_factor": list(spec.rope_scaling["short_factor"]),
            "long_factor": list(spec.rope_scaling["long_factor"])}


def hf_config_defaults(spec: ModelSpec) -> dict:
    """What a key ABSENT from a container's config.json means, for the keys
    `hf_config_check` emits that HF lets a config omit. `Manifest::check_model` compares
    the manifest's expected value against this instead of refusing the config for
    lacking the key -- so an omitted optional field is accepted exactly when it implies
    the value the kernel set was built for, and refused when it does not."""
    if spec.family != "phi3":
        return {}
    raw = _phi3_raw_scaling(spec) or {}
    return {"head_dim": spec.hidden // spec.num_heads,          # HF's own fallback
            "partial_rotary_factor": 1.0,                        # a full rotation
            "rope_scaling": None,                                # plain RoPE
            # the derivation reads it from the sub-object when the top level lacks it
            "original_max_position_embeddings": raw.get("original_max_position_embeddings")}


def hf_config_check(spec: ModelSpec) -> dict:
    d = {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
         "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
         "intermediate_size": spec.intermediate}
    if spec.family in ("qwen3", "gemma3", "hunyuan", "granite", "phi3"):
        d["head_dim"] = spec.head_dim          # Llama configs may omit it (hidden / heads)
    if spec.family == "phi3":
        # The rotation width is compiled into the attention core (ATTN_ROT), and the
        # position table's frequencies (rope_theta, longrope's factor lists, the attention
        # scale that max_position_embeddings sets) are baked in at export time -- none of
        # it derivable from the shape fields above, so a same-shaped container with
        # different RoPE parameters would otherwise load silently and run with the wrong
        # table. Every key is emitted, whether or not the source config spelled it out;
        # what an ABSENT key means comes from hf_config_defaults below, so the check is
        # two-way (a kernel set built from a full-rotation config refuses a 0.75 container
        # and vice versa) without refusing a config for omitting an optional field.
        d["partial_rotary_factor"] = spec.rotary_dim / spec.head_dim
        d["rope_theta"] = spec.rope_theta
        d["rope_scaling"] = _phi3_raw_scaling(spec)
        if spec.rope_scaling is not None:
            orig = spec.rope_scaling["original_max_position_embeddings"]
            d["original_max_position_embeddings"] = orig
            if d["rope_scaling"].get("factor") is None:
                # the factor, and with it rope_scale(), came from max_position_embeddings
                d["max_position_embeddings"] = int(round(spec.rope_scaling["factor"] * orig))
    if spec.family == "gemma3":
        d["sliding_window"] = spec.sliding_window
    if spec.family == "granite":
        # The engine refuses an UNFOLDED container at load, not just at spec
        # derivation: Granite's attention_multiplier replaces 1/sqrt(HD), and
        # attn.h hard-codes the latter. A folded config reads head_dim**-0.5
        # exactly (0.125 at hd 64, a power of two). Swapping an unfolded
        # model.q4nx under a built kernel set would otherwise run and return
        # plausible garbage. See spec.py's _granite_scale_check.
        d["attention_multiplier"] = spec.head_dim ** -0.5
    return d


GEN_KERNELS = "designs/dense/gen_kernels.py"      # the design's kernel-TU generator (export_qwen36_kernels.py runs it per spec)
KERNEL_SOURCES = [
    "designs/dense/*.py", "designs/dense/*.h",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h", "designs/gemv_q4/gemv_q4_prep_rt.cc",
    "designs/gemv_q4/gemv_q4_prep_f32_rt.cc",
    "designs/attn/*.cc", "designs/attn/*.h",
    "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py", "designs/lin_layer/ln_nr.cc",
    "designs/lm_head_q4/*.py",
    # the block prefill route's GEMM (gemm_route); dx_attn.py is already in designs/dense/*.py
    "designs/gemm_q4_prefill/*.py", "designs/gemm_q4_prefill/*.cc", "designs/gemm_q4_prefill/*.h",
    "include/vecmath.h", "include/scalar_fp.h", "ironutil.py", "build_design.py",
]
KERNEL_SOURCES_Q8 = ["designs/gemv_q4/gemv_q8.h"]
Q8_ROLES = frozenset({"attn", "ffn"})     # the only two roles a dense layer has
