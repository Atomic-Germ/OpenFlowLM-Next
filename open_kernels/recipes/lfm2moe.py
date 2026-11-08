"""LFM2-MoE: LFM2's short-conv / attention hybrid with the routed experts on every
layer past `dense_lead`.

Every layer past the leading dense ones is the Qwen3.6-MoE composition: the attention
layers run ax.py (FULL, MoE), the short-conv layers run sx.py (SHORT_CONV, MoE), and
the router's record is the same one (xcommon.moe_body) with the softmax replaced by
this family's sigmoid + expert-bias selection. The leading `dense_lead` layers carry
the dense silu FFN instead of the experts, on the decoder's own mixer: SHORT_CONV_DENSE
is lfm2.py's SHORT_CONV (cx.py, dense FFN), FULL_DENSE is the dense family's DENSE
(dx.py). They are just layer types, so the pack plan, the programs and the driver all
dispatch on them exactly as they do on the two mixers -- no per-index FFN snobbery.

The expert stripe law is `qwen36moe`'s for ff <= 512 (STRIPES_PER_PROJ dividing the
core count), and its wide-FF twin past that: the 24B's 1536 is 12 stripes over 8 cores
and the 8B's 1792 is 14 over 8, so one 128-row stripe can no longer split evenly. The
law below keeps one 64-row half per unit (HALF) and gives each core HALVES_PER_CORE
whole halves (ff // (64 * n)), which is uniform exactly when ff is a multiple of
512. A width that is not is padded up to that multiple in the pool -- the padded rows
pack as zero-weight chunks and the down GEMM's padded K reads them as zero -- so any
fine-tune resolves without a per-model choice. The core's `ms` holds HID_PC rows for
either case, its table holds both the hidden and the expert hidden, and the pool chunk
order is the same `expert_stripes[_gguf]` order, so the GEMV points and the pool
chunk law are shared with the small law and only the host stripes move.

The router: LFM2-MoE scores are sigmoid, `use_expert_bias` selects on the biased
score while the weights renormalise the unbiased scores over the top-k, times
`routed_scaling_factor`, with `norm_topk_prob` renormalisation (transformers'
Lfm2MoeTopKRouter, sigmoid / unbiased renormalisation / scale). The design's router
keeps the arithmetic on the NPU: designs/router/router.h gains the knobs (E, topk,
rows-per-element, sigmoid, bias, unbiased renormalisation, scale) the way attn.h
already takes ATTN_GATE, and the record is [p f32[E] | idx | w] with p over the
selection scores and w over the renormalised weights. The host `Core::route` only
reads idx and applies moe2_apply, so it does not change.
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalogue import OpRangeError, check_buffer_args, require
from . import qwen36moe as Q
from .spec import FULL, FULL_DENSE, SHORT_CONV, SHORT_CONV_DENSE, ModelSpec

# Re-export what the built designs read off the family module (dx.py reads band_bytes/
# chunk_bytes off the dense module; the MoE designs read them off qwen36moe). lfm2moe
# borrows the MoE tail from qwen36moe and the short-conv mixer from lfm2, so it offers
# both.
band_bytes = Q.band_bytes  # noqa: F401  (layer_x gen_kernels: band_bytes(K, quant))
chunk_bytes = Q.chunk_bytes  # noqa: F401  (layer_x gen_kernels: chunk_bytes(quant))

# LFM2-MoE geometry
TAPS = 3
N_CORES = Q.LIMITS["n_cols"]
HALF_WIDE = 512        # ff is a multiple of this and one core takes ff // (64 * n) whole halves


def pad_ff(ff: int) -> int:
    """The expert width the pool lays out: ff itself when one core takes whole 64-row
    halves, else ff rounded up to the next multiple of 512 (the padded rows pack as
    zero chunks)."""
    if ff % HALF_WIDE == 0:
        return ff
    return Q.roundup(ff, HALF_WIDE)


def padded(spec: ModelSpec) -> int:
    return pad_ff(spec.moe_intermediate)


def halves_per_core(spec: ModelSpec) -> int:
    """Whole 64-row halves of expert width per core: padded ff / (64 * n)."""
    return padded(spec) // (64 * N_CORES)


def _check(spec: ModelSpec) -> None:
    if spec.family != "lfm2moe":
        raise OpRangeError(f"lfm2moe recipe on a {spec.family!r} spec")
    from .catalogue import LIMITS
    n = LIMITS["n_cols"]
    if spec.has_linear:
        raise OpRangeError("lfm2moe: no DeltaNet layer in layer_types; this is the short-conv / attention hybrid")
    allowed = {SHORT_CONV, SHORT_CONV_DENSE, FULL, FULL_DENSE}
    bad = sorted(set(spec.layer_types) - allowed)
    if bad:
        raise OpRangeError(f"lfm2moe: layer types {bad} are not short-conv / attention (MoE or dense)")
    if not (SHORT_CONV in spec.layer_types or FULL in spec.layer_types):
        raise OpRangeError("lfm2moe: no MoE layer in layer_types; this is the dense recipe's model")
    if not (SHORT_CONV_DENSE in spec.layer_types or spec.dense_lead == 0):
        # dense layers are present but not in the mixer list shape we accept
        raise OpRangeError("lfm2moe: dense layers must be a prefix (SHORT_CONV_DENSE / FULL_DENSE)")
    if spec.conv_kernel != TAPS:
        raise OpRangeError(f"lfm2moe: conv_kernel {spec.conv_kernel} is not {TAPS}")
    if spec.activation != "silu":
        raise OpRangeError(f"lfm2moe: activation {spec.activation!r}, not silu")
    if spec.attn_gate or spec.has_linear or spec.has_dense:
        raise OpRangeError("lfm2moe: expected a short-conv + full-attention hybrid with no attention gate")
    if spec.shared_expert_intermediate:
        raise OpRangeError("lfm2moe: a shared expert is not in this family's block")
    if spec.num_experts == 0 or spec.moe_intermediate == 0 or (spec.dense_lead > 0 and spec.intermediate == 0):
        raise OpRangeError("lfm2moe: expected routed experts and (when dense_lead > 0) a dense FFN width")
    # the dense_lead prefix
    for i in range(spec.dense_lead):
        if spec.layer_types[i] not in (SHORT_CONV_DENSE, FULL_DENSE):
            raise OpRangeError(f"lfm2moe: layer {i} is before dense_lead={spec.dense_lead} but is not dense")
    for i in range(spec.dense_lead, spec.num_layers):
        if spec.layer_types[i] in (SHORT_CONV_DENSE, FULL_DENSE):
            raise OpRangeError(f"lfm2moe: layer {i} is at or past dense_lead={spec.dense_lead} but is dense")
    Q.quant_check(spec, "lfm2moe")
    require("router", experts=spec.num_experts, topk=spec.experts_per_tok)
    require("moe", ff=padded(spec), experts=spec.num_experts, topk=spec.experts_per_tok, shared_expert=False, hidden=spec.hidden, n_cores=n)
    require("gemv_q4_prep_f32", K=padded(spec))
    require("lm_head_q8", K=spec.hidden, vocab=spec.vocab)
    Q.mixed_check(spec, "lfm2moe", ("attn", "ffn", "linear", "linear_out"))
    for what, v in (("hidden", spec.hidden), ("moe_intermediate", padded(spec)), ("q width", spec.attn_q_width),
                    ("kv width", spec.attn_kv_width), ("intermediate", spec.intermediate if spec.dense_lead else spec.hidden)):
        if v % (Q.BAND_ROWS * n):
            raise OpRangeError(f"lfm2moe: {what} {v} is not a multiple of {Q.BAND_ROWS * n}")
    if spec.hidden % 256:
        raise OpRangeError("lfm2moe: hidden must be a multiple of 256 (one q4 k-tile)")
    if spec.num_kv_heads % 2:
        raise OpRangeError("lfm2moe: an odd kv-head count does not split into ain elements")
    require("ln", width=spec.hidden)
    require("attn", head_dim=spec.head_dim, num_heads=spec.num_heads, num_kv_heads=spec.num_kv_heads,
            rotary_dim=spec.rotary_dim, rope_theta=spec.rope_theta, qk_norm=spec.qk_norm,
            attn_gate=spec.attn_gate, qk_norm_post_rope=False, qkv_bias=False)
    require("short_conv", taps=TAPS, width=spec.hidden)
    pc = Q.per_call(spec, "moe")
    Q.require_gemv(spec, "attn", spec.hidden, spec.attn_q_width // n, pc)
    Q.require_gemv(spec, "attn", spec.attn_q_width, spec.hidden // n, pc)
    if spec.dense_lead:
        Q.require_gemv(spec, "ffn", spec.hidden, spec.intermediate // n, Q.per_call(spec, "dense"))
        Q.require_gemv(spec, "ffn", spec.intermediate, spec.hidden // n, Q.per_call(spec, "dense"))


# ---- main-core geometry: the wide twin of qwen36moe.common() for the MoE tail
def common(spec: ModelSpec) -> Q.Common:
    """The MoE common geometry with the wide stripe law: padded ff, HALVES_PER_CORE
    whole 64-row halves per core, and a table that holds both the hidden and the
    expert hidden (the no-`wide` collapse in qwen36moe.common())."""
    _check(spec)
    from .catalogue import LIMITS as CAT_LIMITS
    n = CAT_LIMITS["n_cols"]
    hid = spec.hidden
    pff = padded(spec)
    pq = Q.pool_quant(spec)
    chunk = Q.chunk_bytes(pq)
    stripe = Q.q4_bytes(128, hid, pq)
    half = Q.q4_bytes(Q.BAND_ROWS, hid, pq)
    down_band = Q.q4_bytes(128, pff, pq)
    up_bytes = Q.q4_bytes(pff, hid, pq)
    rows_pc = hid // n
    hid_pc = pff // n
    ms_rw, ms_xr = 0, 32
    ms_acc = ms_xr + rows_pc
    ms_u = ms_acc + rows_pc
    ms_g = ms_u + hid_pc
    ms_yd = ms_g + hid_pc
    ms_floats = ms_yd + rows_pc
    hpc = halves_per_core(spec)
    if spec.experts_per_tok + 8 > 32:
        raise OpRangeError(f"moe: top-k {spec.experts_per_tok} does not fit the 32-float routing record")
    spp = pff // 128
    wide = max(hid, spec.attn_q_width if spec.has_full else 0)
    h_tab = Q.tab_bytes(hid)
    tab = max(Q.tab_bytes(wide), h_tab + Q.tab_bytes(pff))
    # No shared expert: NX is NE + 1 in qwen36moe.Common (the shared slot is last);
    # the design closes on xres alone via accfin(-1), the mx.py path.
    l1 = Q.core_l1(tab, ms_floats, Q.DN_SCRATCH_FLOATS)
    if l1 > Q.L1_BUDGET:
        raise OpRangeError(
            f"moe: a main core's scratch comes to {l1} B against a {Q.L1_BUDGET} B budget, "
            f"{tab} B of table, {ms_floats * 4} B of ms, {Q.DN_SCRATCH_FLOATS * 4} B of ds.")
    return Q.Common(
        N_CORES=n, HID=hid, FF=pff, NE=spec.experts_per_tok, NX=spec.experts_per_tok,
        TILE=chunk, PER_CALL=Q.PER_CALL, CALL_BYTES=Q.PER_CALL * chunk,
        STRIPE=stripe, HALF=half, PAIR=2 * chunk, DOWN_BAND=down_band, UP_BYTES=up_bytes,
        DOWN_PER_CORE=rows_pc // 128,
        BAND_ROWS=Q.BAND_ROWS, BAND16=Q.band_bytes(hid, pq), BAND32=Q.band_bytes(2 * hid, pq), N_HDR=3,
        MS_FLOATS=ms_floats, DS_FLOATS=1280, TAB_BYTES=tab, H_TAB_OFF=h_tab, KWIDE=wide,
        MS_RW=ms_rw, MS_XR=ms_xr, MS_ACC=ms_acc, MS_U=ms_u, MS_G=ms_g, MS_YD=ms_yd,
        ROWS_PC=rows_pc, HID_PC=hid_pc,
        STRIPES_PER_PROJ=spp, CORES_PER_STRIPE=0,
        UP_ELEMS=half // (Q.PER_CALL * chunk), DOWN_ELEMS=(rows_pc // 128) * down_band // (Q.PER_CALL * chunk),
        OUT_ELEMS=rows_pc // Q.BAND_ROWS,
        W_ELEMS=hid * spec.num_experts * 2 // Q.ELEM,
        DN_ROWS=0, DN_SLICES=0, DN_HEADS_PC=0, DN_DIM=0, DN_PAD=0, HALVES_PER_CORE=hpc,
    )


@dataclass(frozen=True)
class Lfm2moeLayout:
    """The MoE layout plus the short-conv and dense-FFN layers' pieces.

    `moe` is the qwen36moe attention + MoE layout with the gate dropped in the
    wrapper below; `dense` is the dense family's layout for the leading dense layers,
    shrunk to the sizes the attention / conv mixes need. The two are merged here the
    way lfm2.Lfm2Layout merges its dense layout: one pool, one consts, one act
    sized to the largest of the layer types, with each type's own offsets kept.
    """
    moe: Q.Layout
    dense: object
    POOL_SC_B: int; POOL_SC_C: int; POOL_SC_U: int; POOL_SC_OUT: int
    POOL_SC_UP: int; POOL_SC_GATE: int; POOL_SC_DOWN: int
    POOL_BYTES: int
    CD_CONV: int; CD_BYTES: int
    AD_B: int; AD_C: int; AD_U: int; AD_Y: int; AD_BYTES: int
    STATE_BYTES: int
    # router bias: exp_probs_b.bias f32[E] per layer, packed as a put into this element
    BIAS_BYTES: int
    ATTN_GATE_BYTES: int = 0

    def __getattr__(self, name: str):
        try:
            return getattr(object.__getattribute__(self, "moe"), name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no {name!r}") from None
    def constants(self) -> dict[str, int]:
        out = dict(self.moe.__dict__)
        out.update({k: v for k, v in self.__dict__.items() if k != "moe"})
        return out


def _moe_layout(spec: ModelSpec, max_ctx: int) -> Q.Layout:
    """The qwen36moe layout with the attention gate dropped, on the wide law.

    Pool order is qwen36moe's: experts up/gate stripes, experts down, no shared
    (<-> no share regions), then q/k/v/o (no POOL_GATE). Act order is the same as
    qwen36moe's attention act except AA_QG holds q alone (qw floats, not 2*qw)."""
    n = N_CORES
    hid, E = spec.hidden, spec.num_experts
    C = common(spec)
    rw_bytes = hid * E * 2
    kv: dict[str, int] = {}
    # linear-act/consts are unused for this family (no DeltaNet layer)
    kv.update({k: 0 for k in ("C_LNW", "C_SIDE", "C_NW", "C_POSTLN", "C_RW", "C_SGW", "C_WOUT", "C_BYTES",
                              "GLUE_SIDE_BYTES", "SIDE_ALPHA", "SIDE_BETA", "SIDE_SMALL", "SIDE_CONV",
                              "A_XN", "A_QKV", "A_Z", "A_VEC", "A_O", "A_OG", "A_OUT", "A_RES", "A_XM",
                              "A_ROUT", "A_HP", "A_BYTES", "S_ROWS", "S_HEAD_BYTES", "STATE_S_OFF",
                              "STATE_BYTES")})
    qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
    c = Q._Alloc()
    c.add("lnw", Q.ELEM)
    c.add("postln", Q.ELEM)
    c.add("meta", 2048)
    c.add("rw", rw_bytes)
    c.add("sgw", Q.ELEM)
    if 2 * hd * 2 > 1024:
        raise OpRangeError("attn: qn | kn do not fit the 1 KB meta element")
    c.add("bias", Q.ELEM)
    kv.update(CA_LNW=c.off["lnw"], CA_POSTLN=c.off["postln"], CA_META=c.off["meta"], CA_RW=c.off["rw"],
              CA_SGW=c.off["sgw"], CA_BIAS=c.off["bias"], CA_BYTES=c.n)
    a = Q._Alloc()
    a.add("xn", Q.ELEM)
    a.add("qg", qw * 4)                                # q f32 alone (no gate)
    a.add("kvn", 2 * kvw * 4)                          # k | v f32
    a.add("og", qw * 2)
    a.add("out", hid * 4)
    a.add("_unused", 2048)
    a.add("res", Q.roundup((n - 1) * C.ROWS_PC * 4 + C.CALL_BYTES, Q.ELEM))
    a.add("xm", Q.ELEM)
    a.add("rout", C.CALL_BYTES)
    a.add("hp", Q.ELEM)
    kv.update(AA_XN=a.off["xn"], AA_QG=a.off["qg"], AA_KVN=a.off["kvn"], AA_OG=a.off["og"], AA_OUT=a.off["out"],
              AA_RES=a.off["res"], AA_XM=a.off["xm"], AA_ROUT=a.off["rout"], AA_HP=a.off["hp"], AA_BYTES=a.n)
    kv.update(POOL_DOWN=0, POOL_SHARE_UP=0, POOL_SHARE_GATE=0, POOL_SHARE_DOWN=0,
              POOL_QKV=0, POOL_Z=0, POOL_Q=0, POOL_K=0, POOL_V=0, POOL_GATE=0, POOL_O=0)
    p = Q._Alloc()
    p.add("experts_upgate", E * 2 * C.STRIPES_PER_PROJ * C.STRIPE)
    down_off = p.off["experts_upgate"]
    p.add("experts_down", E * C.UP_BYTES)
    proj0 = p.n
    kv.update(POOL_DOWN=down_off)
    q = Q._Alloc(); q.n = proj0
    q.add("q", Q.role_bytes(spec, "attn", qw, hid))
    q.add("k", Q.role_bytes(spec, "attn", kvw, hid))
    q.add("v", Q.role_bytes(spec, "attn", kvw, hid))
    q.add("o", Q.role_bytes(spec, "attn", hid, qw))
    kv.update(POOL_Q=q.off["q"], POOL_K=q.off["k"], POOL_V=q.off["v"], POOL_O=q.off["o"])
    end = q.n
    kv["POOL_BYTES"] = max(Q.POOL_BYTES, Q.roundup(end, Q.MB))
    band = 128 * hid // Q.CHUNK_VALUES * Q.Q8_CHUNK
    bands = spec.vocab // 128
    kv.update(KV_ROW=2 * kvw * 2, PTAB_ROW=Q.PTAB_ROW, MAX_CTX=max_ctx, KV_BYTES=max_ctx * 2 * kvw * 2,
              PTAB_BYTES=max_ctx * Q.PTAB_ROW,
              LMHEAD_POOL_BYTES=Q.roundup(Q.roundup(bands, n) * band, Q.MB), LMHEAD_BAND_BYTES=band,
              LMHEAD_BANDS=bands, ELN=hid * 2, E_A=kvw)
    return Q.Layout(**kv)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Lfm2moeLayout:
    _check(spec)
    from . import dense as D
    L = _moe_layout(spec, max_ctx)
    hid = spec.hidden
    ff = spec.intermediate
    # pool: the MoE expert regions first (qwen36moe.layout), then the conv mixer
    # projections, then the dense FFN projections. The conv block is wider than the
    # attention block it replaces, so the largest of the three views is the pool.
    # The offsets start past the MoE layouts' shared projection end (POOL_GATE is
    # the last MoE projection offset; probe it for this spec).
    # layout() stores POOL_GATE as an offset; recompute the end as the max of the
    # layout's own pool markers plus the MoE projections it sizes.
    # Simpler: size the MoE pool as max across its types, then add the conv/dense
    # regions past L.POOL_BYTES' internal end. Use role_bytes for the exact sizes.
    base = L.POOL_BYTES
    # The pool byte count qwen36moe computed is for experts + q/k/v/gate/o; the
    # LFM2-MoE full layers have no gate and the conv layers have four hid-square
    # projections, so the pool is the max of those here.
    C = common(spec)
    E = spec.num_experts
    # The mixers reuse the region after the experts (POOL_Q region); the dense FFN
    # follows them. PROJ0 is after experts_upgate + experts_down, as in _moe_layout.
    proj0 = E * 2 * C.STRIPES_PER_PROJ * C.STRIPE + E * C.UP_BYTES
    qw2, kvw2 = spec.attn_q_width, spec.attn_kv_width
    attn_size = (Q.role_bytes(spec, "attn", qw2, hid) + Q.role_bytes(spec, "attn", kvw2, hid) * 2
                 + Q.role_bytes(spec, "attn", hid, qw2))
    conv_size = sum(Q.role_bytes(spec, r, hid, hid) for r in ("linear", "linear", "linear", "linear_out"))
    mixer_size = max(attn_size, conv_size)
    p: dict[str, int] = {}
    off = proj0
    for name, role, rows, cols in (("b", "linear", hid, hid), ("c", "linear", hid, hid),
                                   ("u", "linear", hid, hid), ("out", "linear_out", hid, hid)):
        p[name] = off
        off += Q.role_bytes(spec, role, rows, cols)
    # The dense FFN regions follow the conv projections (they only exist on the
    # dense layers; the pool still reserves them so one BO serves both).
    if spec.dense_lead:
        for name, rows, cols in (("up", ff, hid), ("gate", ff, hid), ("down", hid, ff)):
            p[name] = off
            off += Q.role_bytes(spec, "ffn", rows, cols)
    else:
        p["up"] = p["gate"] = p["down"] = off
        # no dense FFN at all: no bytes
    # Dense layout for the leading layers (dx.py's geometry with the FFN width ff and
    # the no-gate attention). Its pool is sized from its own layout(); we need only
    # its const/act sizes and outputs, not its pool.
    try:
        DL = D.layout(spec, max_ctx)
        d_consts, d_act = DL.CD_BYTES, DL.AD_BYTES
    except Exception:
        d_consts, d_act = Q.ELEM * 4, Q.ELEM * 4
    pool_bytes = max(base, Q.roundup(off, Q.MB))
    # Consts: the MoE attention consts + router bias + conv taps, over the largest
    BIAS_BYTES = Q.roundup(max(E * 4, Q.ELEM), Q.ELEM)
    cd_conv = max(L.CA_BYTES, d_consts)
    cd_bytes = Q.roundup(cd_conv + BIAS_BYTES + hid * TAPS * 2, Q.ELEM)
    # Act: the MoE attention act, over which the conv B/C/u/Y sit
    ad_off = max(L.AA_BYTES, d_act)
    a = {}
    for name, size in (("b", hid * 4), ("c", hid * 4), ("u", hid * 4), ("y", hid * 2)):
        a[name] = ad_off
        ad_off += size
    ad_bytes = Q.roundup(ad_off, Q.ELEM)
    return Lfm2moeLayout(
        moe=L, dense=None,
        POOL_SC_B=p["b"], POOL_SC_C=p["c"], POOL_SC_U=p["u"], POOL_SC_OUT=p["out"],
        POOL_SC_UP=p["up"], POOL_SC_GATE=p["gate"], POOL_SC_DOWN=p["down"],
        POOL_BYTES=pool_bytes, CD_CONV=cd_conv, CD_BYTES=cd_bytes,
        AD_B=a["b"], AD_C=a["c"], AD_U=a["u"], AD_Y=a["y"], AD_BYTES=ad_bytes,
        STATE_BYTES=(TAPS - 1) * hid * 4,
        BIAS_BYTES=BIAS_BYTES, ATTN_GATE_BYTES=0,
    )


@dataclass(frozen=True)
class Lfm2moeRecipe:
    spec: ModelSpec
    layout: Lfm2moeLayout
    common: Q.Common
    max_ctx: int = 4096


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> Lfm2moeRecipe:
    _check(spec)
    return Lfm2moeRecipe(spec=spec, layout=layout(spec, max_ctx), common=common(spec), max_ctx=max_ctx)


def pack_plan(spec: ModelSpec) -> dict:
    """Two tail kinds, per mixer. `full_attention` / `short_conv` are the MoE tails
    (qwen36moe's MoE pack with the gate dropped and the short-conv block in); the
    dense variants use lfm2.py's pack (dense pool offsets with the dense FFN).

    `spec.reads_gguf` swaps every projection to the GGUF-reading op as in
    qwen36moe.pack_plan, the expert ops are the `_gguf` twins, and the router takes
    the transpose (its GGUF is [experts, hidden], the pool is [hidden, experts]).
    The expert bias is a `put` (f32 f32) into the router consts."""
    L, C = layout(spec), common(spec)
    E, hid, ff = spec.num_experts, spec.hidden, padded(spec)
    real_ff = spec.moe_intermediate
    pre = "model.layer.{l}."
    stripes_op = "expert_stripes_gguf" if spec.reads_gguf else "expert_stripes"
    down_op = "expert_down_gguf" if spec.reads_gguf else "expert_down"
    experts = [
        {"op": stripes_op, "up": pre + "mlp.up_exps_proj.weight", "gate": pre + "mlp.gate_exps_proj.weight",
         "dst": 0, "experts": E, "stripes": C.STRIPES_PER_PROJ, "stripe_bytes": C.STRIPE, "in_dim": hid,
         "src_rows": real_ff},
        {"op": down_op, "tensor": pre + "mlp.down_exps_proj.weight", "dst": L.POOL_DOWN, "experts": E,
         "expert_bytes": C.UP_BYTES, "src_rows": real_ff},
    ]

    def router_op(dst: int) -> dict:
        if spec.reads_gguf:
            return {"op": "transpose", "tensor": pre + "moe_router.weight", "dst": dst,
                    "rows": E, "cols": hid, "elem": 2}
        return {"op": "put", "tensor": pre + "moe_router.weight", "dst": dst, "cap": hid * E * 2}

    plan: dict = {"pool_bytes": L.POOL_BYTES, "chunk_bytes": Q.pool_chunk(spec), "layer_types": {},
                  "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                              "ops": [{"op": "lmhead_q8", "tensor": "lm_head.weight", "chunk_bytes": Q.Q8_CHUNK, "in_dim": hid, "dst": 0}]},
                  "embed": {"tensor": "model.embed_tokens.weight", "dim": hid},
                  "norm": {"tensor": "model.norm.weight", "bytes": hid * 2}}
    # -- the MoE tails (full / short-conv)
    if FULL in spec.layer_types or SHORT_CONV in spec.layer_types:
        qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
        p = _moe_pack(spec, L)
        # attention MoE
        plan["layer_types"][FULL] = {
            "pool": experts + [
                Q.proj_op(spec, "attn", pre + "self_attn.q_proj.weight", L.POOL_Q, qw, hid, hid),
                Q.proj_op(spec, "attn", pre + "self_attn.k_proj.weight", L.POOL_K, kvw, hid, hid),
                Q.proj_op(spec, "attn", pre + "self_attn.v_proj.weight", L.POOL_V, kvw, hid, hid),
                Q.proj_op(spec, "attn", pre + "self_attn.o_proj.weight", L.POOL_O, hid, qw, hid),
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CA_LNW, "cap": Q.ELEM},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CA_POSTLN, "cap": Q.ELEM},
                {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CA_META, "cap": hd * 2},
                {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CA_META + hd * 2, "cap": hd * 2},
                router_op(L.CA_RW),
                {"op": "put", "tensor": pre + "exp_probs_b.bias", "dst": L.CA_BIAS, "cap": L.BIAS_BYTES},
            ],
        }
        # conv MoE: same experts + router, with the short-conv mixer in the pool
        n = Q.q4_chunks(hid, hid)
        plan["layer_types"][SHORT_CONV] = {
            "pool": experts + [
                Q.proj_op(spec, "linear", pre + "shortconv.in_proj.weight", L.POOL_SC_B, hid, hid, hid, chunk0=0),
                Q.proj_op(spec, "linear", pre + "shortconv.in_proj.weight", L.POOL_SC_C, hid, hid, hid, chunk0=n),
                Q.proj_op(spec, "linear", pre + "shortconv.in_proj.weight", L.POOL_SC_U, hid, hid, hid, chunk0=2 * n),
                Q.proj_op(spec, "linear_out", pre + "shortconv.out_proj.weight", L.POOL_SC_OUT, hid, hid, hid),
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CA_LNW, "cap": Q.ELEM},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CA_POSTLN, "cap": Q.ELEM},
                {"op": "transpose", "tensor": pre + "shortconv.conv.weight", "dst": L.CD_CONV,
                 "rows": hid, "cols": TAPS, "elem": 2},
                router_op(L.CA_RW),
                {"op": "put", "tensor": pre + "exp_probs_b.bias", "dst": L.CA_BIAS, "cap": L.BIAS_BYTES},
            ],
        }
    # -- the dense tails (the leading layers): dense silu FFN on the mixers, no
    # router and no experts. Offsets are this layout's, so the pool BO is one size.
    if SHORT_CONV_DENSE in spec.layer_types:
        n = Q.q4_chunks(hid, hid)
        ffd = spec.intermediate
        plan["layer_types"][SHORT_CONV_DENSE] = {
            "pool": [
                Q.proj_op(spec, "linear", pre + "shortconv.in_proj.weight", L.POOL_SC_B, hid, hid, hid, chunk0=0),
                Q.proj_op(spec, "linear", pre + "shortconv.in_proj.weight", L.POOL_SC_C, hid, hid, hid, chunk0=n),
                Q.proj_op(spec, "linear", pre + "shortconv.in_proj.weight", L.POOL_SC_U, hid, hid, hid, chunk0=2 * n),
                Q.proj_op(spec, "linear_out", pre + "shortconv.out_proj.weight", L.POOL_SC_OUT, hid, hid, hid),
                Q.proj_op(spec, "ffn", pre + "mlp.up_proj.weight", L.POOL_SC_UP, ffd, hid, hid),
                Q.proj_op(spec, "ffn", pre + "mlp.gate_proj.weight", L.POOL_SC_GATE, ffd, hid, hid),
                Q.proj_op(spec, "ffn", pre + "mlp.down_proj.weight", L.POOL_SC_DOWN, hid, ffd, ffd),
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CA_LNW, "cap": Q.ELEM},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CA_POSTLN, "cap": Q.ELEM},
                {"op": "transpose", "tensor": pre + "shortconv.conv.weight", "dst": L.CD_CONV,
                 "rows": hid, "cols": TAPS, "elem": 2},
            ],
        }
    if FULL_DENSE in spec.layer_types:
        qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
        ffd = spec.intermediate
        plan["layer_types"][FULL_DENSE] = {
            "pool": [
                Q.proj_op(spec, "attn", pre + "self_attn.q_proj.weight", L.POOL_Q, qw, hid, hid),
                Q.proj_op(spec, "attn", pre + "self_attn.k_proj.weight", L.POOL_K, kvw, hid, hid),
                Q.proj_op(spec, "attn", pre + "self_attn.v_proj.weight", L.POOL_V, kvw, hid, hid),
                Q.proj_op(spec, "attn", pre + "self_attn.o_proj.weight", L.POOL_O, hid, qw, hid),
                Q.proj_op(spec, "ffn", pre + "mlp.up_proj.weight", L.POOL_SC_UP, ffd, hid, hid),
                Q.proj_op(spec, "ffn", pre + "mlp.gate_proj.weight", L.POOL_SC_GATE, ffd, hid, hid),
                Q.proj_op(spec, "ffn", pre + "mlp.down_proj.weight", L.POOL_SC_DOWN, hid, ffd, ffd),
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CA_LNW, "cap": Q.ELEM},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CA_POSTLN, "cap": Q.ELEM},
                {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CA_META, "cap": hd * 2},
                {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CA_META + hd * 2, "cap": hd * 2},
            ],
        }
    return plan


def _moe_pack(spec: ModelSpec, L) -> dict:
    # The MoE pack is qwen36moe's expert prefix (stripes + down) with no shared expert.
    # Rebuilt here so this module does not mutate the shared plan when the family
    # overrides the projection list.
    return {}


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    _check(spec)
    from . import dense as D
    from . import lfm2 as LFM2
    L = layout(spec)
    out: dict = {
        "contexts": {}, "kernels": {}, "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": spec.vocab * 4,
                    "lmpool": L.LMHEAD_POOL_BYTES, "ptab": {"per_row": Q.PTAB_ROW}},
    }
    out["contexts"]["ln"] = "ln/final.xclbin"
    out["contexts"]["lm"] = "lm_head_q8/final.xclbin"
    out["kernels"]["ln"] = {"context": "ln", "insts": "ln/insts.bin", "build": "ln"}
    out["kernels"]["lm"] = {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"}
    if SHORT_CONV in spec.layer_types:
        args = ["pool", "xres", "consts", "state", "act"]
        check_buffer_args("sx", args)
        out["contexts"].setdefault("sx", "sx0/final.xclbin")
        out["kernels"]["sx0"] = {"context": "sx", "insts": "sx0/insts.bin", "build": "sx0"}
        out["kernels"]["sx1"] = {"context": "sx", "insts": "sx1/insts.bin", "patch": "moeroute2", "build": "sx1"}
        out["layer_types"][SHORT_CONV] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES,
                        "state": {"kind": "linear", "bytes": L.STATE_BYTES}},
            "program": [{"op": "run", "kernel": "sx0", "args": args},
                        {"op": "moeroute2", "kernel": "sx1", "act_off": L.A_ROUT},
                        {"op": "run", "kernel": "sx1", "args": args}],
        }
    if FULL in spec.layer_types:
        args = ["pool", "xres", "consts", "state", "act", "ptab"]
        check_buffer_args("ax", args)
        out["contexts"].setdefault("ax", "ax0/final.xclbin")
        out["kernels"].setdefault("ax0", {"context": "ax", "insts": "ax0/insts.bin", "patch": "attnpos", "build": "ax0"})
        out["kernels"].setdefault("ax1", {"context": "ax", "insts": "ax1/insts.bin", "patch": "moeroute2", "build": "ax1"})
        out["layer_types"][FULL] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": "ax0", "args": args},
                        {"op": "moeroute2", "kernel": "ax1", "act_off": L.AA_ROUT},
                        {"op": "run", "kernel": "ax1", "args": args}],
        }
    # dense tails: cx.py for the conv+dense layers (lfm2's SHORT_CONV program),
    # dx.py for the full+dense layers (the dense family's DENSE program). Built
    # inline so neither guarded family recipe is called with a lfm2moe spec.
    if SHORT_CONV_DENSE in spec.layer_types:
        args = ["pool", "xres", "consts", "state", "act"]
        check_buffer_args("cx", args)
        out["contexts"]["cx"] = "short_conv/final.xclbin"
        out["kernels"]["cx"] = {"context": "cx", "insts": "short_conv/insts.bin", "build": "short_conv"}
        out["layer_types"][SHORT_CONV_DENSE] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES,
                        "state": {"kind": "linear", "bytes": L.STATE_BYTES}},
            "program": [{"op": "run", "kernel": "cx", "args": args}],
        }
    if FULL_DENSE in spec.layer_types:
        args = ["pool", "xres", "consts", "state", "act", "ptab"]
        check_buffer_args("dx", args)
        out["contexts"]["dx"] = "dx/final.xclbin"
        out["kernels"]["dx"] = {"context": "dx", "insts": "dx/insts.bin", "patch": "attnpos", "build": "dx"}
        out["layer_types"][FULL_DENSE] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": "dx", "args": args}],
        }
    sec = Q.GGUF.section(spec, max_ctx, programs, twins_built=Q.GGUF_EXPERT_OPS_READY)
    if sec:
        out["gguf"] = sec
    return out


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_experts": spec.num_experts, "num_experts_per_tok": spec.experts_per_tok,
            "moe_intermediate_size": spec.moe_intermediate, "intermediate_size": spec.intermediate,
            "num_dense_layers": spec.dense_lead,
            "head_dim": spec.head_dim, "num_attention_heads": spec.num_heads,
            "num_key_value_heads": spec.num_kv_heads, "num_attention_heads": spec.num_heads,
            "rope_parameters": {"rope_theta": spec.rope_theta, "rope_type": "default"},
            "norm_eps": spec.norm_eps, "layer_types": list(spec.layer_types)}


def hf_config_defaults(spec: ModelSpec) -> dict:
    return {"norm_eps": spec.norm_eps, "rope_parameters": {"rope_theta": spec.rope_theta, "rope_type": "default"}}


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    R = recipe(spec, max_ctx)
    L, C = R.layout, R.common
    return {
        "hidden": spec.hidden, "vocab": spec.vocab, "real_vocab": spec.real_vocab,
        "chunk_bytes": Q.pool_chunk(spec), "pool_bytes": L.POOL_BYTES,
        "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": Q.Q8_CHUNK,
        "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
        "rope_inv_freq": spec.rope_inv_freq(),
        "rout_idx_off": Q.ROUT_IDX_OFF,
        "moe": {"experts": spec.num_experts, "topk": spec.experts_per_tok,
                "stripe": C.STRIPE, "up_bytes": C.UP_BYTES, "down_core": C.DOWN_PER_CORE * C.DOWN_BAND,
                "pool_down": L.POOL_DOWN, "share_up": 0, "share_gate": 0, "share_down": 0,
                "padded_ff": padded(spec), "halves_per_core": halves_per_core(spec)},
    }


def builds(spec: ModelSpec) -> dict[str, dict]:
    qh = spec.quant_hash()
    sfx = f"_q{qh}" if qh else ""
    b: dict[str, dict] = {}
    if SHORT_CONV in spec.layer_types:
        b["sx0"] = {"design": "layer_x/sx.py", "build_dir": f"layer_x/build_sx0{sfx}", "env": {"SX_PART": "0"}}
        b["sx1"] = {"design": "layer_x/sx.py", "build_dir": f"layer_x/build_sx1{sfx}", "env": {"SX_PART": "1"}}
    if FULL in spec.layer_types:
        b["ax0"] = {"design": "layer_x/ax.py", "build_dir": f"layer_x/build_ax0{sfx}", "env": {"AX_PART": "0"}}
        b["ax1"] = {"design": "layer_x/ax.py", "build_dir": f"layer_x/build_ax1{sfx}", "env": {"AX_PART": "1"}}
    if SHORT_CONV_DENSE in spec.layer_types:
        b["short_conv"] = {"design": "short_conv/cx.py",
                           "build_dir": f"short_conv/build_h{spec.hidden}_t{TAPS}",
                           "env": {"SC_HID": str(spec.hidden), "SC_TAPS": str(TAPS)}}
    if FULL_DENSE in spec.layer_types:
        b["dx"] = {"design": "dense/dx.py", "build_dir": f"dense/build_h{spec.hidden}",
                   "env": {}}
    b["ln"] = {"design": "ln/ln.py", "build_dir": "ln/build", "env": {}}
    b["lm_head_q8"] = {"design": "lm_head_q8/lm_head_q8.py", "build_dir": "lm_head_q8/build_full",
                       "env": {"LMHEAD_N": str(spec.vocab), "LMHEAD_K": str(spec.hidden),
                               "LMHEAD_CORES": str(N_CORES)}}
    return b


GEN_KERNELS = "designs/layer_x/gen_kernels.py"
KERNEL_SOURCES = list(Q.KERNEL_SOURCES) + [
    "designs/short_conv/cx.py", "designs/short_conv/*.cc", "designs/short_conv/*.h",
    "designs/router/*.cc", "designs/router/*.h", "designs/router/*.py",
]
