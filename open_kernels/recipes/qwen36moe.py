"""The Qwen3.5 / Qwen3.6-MoE recipe: ModelSpec -> everything the whole-layer
designs (designs/layer_x/lx.py, ax.py), the packers and the driver need.

    Layout   the byte layouts of consts / act / state / kv / ptab / pool (designs/layer_x/layout.py)
    Common   the main-core geometry shared by both layer types (designs/layer_x/xcommon.py)
    Linear   the linear-attention layer's dispatch geometry (lx.py)
    Attn     the full-attention layer's dispatch geometry (ax.py)
    pack_plan  which tensor lands at which offset in which chunk order (pool, consts, lm_head)
    programs   the per-layer-type verb sequence the driver runs, and the tail
    builds     the kernel sets to build (design source + compile-time knobs)

Everything is arithmetic on the spec except what the catalogue pins: the
helper-core placement, the shim budget and the 16 KB program memory are
properties of the two hand-placed designs, so `recipe()` checks the spec
against the catalogue's validated points and refuses anything else.

The 27B numbers this reproduces byte-for-byte are frozen in
specs/open-engine/tests/test_recipe_layout.py; a change here that moves an
offset fails that test before it reaches a build.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalogue import LIMITS, OpRangeError, check_buffer_args, require
from .spec import FULL, LINEAR, ModelSpec

# ---- the q4_1 / q8 pool chunk formats (gemv_q4.h, lm_head_q8.h): format constants, not model ones
CHUNK = 5120                 # q4_1: 32 rows x 256 K (8192 values) + bf16 d, m per 32-block
CHUNK_F32 = 6144             # GGUF-direct (spec.quant "q4_1_f32"): the same tile, the fp16
                             # block scales/mins widened EXACTLY to f32 (open_kernels/gguf_pool.py)
CHUNK_VALUES = 8192
CHUNK_ROWS = 32
Q8_CHUNK = 8704              # lm_head q8: 8192 int8 + 256 bf16 scales
Q8_CHUNK_F32 = 9216          # GGUF-direct: the same tile with f32 scales
ELEM = 4096                  # one act / x-stream element
BAND_ROWS = 64               # rows per GEMV band (one y element of 64 floats)
PER_CALL = 2                 # chunks per w element
CALL_BYTES = PER_CALL * CHUNK
MB = 1 << 20
POOL_BYTES = 512 * MB        # one layer's weight pool (fixed BO size; the recipe checks it fits)
PTAB_ROW = 1024              # the position record: [i32 pos | i32 nf | ... cos @512 | sin @640]
PTAB_COS, PTAB_SIN = 512, 640
ROUT_IDX_OFF = 1024          # int32 idx[topk] inside the router record (f32 probs first)
DN_RECORD_FLOATS = 512       # the DeltaNet per-head record [k | q | v | decay | beta | pad] (dnx.h)


def chunk_bytes(quant: str) -> int:
    """The q4 pool chunk bytes for a scale layout: q4nx bf16 (5120) or GGUF-direct f32 (6144)."""
    if quant == "q4_1":
        return CHUNK
    if quant == "q4_1_f32":
        return CHUNK_F32
    raise OpRangeError(f"quant={quant!r}: no open q4 pool chunk layout "
                       f"(have q4_1, q4_1_f32)")


def q8_chunk_bytes(quant: str) -> int:
    """The lm_head q8 chunk bytes: q4nx bf16 scales (8704) or GGUF-direct f32 (9216)."""
    if quant in ("q4_1", "q4_1_f32", "q8_0"):
        return Q8_CHUNK
    if quant == "q8_0_f32":
        return Q8_CHUNK_F32
    raise OpRangeError(f"quant={quant!r}: no open q8 pool chunk layout")


def q4_bytes(rows: int, cols: int, quant: str = "q4_1") -> int:
    n = rows * cols
    if n % CHUNK_VALUES:
        raise OpRangeError(f"q4 tensor [{rows}, {cols}] is not a whole number of {CHUNK_VALUES}-value chunks")
    return n // CHUNK_VALUES * chunk_bytes(quant)


def q4_chunks(rows: int, cols: int, quant: str = "q4_1") -> int:
    return q4_bytes(rows, cols, quant) // chunk_bytes(quant)


def band_bytes(K: int, quant: str = "q4_1") -> int:
    """One 64-row band of a K-wide standard-layout matrix: K/128 chunks."""
    return q4_bytes(BAND_ROWS, K, quant)


def tab_bytes(K: int) -> int:
    """gemv_q4_tab_bytes(K) = 2.25 K (gemv_tab.h)."""
    return 2 * K + K // 8 + K // 8


def roundup(n: int, m: int) -> int:
    return (n + m - 1) // m * m


class _Alloc:
    """Sequential byte allocator for a buffer layout: name -> offset, in order."""

    def __init__(self):
        self.off: dict[str, int] = {}
        self.n = 0

    def add(self, name: str, size: int, align: int = 1) -> int:
        self.n = roundup(self.n, align)
        self.off[name] = self.n
        self.n += size
        return self.off[name]


@dataclass(frozen=True)
class Layout:
    # consts, linear layer
    C_LNW: int; C_SIDE: int; C_NW: int; C_POSTLN: int; C_RW: int; C_SGW: int; C_WOUT: int; C_BYTES: int
    GLUE_SIDE_BYTES: int
    SIDE_ALPHA: int; SIDE_BETA: int; SIDE_SMALL: int; SIDE_CONV: int     # inside the glue side blob
    # consts, attention layer
    CA_LNW: int; CA_POSTLN: int; CA_META: int; CA_RW: int; CA_SGW: int; CA_BYTES: int
    # act, linear layer
    A_XN: int; A_QKV: int; A_Z: int; A_VEC: int; A_O: int; A_OG: int; A_OUT: int
    A_RES: int; A_XM: int; A_ROUT: int; A_HP: int; A_BYTES: int
    # act, attention layer
    AA_XN: int; AA_QG: int; AA_KVN: int; AA_OG: int; AA_OUT: int
    AA_RES: int; AA_XM: int; AA_ROUT: int; AA_HP: int; AA_BYTES: int
    # state BO (linear layers)
    S_ROWS: int; S_HEAD_BYTES: int; STATE_S_OFF: int; STATE_BYTES: int
    # pool offsets
    POOL_QKV: int; POOL_Z: int
    POOL_Q: int; POOL_K: int; POOL_V: int; POOL_GATE: int; POOL_O: int
    POOL_DOWN: int; POOL_SHARE_UP: int; POOL_SHARE_GATE: int; POOL_SHARE_DOWN: int
    POOL_BYTES: int
    # KV cache / position table
    KV_ROW: int; PTAB_ROW: int; MAX_CTX: int; KV_BYTES: int; PTAB_BYTES: int
    # lm_head
    LMHEAD_POOL_BYTES: int; LMHEAD_BAND_BYTES: int; LMHEAD_BANDS: int

    def constants(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Common:
    """xcommon.py's geometry: the main cores' streams, the MoE block, DeltaNet on the main cores."""
    N_CORES: int; HID: int; FF: int; NE: int; NX: int
    TILE: int; PER_CALL: int; CALL_BYTES: int
    STRIPE: int; HALF: int; PAIR: int; DOWN_BAND: int; UP_BYTES: int; DOWN_PER_CORE: int
    BAND_ROWS: int; BAND16: int; BAND32: int; N_HDR: int
    MS_FLOATS: int; DS_FLOATS: int; TAB_BYTES: int; H_TAB_OFF: int; KWIDE: int
    MS_RW: int; MS_XR: int; MS_ACC: int; MS_U: int; MS_G: int; MS_YD: int
    ROWS_PC: int; HID_PC: int            # MoE rows per core (down / block output), hidden per core
    STRIPES_PER_PROJ: int; CORES_PER_STRIPE: int
    UP_ELEMS: int; DOWN_ELEMS: int; OUT_ELEMS: int
    W_ELEMS: int
    DN_ROWS: int; DN_SLICES: int; DN_HEADS_PC: int; DN_DIM: int


@dataclass(frozen=True)
class Linear:
    """lx.py's geometry."""
    QKV_PC: int; Z_PC: int; OUT_PC: int
    QKV_DIM: int; VW: int                # fused q|k|v rows; the value width (z, o, og)
    NCH: int; NHEAD: int; TILE: int; NT: int; AB_ELEMS: int; G: int; NG: int
    KEY_WIDTH: int; HEADS_PER_TILE: int; VALUE_TILE0: int
    RECORD_BYTES: int; O_HEAD_BYTES: int
    OUT_K: int; QKV_K: int


@dataclass(frozen=True)
class Attn:
    """ax.py's geometry."""
    NH: int; KVH: int; HD: int; ROT: int
    Q_PC: int; KV_PC: int; O_PC: int
    QW: int; KVW: int                    # q (and gate) width, k (and v) width
    O_K: int; QKV_K: int
    META_BYTES: int                      # [qn | kn] bf16, the first meta element
    HEAD_BYTES: int                      # one f32 head = one ain element


@dataclass(frozen=True)
class Recipe:
    spec: ModelSpec
    layout: Layout
    common: Common
    linear: Linear | None
    attn: Attn | None
    max_ctx: int = 4096


def _check(spec: ModelSpec) -> None:
    if spec.family != "qwen36moe":
        raise OpRangeError(f"qwen36moe recipe given a {spec.family!r} spec")
    if spec.quant != "q4_1":
        raise OpRangeError(f"qwen36moe: quant={spec.quant!r}; the gemv_q4 template reads q4_1 chunks only")
    if spec.num_experts == 0 or spec.shared_expert_intermediate == 0:
        raise OpRangeError("qwen36moe: a dense model or one without a shared expert is not this recipe")
    n = LIMITS["n_cols"]
    require("ln", width=spec.hidden)
    require("router", experts=spec.num_experts, topk=spec.experts_per_tok)
    require("moe", ff=spec.moe_intermediate, experts=spec.num_experts, topk=spec.experts_per_tok,
            shared_expert=True, hidden=spec.hidden, n_cores=n)
    if spec.shared_expert_intermediate != spec.moe_intermediate:
        raise OpRangeError("qwen36moe: the shared expert must have the routed experts' width "
                           f"({spec.shared_expert_intermediate} vs {spec.moe_intermediate})")
    require("gemv_q4_prep_f32", K=spec.moe_intermediate)
    require("lm_head_q8", K=spec.hidden, vocab=spec.vocab)
    if spec.has_linear:
        require("deltanet", heads=spec.lin_value_heads, dim=spec.lin_value_dim, key_heads=spec.lin_key_heads,
                conv_kernel=spec.conv_kernel)
        if spec.lin_key_dim != spec.lin_value_dim:
            raise OpRangeError("qwen36moe: DeltaNet key and value head dims must match")
        require("gemv_q4", K=spec.hidden, rs=2, rows_per_core=spec.lin_qkv_dim // n, per_call=PER_CALL)
        require("gemv_q4", K=spec.hidden, rs=2, rows_per_core=spec.lin_value_width // n, per_call=PER_CALL)
        require("gemv_q4", K=spec.lin_value_width, rs=2, rows_per_core=spec.hidden // n, per_call=PER_CALL)
    if spec.has_full:
        require("attn", head_dim=spec.head_dim, num_heads=spec.num_heads, num_kv_heads=spec.num_kv_heads,
                rotary_dim=spec.rotary_dim, rope_theta=spec.rope_theta, qk_norm=spec.qk_norm,
                attn_gate=spec.attn_gate)
        require("gemv_q4", K=spec.hidden, rs=2, rows_per_core=spec.attn_q_width // n, per_call=PER_CALL)
        require("gemv_q4", K=spec.hidden, rs=2, rows_per_core=spec.attn_kv_width // n, per_call=PER_CALL)
        require("gemv_q4", K=spec.attn_q_width, rs=2, rows_per_core=spec.hidden // n, per_call=PER_CALL)
    require("gemv_q4", K=spec.hidden, rs=4, rows_per_core=BAND_ROWS, per_call=PER_CALL)   # expert up / gate halves


def common(spec: ModelSpec) -> Common:
    n = LIMITS["n_cols"]
    hid, ff, ne = spec.hidden, spec.moe_intermediate, spec.experts_per_tok
    stripe = q4_bytes(128, hid)                     # 128 rows x HID: one up (or gate) stripe, RS=4
    half = q4_bytes(BAND_ROWS, hid)                 # 64 rows x HID
    down_band = q4_bytes(128, ff)                   # 128 rows x FF, RS=4
    up_bytes = q4_bytes(ff, hid)                    # one expert's up (= gate = down)
    rows_pc, hid_pc = hid // n, ff // n
    # ms scratch (floats): rw[32] | xr[rows_pc] | acc[rows_pc] | u[hid_pc] | g[hid_pc] | yd[rows_pc]
    ms_rw, ms_xr = 0, 32
    ms_acc = ms_xr + rows_pc
    ms_u = ms_acc + rows_pc
    ms_g = ms_u + hid_pc
    ms_yd = ms_g + hid_pc
    ms_floats = ms_yd + rows_pc
    if 8 + ne > 32:
        raise OpRangeError(f"moe: top-k {ne} does not fit the 32-float routing record")
    wide = max(hid, spec.lin_value_width if spec.has_linear else 0, spec.attn_q_width if spec.has_full else 0)
    tab = tab_bytes(wide)
    h_tab = tab_bytes(hid)
    if h_tab + tab_bytes(ff) > tab:
        raise OpRangeError("moe: the hidden h's table does not fit past xm's in the core scratch")
    dn_dim = spec.lin_value_dim if spec.has_linear else 0
    dn_rows = CALL_BYTES // (dn_dim * 4) if dn_dim else 0      # S rows per streamed 10 KB element
    dn_slices = roundup(dn_dim, dn_rows) // dn_rows if dn_dim else 0
    return Common(
        N_CORES=n, HID=hid, FF=ff, NE=ne, NX=ne + 1,
        TILE=CHUNK, PER_CALL=PER_CALL, CALL_BYTES=CALL_BYTES,
        STRIPE=stripe, HALF=half, PAIR=2 * CHUNK, DOWN_BAND=down_band, UP_BYTES=up_bytes,
        DOWN_PER_CORE=rows_pc // 128,
        BAND_ROWS=BAND_ROWS, BAND16=band_bytes(hid), BAND32=band_bytes(2 * hid), N_HDR=3,
        MS_FLOATS=ms_floats, DS_FLOATS=1280, TAB_BYTES=tab, H_TAB_OFF=h_tab, KWIDE=wide,
        MS_RW=ms_rw, MS_XR=ms_xr, MS_ACC=ms_acc, MS_U=ms_u, MS_G=ms_g, MS_YD=ms_yd,
        ROWS_PC=rows_pc, HID_PC=hid_pc,
        STRIPES_PER_PROJ=ff // 128, CORES_PER_STRIPE=n // (ff // 128),
        UP_ELEMS=half // CALL_BYTES, DOWN_ELEMS=(rows_pc // 128) * down_band // CALL_BYTES,
        OUT_ELEMS=rows_pc // BAND_ROWS,
        W_ELEMS=hid * spec.num_experts * 2 // ELEM,
        DN_ROWS=dn_rows, DN_SLICES=dn_slices, DN_HEADS_PC=(spec.lin_value_heads // n) if dn_dim else 0,
        DN_DIM=dn_dim,
    )


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    n = LIMITS["n_cols"]
    hid, E, ff = spec.hidden, spec.num_experts, spec.moe_intermediate
    C = common(spec)
    rw_bytes = hid * E * 2                            # router W bf16 [HID, E]
    kv = {}

    # ---- consts, linear layer: [lnw][glue side minus xn][nw][postln][router W][sgw][out_proj q4]
    vw = spec.lin_value_width if spec.has_linear else 0
    nch = spec.lin_qkv_dim if spec.has_linear else 0
    if spec.has_linear:
        alpha = hid * spec.lin_value_heads * 2        # alpha / beta projections bf16 [HID, heads]
        side = _Alloc()
        side.add("alpha", alpha)
        side.add("beta", alpha)
        side.add("small", ELEM)                       # [a f32[32] | dt_bias f32[32] | pad]
        side.add("conv", spec.conv_kernel * nch * 2)  # conv1d transposed to [groups][taps][1024] bf16
        glue_side = side.n
        c = _Alloc()
        c.add("lnw", ELEM)
        c.add("side", glue_side)
        c.add("nw", ELEM)
        c.add("postln", ELEM)
        c.add("rw", rw_bytes)
        c.add("sgw", ELEM)
        c.add("wout", 2 * q4_bytes(hid, vw))          # the region is twice the tensor (the captured
        kv.update(C_LNW=c.off["lnw"], C_SIDE=c.off["side"], C_NW=c.off["nw"], C_POSTLN=c.off["postln"],
                  C_RW=c.off["rw"], C_SGW=c.off["sgw"], C_WOUT=c.off["wout"], C_BYTES=c.n,
                  GLUE_SIDE_BYTES=glue_side, SIDE_ALPHA=side.off["alpha"], SIDE_BETA=side.off["beta"],
                  SIDE_SMALL=side.off["small"], SIDE_CONV=side.off["conv"])
        # fixture was; it is a BO size only, kept for byte identity with the shipped builds)
        # ---- act, linear layer
        a = _Alloc()
        a.add("xn", ELEM)
        a.add("qkv", nch * 4)
        a.add("z", vw * 4)
        a.add("vec", spec.lin_value_heads * DN_RECORD_FLOATS * 4)
        a.add("o", vw * 4)
        a.add("og", vw * 2)
        a.add("out", hid * 4)
        a.add("res", roundup((n - 1) * C.ROWS_PC * 4 + CALL_BYTES, ELEM))   # the MoE header reads 10 KB slices
        a.add("xm", ELEM)
        a.add("rout", CALL_BYTES)
        a.add("hp", ELEM)
        kv.update(A_XN=a.off["xn"], A_QKV=a.off["qkv"], A_Z=a.off["z"], A_VEC=a.off["vec"], A_O=a.off["o"],
                  A_OG=a.off["og"], A_OUT=a.off["out"], A_RES=a.off["res"], A_XM=a.off["xm"],
                  A_ROUT=a.off["rout"], A_HP=a.off["hp"], A_BYTES=a.n)
        # ---- state BO: [conv state bf16 (taps-1) x NCH][S: heads x S_ROWS rows x dim f32]
        s_rows = C.DN_SLICES * C.DN_ROWS
        s_head = s_rows * C.DN_DIM * 4
        s_off = (spec.conv_kernel - 1) * nch * 2
        kv.update(S_ROWS=s_rows, S_HEAD_BYTES=s_head, STATE_S_OFF=s_off,
                  STATE_BYTES=s_off + spec.lin_value_heads * s_head)
    else:
        kv.update({k: 0 for k in ("C_LNW", "C_SIDE", "C_NW", "C_POSTLN", "C_RW", "C_SGW", "C_WOUT", "C_BYTES",
                                  "GLUE_SIDE_BYTES", "SIDE_ALPHA", "SIDE_BETA", "SIDE_SMALL", "SIDE_CONV",
                                  "A_XN", "A_QKV", "A_Z", "A_VEC", "A_O", "A_OG", "A_OUT", "A_RES", "A_XM",
                                  "A_ROUT", "A_HP", "A_BYTES", "S_ROWS", "S_HEAD_BYTES", "STATE_S_OFF",
                                  "STATE_BYTES")})

    # ---- consts, attention layer: [lnw][postln][meta: qn | kn][router W][sgw]
    if spec.has_full:
        qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
        c = _Alloc()
        c.add("lnw", ELEM)
        c.add("postln", ELEM)
        c.add("meta", 2048)                            # [qn bf16 HD @0 | kn @HD*2]; 1 KB used of 2
        c.add("rw", rw_bytes)
        c.add("sgw", ELEM)
        if 2 * hd * 2 > 1024:
            raise OpRangeError("attn: qn | kn do not fit the 1 KB meta element")
        kv.update(CA_LNW=c.off["lnw"], CA_POSTLN=c.off["postln"], CA_META=c.off["meta"], CA_RW=c.off["rw"],
                  CA_SGW=c.off["sgw"], CA_BYTES=c.n)
        a = _Alloc()
        a.add("xn", ELEM)
        a.add("qg", 2 * qw * 4)                        # q | gate f32
        a.add("kvn", 2 * kvw * 4)                      # k | v f32
        a.add("og", qw * 2)
        a.add("out", hid * 4)
        a.add("_unused", 2048)                         # kept for byte identity with the shipped builds
        a.add("res", roundup((n - 1) * C.ROWS_PC * 4 + CALL_BYTES, ELEM))
        a.add("xm", ELEM)
        a.add("rout", CALL_BYTES)
        a.add("hp", ELEM)
        kv.update(AA_XN=a.off["xn"], AA_QG=a.off["qg"], AA_KVN=a.off["kvn"], AA_OG=a.off["og"], AA_OUT=a.off["out"],
                  AA_RES=a.off["res"], AA_XM=a.off["xm"], AA_ROUT=a.off["rout"], AA_HP=a.off["hp"], AA_BYTES=a.n)
        kv_row = 2 * kvw * 2                           # [K_t bf16 | V_t bf16]
    else:
        kv.update({k: 0 for k in ("CA_LNW", "CA_POSTLN", "CA_META", "CA_RW", "CA_SGW", "CA_BYTES", "AA_XN", "AA_QG",
                                  "AA_KVN", "AA_OG", "AA_OUT", "AA_RES", "AA_XM", "AA_ROUT", "AA_HP", "AA_BYTES")})
        kv_row = 0

    # ---- the layer pool: experts first (routed up/gate stripes, routed down, shared), then the projections
    p = _Alloc()
    p.add("experts_upgate", E * 2 * C.STRIPES_PER_PROJ * C.STRIPE)
    p.add("experts_down", E * C.UP_BYTES)
    p.add("share_up", C.UP_BYTES)
    p.add("share_gate", C.UP_BYTES)
    p.add("share_down", C.UP_BYTES)
    proj0 = p.n
    kv.update(POOL_DOWN=p.off["experts_down"], POOL_SHARE_UP=p.off["share_up"], POOL_SHARE_GATE=p.off["share_gate"],
              POOL_SHARE_DOWN=p.off["share_down"])
    end = proj0
    if spec.has_linear:
        q = _Alloc(); q.n = proj0
        q.add("qkv", q4_bytes(nch, hid))
        q.add("z", q4_bytes(vw, hid))
        kv.update(POOL_QKV=q.off["qkv"], POOL_Z=q.off["z"])
        end = max(end, q.n)
    else:
        kv.update(POOL_QKV=0, POOL_Z=0)
    if spec.has_full:
        q = _Alloc(); q.n = proj0
        q.add("q", q4_bytes(qw, hid))
        q.add("k", q4_bytes(kvw, hid))
        q.add("v", q4_bytes(kvw, hid))
        q.add("gate", q4_bytes(qw, hid))
        q.add("o", q4_bytes(hid, qw))
        kv.update(POOL_Q=q.off["q"], POOL_K=q.off["k"], POOL_V=q.off["v"], POOL_GATE=q.off["gate"], POOL_O=q.off["o"])
        end = max(end, q.n)
    else:
        kv.update(POOL_Q=0, POOL_K=0, POOL_V=0, POOL_GATE=0, POOL_O=0)
    if end > POOL_BYTES:
        raise OpRangeError(f"qwen36moe: the layer's weights ({end} B) exceed the {POOL_BYTES} B pool")
    kv["POOL_BYTES"] = POOL_BYTES

    # ---- lm_head q8: 128-row bands of HID. The pool BO holds a whole number of bands per core
    # (the closed engine's size, 517 MB for the 27B); the design streams exactly `bands`.
    band = 128 * hid // CHUNK_VALUES * Q8_CHUNK
    bands = spec.vocab // 128
    kv.update(KV_ROW=kv_row, PTAB_ROW=PTAB_ROW, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * PTAB_ROW,
              LMHEAD_POOL_BYTES=roundup(roundup(bands, n) * band, MB), LMHEAD_BAND_BYTES=band, LMHEAD_BANDS=bands)
    return Layout(**kv)


def linear(spec: ModelSpec) -> Linear | None:
    if not spec.has_linear:
        return None
    n = LIMITS["n_cols"]
    C = common(spec)
    nch, vw = spec.lin_qkv_dim, spec.lin_value_width
    tile = 1024                                     # dn_glue's channel tile
    key_width = spec.lin_key_heads * spec.lin_key_dim
    return Linear(
        QKV_PC=nch // BAND_ROWS // n, Z_PC=vw // BAND_ROWS // n, OUT_PC=spec.hidden // BAND_ROWS // n,
        QKV_DIM=nch, VW=vw,
        NCH=nch, NHEAD=spec.lin_value_heads, TILE=tile, NT=nch // tile,
        AB_ELEMS=spec.hidden * spec.lin_value_heads * 2 // ELEM, G=tile, NG=vw // tile,
        KEY_WIDTH=key_width, HEADS_PER_TILE=tile // spec.lin_value_dim, VALUE_TILE0=2 * key_width // tile,
        RECORD_BYTES=DN_RECORD_FLOATS * 4, O_HEAD_BYTES=spec.lin_value_dim * 4,
        OUT_K=vw, QKV_K=spec.hidden,
    )


def attn(spec: ModelSpec) -> Attn | None:
    if not spec.has_full:
        return None
    n = LIMITS["n_cols"]
    qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
    return Attn(
        NH=spec.num_heads, KVH=spec.num_kv_heads, HD=hd, ROT=spec.rotary_dim,
        Q_PC=qw // BAND_ROWS // n, KV_PC=kvw // BAND_ROWS // n, O_PC=spec.hidden // BAND_ROWS // n,
        QW=qw, KVW=kvw, O_K=qw, QKV_K=spec.hidden,
        META_BYTES=2 * hd * 2, HEAD_BYTES=hd * 4,
    )


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> Recipe:
    _check(spec)
    return Recipe(spec=spec, layout=layout(spec, max_ctx), common=common(spec), linear=linear(spec), attn=attn(spec),
                  max_ctx=max_ctx)


# ---- the packing plan: tensor -> offset -> chunk order. `{l}` is the layer index.
def pack_plan(spec: ModelSpec) -> dict:
    L, C = layout(spec), common(spec)
    E, hid, ff = spec.num_experts, spec.hidden, spec.moe_intermediate
    pre = "model.layer.{l}."
    experts = [
        {"op": "expert_stripes", "up": pre + "mlp.up_exps_proj.weight", "gate": pre + "mlp.gate_exps_proj.weight",
         "dst": 0, "experts": E, "stripes": C.STRIPES_PER_PROJ, "stripe_bytes": C.STRIPE, "in_dim": hid},
        {"op": "expert_down", "tensor": pre + "mlp.down_exps_proj.weight", "dst": L.POOL_DOWN, "experts": E,
         "expert_bytes": C.UP_BYTES},
        {"op": "std_perm", "tensor": pre + "mlp.share_up_exps_proj.weight", "dst": L.POOL_SHARE_UP,
         "nch": q4_chunks(ff, hid), "in_dim": hid},
        {"op": "std_perm", "tensor": pre + "mlp.share_gate_exps_proj.weight", "dst": L.POOL_SHARE_GATE,
         "nch": q4_chunks(ff, hid), "in_dim": hid},
        {"op": "std_perm", "tensor": pre + "mlp.share_down_exps_proj.weight", "dst": L.POOL_SHARE_DOWN,
         "nch": q4_chunks(hid, ff), "in_dim": ff},
    ]
    plan: dict = {"pool_bytes": L.POOL_BYTES, "chunk_bytes": CHUNK, "layer_types": {},
                  "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                              "ops": [{"op": "lmhead_q8", "tensor": "lm_head.weight", "chunk_bytes": Q8_CHUNK, "dst": 0}]},
                  "embed": {"tensor": "model.embed_tokens.weight", "dim": hid},
                  "norm": {"tensor": "model.norm.weight", "bytes": hid * 2}}
    if spec.has_linear:
        vw, nch = spec.lin_value_width, spec.lin_qkv_dim
        side = L.C_SIDE
        plan["layer_types"][LINEAR] = {
            "pool": experts + [
                {"op": "std_perm", "tensor": pre + "linear_attn.qkv_proj.weight", "dst": L.POOL_QKV,
                 "nch": q4_chunks(nch, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.gate_proj.weight", "dst": L.POOL_Z,
                 "nch": q4_chunks(vw, hid), "in_dim": hid},
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.C_LNW, "cap": ELEM},
                {"op": "put", "tensor": pre + "linear_attn.ssm_alpha_proj.weight", "dst": side + L.SIDE_ALPHA,
                 "cap": L.SIDE_BETA - L.SIDE_ALPHA},
                {"op": "put", "tensor": pre + "linear_attn.ssm_beta_proj.weight", "dst": side + L.SIDE_BETA,
                 "cap": L.SIDE_SMALL - L.SIDE_BETA},
                {"op": "put", "tensor": pre + "linear_attn.ssm_a", "dst": side + L.SIDE_SMALL,
                 "cap": spec.lin_value_heads * 4},
                {"op": "put", "tensor": pre + "linear_attn.ssm_dt.bias", "dst": side + L.SIDE_SMALL + spec.lin_value_heads * 4,
                 "cap": spec.lin_value_heads * 4},
                {"op": "conv_transpose", "tensor": pre + "linear_attn.ssm_conv1d.weight", "dst": side + L.SIDE_CONV,
                 "taps": spec.conv_kernel, "groups": nch // 1024, "width": 1024},
                {"op": "put", "tensor": pre + "linear_attn.ssm_norm.weight", "dst": L.C_NW, "cap": spec.lin_value_dim * 2},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.C_POSTLN, "cap": ELEM},
                {"op": "put", "tensor": pre + "moe_router.weight", "dst": L.C_RW, "cap": hid * E * 2},
                {"op": "put", "tensor": pre + "shared_expert_gate.weight", "dst": L.C_SGW, "cap": ELEM},
                {"op": "std_perm", "tensor": pre + "linear_attn.ssm_out_proj.weight", "dst": L.C_WOUT,
                 "nch": q4_chunks(hid, vw), "in_dim": vw},
            ],
        }
    if spec.has_full:
        qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
        nq = q4_chunks(qw, hid)
        plan["layer_types"][FULL] = {
            "pool": experts + [
                # q_proj is the fused [q | gate] rows; the pool splits the halves
                {"op": "std_perm", "tensor": pre + "self_attn.q_proj.weight", "dst": L.POOL_Q, "chunk0": 0,
                 "nch": nq, "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.k_proj.weight", "dst": L.POOL_K,
                 "nch": q4_chunks(kvw, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.v_proj.weight", "dst": L.POOL_V,
                 "nch": q4_chunks(kvw, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.q_proj.weight", "dst": L.POOL_GATE, "chunk0": nq,
                 "nch": nq, "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.o_proj.weight", "dst": L.POOL_O,
                 "nch": q4_chunks(hid, qw), "in_dim": qw},
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CA_LNW, "cap": ELEM},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CA_POSTLN, "cap": ELEM},
                {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CA_META, "cap": hd * 2},
                {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CA_META + hd * 2, "cap": hd * 2},
                {"op": "put", "tensor": pre + "moe_router.weight", "dst": L.CA_RW, "cap": hid * E * 2},
                {"op": "put", "tensor": pre + "shared_expert_gate.weight", "dst": L.CA_SGW, "cap": ELEM},
            ],
        }
    return plan


# ---- the step program (what the driver runs per layer type), and the kernel sets that serve it
def programs(spec: ModelSpec) -> dict:
    L = layout(spec)
    out: dict = {
        "contexts": {}, "kernels": {}, "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": spec.vocab * 4,
                    "lmpool": L.LMHEAD_POOL_BYTES, "ptab": {"per_row": PTAB_ROW}},
    }
    out["contexts"]["ln"] = "ln/final.xclbin"
    out["contexts"]["lm"] = "lm_head_q8/final.xclbin"
    out["kernels"]["ln"] = {"context": "ln", "insts": "ln/insts.bin", "build": "ln"}
    out["kernels"]["lm"] = {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"}
    if spec.has_linear:
        args = ["pool", "xres", "consts", "state", "act"]
        check_buffer_args("lx", args)
        out["contexts"]["lx"] = "lx0/final.xclbin"
        out["kernels"]["lx0"] = {"context": "lx", "insts": "lx0/insts.bin", "build": "lx0"}
        out["kernels"]["lx1"] = {"context": "lx", "insts": "lx1/insts.bin", "patch": "moeroute2", "build": "lx1"}
        out["layer_types"][LINEAR] = {
            "buffers": {"consts": L.C_BYTES, "act": L.A_BYTES, "state": {"kind": "linear", "bytes": L.STATE_BYTES}},
            "program": [{"op": "run", "kernel": "lx0", "args": args},
                        {"op": "moeroute2", "kernel": "lx1", "act_off": L.A_ROUT},
                        {"op": "run", "kernel": "lx1", "args": args}],
        }
    if spec.has_full:
        args = ["pool", "xres", "consts", "state", "act", "ptab"]
        check_buffer_args("ax", args)
        out["contexts"]["ax"] = "ax0/final.xclbin"
        out["kernels"]["ax0"] = {"context": "ax", "insts": "ax0/insts.bin", "patch": "attnpos", "build": "ax0"}
        out["kernels"]["ax1"] = {"context": "ax", "insts": "ax1/insts.bin", "patch": "moeroute2", "build": "ax1"}
        out["layer_types"][FULL] = {
            "buffers": {"consts": L.CA_BYTES, "act": L.AA_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": "ax0", "args": args},
                        {"op": "moeroute2", "kernel": "ax1", "act_off": L.AA_ROUT},
                        {"op": "run", "kernel": "ax1", "args": args}],
        }
    return out


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_experts": spec.num_experts, "num_experts_per_tok": spec.experts_per_tok,
            "moe_intermediate_size": spec.moe_intermediate, "head_dim": spec.head_dim,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "layer_types": list(spec.layer_types)}


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    """The manifest's `layout` block for this family."""
    R = recipe(spec, max_ctx)
    L, C = R.layout, R.common
    return {
        "hidden": spec.hidden, "vocab": spec.vocab, "real_vocab": spec.real_vocab,
        "chunk_bytes": CHUNK, "pool_bytes": L.POOL_BYTES,
        "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": Q8_CHUNK,
        "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
        "rope_inv_freq": spec.rope_inv_freq(),
        "rout_idx_off": ROUT_IDX_OFF,
        "moe": {"experts": spec.num_experts, "topk": spec.experts_per_tok,
                "stripe": C.STRIPE, "up_bytes": C.UP_BYTES, "down_core": C.DOWN_PER_CORE * C.DOWN_BAND,
                "pool_down": L.POOL_DOWN, "share_up": L.POOL_SHARE_UP, "share_gate": L.POOL_SHARE_GATE,
                "share_down": L.POOL_SHARE_DOWN},
    }


def builds(spec: ModelSpec) -> dict[str, dict]:
    """name -> {design, build_dir, env}: the kernel sets export_qwen36_kernels.py builds (paths
    relative to open_kernels/designs)."""
    b: dict[str, dict] = {}
    if spec.has_linear:
        b["lx0"] = {"design": "layer_x/lx.py", "build_dir": "layer_x/build_lx0", "env": {"LX_PART": "0"}}
        b["lx1"] = {"design": "layer_x/lx.py", "build_dir": "layer_x/build_lx1", "env": {"LX_PART": "1"}}
    if spec.has_full:
        b["ax0"] = {"design": "layer_x/ax.py", "build_dir": "layer_x/build_ax0", "env": {"AX_PART": "0"}}
        b["ax1"] = {"design": "layer_x/ax.py", "build_dir": "layer_x/build_ax1", "env": {"AX_PART": "1"}}
    b["ln"] = {"design": "ln/ln.py", "build_dir": "ln/build", "env": {}}
    b["lm_head_q8"] = {"design": "lm_head_q8/lm_head_q8.py", "build_dir": "lm_head_q8/build_full",
                       "env": {"LMHEAD_N": str(spec.vocab), "LMHEAD_CORES": str(LIMITS["n_cols"])}}
    return b


# the design sources a build of this recipe depends on (for the build key), relative to open_kernels/
GEN_KERNELS = "designs/layer_x/gen_kernels.py"      # the design's kernel-TU generator (export_qwen36_kernels.py runs it per spec)
KERNEL_SOURCES = [
    "designs/layer_x/*.py", "designs/layer_x/*.cc", "designs/layer_x/*.h",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h", "designs/gemv_q4/gemv_q4_prep_k2048.cc",
    "designs/attn/*.cc", "designs/attn/*.h",
    "designs/dn_glue/*.cc", "designs/dn_glue/*.h", "designs/dn_post/*.cc",
    "designs/router/*.cc", "designs/router/*.h",
    "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py", "designs/lin_layer/ln_nr.cc",
    "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.cc", "designs/lm_head_q8/*.h",
    "include/vecmath.h", "ironutil.py", "build_design.py",
]
