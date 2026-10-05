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

import math
from dataclasses import dataclass, field

from .catalogue import LIMITS, OpRangeError, check_buffer_args, require
from .attnknobs import knobs as attn_knobs, probe_env  # noqa: F401  (probe_env: cache.py reads it off the family module)
from .spec import FULL, LINEAR, QUANT_FORMATS, ModelSpec

# ---- the q4_1 / q8 pool chunk formats (gemv_q4.h, lm_head_q8.h): format constants, not model ones
CHUNK = 5120                 # q4_1: 32 rows x 256 K (8192 values) + bf16 d, m per 32-block
CHUNK_VALUES = 8192
CHUNK_ROWS = 32
CHUNK_COLS = CHUNK_VALUES // CHUNK_ROWS   # 256: the k-tile the band law and std_perm both count in
Q8_CHUNK = 8704              # lm_head q8: 8192 int8 + 256 bf16 scales
ELEM = 4096                  # one act / x-stream element
BAND_ROWS = 64               # rows per GEMV band (one y element of 64 floats)
PER_CALL = 2                 # chunks per w element
CALL_BYTES = PER_CALL * CHUNK
# t2 (OPEN-QUANT-T2): 32 rows x 256 K of 2-bit ternary codes + a bf16 scale per (row, 128 K),
# 2176 B, unpadded, streamed ONE per w element. Two per element halved the core program's GEMV
# trip counts and overflowed the 27B's lx main core (LLVM unrolled the loops; then the second
# call per element cost 1.3 KB of argument setup). One per element keeps the q4_1 loop shapes,
# and probe 0a measured 2176 B elements at the same DMA floor. 2176 is not a whole number of
# 512 B DeltaNet state rows, so the S slices are 4 rows (2048 B) at a 2048 B stride inside the
# 2176 B elements (`dn_slice_bytes`; xcommon.dn_sequence's overlapping tap). It was padded to
# 2560 (5 rows) until 2026-10-03: 15% of every layer's weight stream was zeros.
T2_CHUNK = 2176
T2_PER_CALL = 1
MB = 1 << 20
POOL_BYTES = 512 * MB        # one layer's weight pool (fixed BO size; the recipe checks it fits)
PTAB_ROW = 1024              # the position record: [i32 pos | i32 nf | ... cos @512 | sin @640]
PTAB_COS, PTAB_SIN = 512, 640
ROUT_IDX_OFF = 1024          # int32 idx[topk] inside the router record (f32 probs first)
DN_RECORD_FLOATS = 512       # the DeltaNet per-head record [k | q | v | decay | beta | pad] (dnx.h)
AB_LANES = 32                # dn_glue.h's kV: the lanes of one alpha/beta W row and of the
                             # accumulator. A model with fewer value heads is ZERO-PADDED to
                             # this width rather than given a narrower vector type, so the
                             # W element stays 64 rows x 32 bf16 = 4 KB (see glue_ab_tile).


def q4_bytes(rows: int, cols: int) -> int:
    n = rows * cols
    if n % CHUNK_VALUES:
        raise OpRangeError(f"q4 tensor [{rows}, {cols}] is not a whole number of {CHUNK_VALUES}-value chunks")
    return n // CHUNK_VALUES * CHUNK


def q4_chunks(rows: int, cols: int) -> int:
    return q4_bytes(rows, cols) // CHUNK


def band_bytes(K: int) -> int:
    """One 64-row band of a K-wide standard-layout matrix: K/128 chunks."""
    return q4_bytes(BAND_ROWS, K)


def per_band(K: int) -> int:
    """Chunks in one band, the GEMV's runtime band law (gemv_q4_pool_group_rt). The kernel
    derives K back from it as 256 * per_band / rs, so this counts CHUNKS, never w elements:
    handing it the element count reads the activation table at half width.

    The count has to be EVEN. At rs = 2 chunk i of a band covers row half i % 2 and k-tile
    i // 2, two chunks per k-tile, so an odd count is a band no walk can consume. K = 2944 --
    the width GPT-OSS's container actually ships -- is the case that made this worth saying:
    band_bytes returns cleanly on it and hands back 23. The refusal is here rather than in a
    catalogue entry so OPEN_KERNELS_UNVALIDATED cannot soften it (OPEN-WIDTH-PAD)."""
    n = band_bytes(K) // CHUNK
    if n % 2:
        raise OpRangeError(f"a {K}-wide band is {n} chunks, an odd count: the rs=2 band law "
                           f"puts two chunks on every k-tile, so a band is always even. "
                           f"K must be a multiple of 256 (K={K} is {K % 256} over)")
    return n


# ---- the per-role weight format (OPEN-QUANT-Q8). A projection the container stores at q8
# is streamed as 16-row half-tiles of its chunks: the same 64-row band, twice the bytes,
# four half-tiles per k-tile instead of two chunks (designs/gemv_q4/gemv_q8.h).
def role_bytes(spec: ModelSpec, role: str, rows: int, cols: int) -> int:
    return q4_bytes(rows, cols) * (2 if spec.quant_of(role) == "q8" else 1)


def role_chunks(spec: ModelSpec, role: str, rows: int, cols: int) -> int:
    """Pool elements the projection occupies: q4_1 chunks, or q8 half-tiles (twice as many)."""
    return role_bytes(spec, role, rows, cols) // CHUNK


def quant_check(spec: ModelSpec, who: str) -> None:
    for fmt in spec.quant_map.values():
        if fmt not in QUANT_FORMATS:
            raise OpRangeError(f"{who}: quant={fmt!r}; the gemv templates read q4_1 chunks "
                               f"({CHUNK} B) and q8 chunks ({Q8_CHUNK} B) only")


def mixed_check(spec: ModelSpec, who: str, roles) -> None:
    """A container that MIXES weight formats puts both GEMV bodies on one main core, and
    16 KB of program memory only holds them because the q4_1 pair -- the band into a y
    element and the band into the act scratch -- is folded into ONE entry point with a
    runtime destination (`gemv_q4_gyms`, designs/*/gen_kernels.py). That fold covers the
    q4_1 side only. A spec that puts the dense FFN's up | gate bands at q8 while some
    other role is still q4_1 would need a SECOND fold on the q8 side, so it is refused by
    name rather than half-supported: put `ffn` back to q4_1, or move every role to q8.
    An all-q4_1 or an all-q8 spec never reaches this check's condition."""
    q8 = spec.q8_roles
    left = sorted(r for r in roles if spec.quant_of(r) != "q8")
    if "ffn" in q8 and left:
        raise OpRangeError(f"{who}: quant ffn=q8 with {left} still q4_1 -- a mixed-format main "
                           f"core folds the q4_1 GEMV pair into one entry (gemv_q4_gyms) and has "
                           f"no room for a second fold on the q8 side (16 KB program memory). "
                           f"Set ffn back to q4_1, or every role to q8")


def is_t2(spec: ModelSpec) -> bool:
    """Every projection at t2 (the only t2 form there is: one GEMV body per main core)."""
    return any(f == "t2" for f in spec.quant_map.values())


def t2_check(spec: ModelSpec, who: str, roles) -> None:
    if not is_t2(spec):
        return
    if any(spec.quant_of(r) != "t2" for r in roles):
        raise OpRangeError(f"{who}: quant t2 on some projections only -- the main core holds one GEMV "
                           f"body, so t2 is every projection or none")
    if not spec.hadamard:
        raise OpRangeError(f"{who}: quant t2 without spec.hadamard -- the t2 preps are the rotated-basis "
                           f"ones (OPEN-HADAMARD); no unrotated ternary model has been brought up")


def proj_op(spec: ModelSpec, role: str, tensor: str, dst: int, rows: int, cols: int,
            in_dim: int, chunk0: int | None = None) -> dict:
    """The pack op for one weight projection: `std_perm` at q4_1 (unchanged), `q8_perm` at
    q8 (twice the pool elements, 16-row half-tiles). `chunk0` is a SOURCE file-chunk offset
    in either format, so the fused [q | gate] split reads the same way."""
    q = spec.quant_of(role)
    op: dict = {"op": {"q8": "q8_perm", "t2": "t2_perm"}.get(q, "std_perm"), "tensor": tensor, "dst": dst}
    if chunk0 is not None:
        op["chunk0"] = chunk0
    op["nch"] = role_chunks(spec, role, rows, cols)
    op["in_dim"] = in_dim
    if q == "t2":
        op["chunk_bytes"] = T2_CHUNK      # the pool stride pools.cpp packs at: the kernels' w element
    return op


def require_gemv(spec: ModelSpec, role: str, K: int, rows_per_core: int, pc: int, rs: int = 2) -> None:
    """The GEMV point this projection needs, from the role's format. A q4_1 role asks
    exactly what it always asked; a q8 one asks the gemv_q8 template at rs 4."""
    if spec.quant_of(role) == "q8":
        require("gemv_q8", K=K, rs=4, rows_per_core=rows_per_core, per_call=pc)
    elif spec.quant_of(role) == "t2":
        require("gemv_t2", K=K, rs=rs, rows_per_core=rows_per_core, per_call=pc)
    else:
        require("gemv_q4", K=K, rs=rs, rows_per_core=rows_per_core, per_call=pc)


def tab_bytes(K: int) -> int:
    """gemv_q4_tab_bytes(K) = 2.25 K (gemv_tab.h)."""
    return 2 * K + K // 8 + K // 8


def roundup(n: int, m: int) -> int:
    return (n + m - 1) // m * m


def pad_width(w: int, n_cores: int) -> int:
    """The smallest width at or above w that a pool can actually be built at.

    Two rules have to hold at once and one rounding satisfies both, because
    BAND_ROWS * 8 = 512 is itself a multiple of the chunk's 256 columns:

      * dense.cores_for wants the width divisible by BAND_ROWS * n_cores, or the
        family drops to fewer cores (GPT-OSS's 2880 falls all the way to one);
      * q4_bytes wants rows * cols a whole number of 8192-value chunks, and
        band_bytes(2880) does not - it raises before any core count matters.

    At eight cores -- and at four -- the first rounding implies the second, because
    BAND_ROWS * 8 is 512 and BAND_ROWS * 4 is 256. Below that it does not: rounding
    2880 to BAND_ROWS * 1 left it at 2880, and to BAND_ROWS * 2 gave 2944, which is
    the worst answer available. 2944 passes band_bytes, and then per_band is 23 and
    pack.std_perm is non-injective. So the rounding is to lcm(BAND_ROWS * n_cores,
    CHUNK_COLS), which is the same 512 at eight cores and 256 at one.

    No shipped family needs this: every one of them is already a multiple of 512,
    so pad_width returns their own width unchanged. It exists for GPT-OSS, whose
    2880 satisfies neither rule.

    Padding a width is NOT the same as being able to build the family at it. The
    norm in particular must keep dividing by the model's own width - ln.h divides
    the sum of squares by the width it was COMPILED at (LN_N), so a 3072-wide norm
    over 2880 real channels scales every residual by sqrt(3072/2880), 3.3% high,
    on every layer. See OPEN-WIDTH-PAD for what this does and does not settle.
    """
    return roundup(w, math.lcm(BAND_ROWS * n_cores, CHUNK_COLS))


def ab_lanes(spec: ModelSpec) -> int:
    """Columns of the packed alpha / beta projection: the value-head count rounded up to
    dn_glue's accumulator width. 32 heads (every validated model) is itself."""
    return roundup(spec.lin_value_heads, AB_LANES)


def glue_fills(xn_elems: int, half_outer: bool) -> int:
    """DMA fills the dense glue's `side` channel issues per linear-attention dispatch, for an
    xn of `xn_elems` 4 KB halves. Accumulator-outer (lx.py's original walk) carries every half
    once per accumulator, with that accumulator's tiles for it: 4 fills a half. Half-outer
    carries the half once and then both accumulators' tiles: 3. Then `small` and the conv
    taps."""
    return (3 if half_outer else 4) * xn_elems + 2


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
    # element sizes: the norm helper's (HID*2) and the attention helper's (one KV row half)
    ELN: int = 0; E_A: int = 0
    # ffn="dense" only (the qwen35 composition): the FFN's pool block and the two extra act stages
    POOL_FFN_UP: int = 0; POOL_FFN_GATE: int = 0; POOL_FFN_DOWN: int = 0
    A_H: int = 0; A_OUT2: int = 0; AA_H: int = 0; AA_OUT2: int = 0
    # ... and the second down piece's output when the down GEMV is split along K (`down_split`)
    A_OUT2B: int = 0; AA_OUT2B: int = 0

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
    DN_PAD: int = 0                      # DN_SLICES * DN_ROWS: the padded S row count (dnx.h's kPad)


@dataclass(frozen=True)
class Linear:
    """lx.py's geometry."""
    QKV_PC: int; Z_PC: int; OUT_PC: int
    QKV_DIM: int; VW: int                # fused q|k|v rows; the value width (z, o, og)
    NCH: int; NHEAD: int; TILE: int; NT: int; AB_ELEMS: int; G: int; NG: int
    KEY_WIDTH: int; HEADS_PER_TILE: int; VALUE_TILE0: int
    RECORD_BYTES: int; O_HEAD_BYTES: int
    OUT_K: int; QKV_K: int
    OG_ELEMS: int = 0                    # 4 KB x-stream elements the og (bf16[VW]) arrives in
    XN_SIDE_ELEMS: int = 0               # 4 KB side elements the glue's xn copy arrives in
    # The alpha / beta projection's packed width and the rows of it one 4 KB side element
    # holds: 32 lanes x 64 rows for every published size up to 32 value heads, 64 x 32 for
    # the 27B's 48 (glue_ab_w.cc). And the glue's walk order: per accumulator then per xn
    # half (every size that fits the side channel that way), or per half then per
    # accumulator, which carries each half once -- 3 fills a half instead of 4.
    AB_LANES: int = 32
    AB_ROWS: int = 64
    GLUE_HALF_OUTER: bool = False


@dataclass(frozen=True)
class Attn:
    """ax.py's geometry."""
    NH: int; KVH: int; HD: int; ROT: int
    Q_PC: int; KV_PC: int; O_PC: int
    QW: int; KVW: int                    # q (and gate) width, k (and v) width
    O_K: int; QKV_K: int
    META_BYTES: int                      # [qn | kn] bf16, the first meta element
    HEAD_BYTES: int                      # one f32 head = one ain element
    # the attention helper's element is ONE cache-row half (attn.h): E_A = KVH*HD bf16 bytes,
    # so a q / k / v / gate element carries HPE = KVH/2 f32 heads and an og element HPO = KVH.
    # For the 27B (KVH 2) E_A is HEAD_BYTES and these are 1 / 2 / NH / KVH / NH/2.
    E_A: int = 0; HPE: int = 0; HPO: int = 0
    Q_AIN_ELEMS: int = 0; K_AIN_ELEMS: int = 0; OG_AOUT_ELEMS: int = 0
    OG_ELEMS: int = 0                    # 4 KB x-stream elements the og (bf16[QW]) arrives in
    # the fast attention path (recipes/attnknobs.py); the defaults are the single-core
    # kernel every family compiled before it existed
    VEXP: int = 0; MLS: int = 0; ACORES: int = 1; NHL: int = 0; RB: int = 1


@dataclass(frozen=True)
class Ffn:
    """The dense FFN tail (ffn="dense", the qwen35 composition): designs/dense/dx.py's
    steps 6-7 -- up | gate per band into `ms`, act, down against h -- on the layer_x fabric.
    The arithmetic is recipes/dense.py's `DenseGeometry`, at this family's widths."""
    FF: int; UP_PC: int; DOWN_PC: int
    MS_U: int; MS_G: int; MS_FLOATS: int
    XN_ELEMS: int; XM_ELEMS: int; H_ELEMS: int
    # The down GEMV's K pieces when FF's activation table does not fit a main core
    # (`down_split`): each piece is a whole number of h's f32 elements and runs as its own
    # GEMV into its own act region. Empty -- one GEMV over FF -- for every size that fits.
    DOWN_SPLIT: tuple = ()


@dataclass(frozen=True)
class Recipe:
    spec: ModelSpec
    layout: Layout
    common: Common
    linear: Linear | None
    attn: Attn | None
    max_ctx: int = 4096
    ffn: Ffn | None = None               # ffn="dense": the FFN tail's geometry
    kind: str = "moe"                    # "moe" | "dense": which tail the designs build
    q8: frozenset = frozenset()          # the roles streamed at q8 (OPEN-QUANT-Q8); empty is today
    ln_split: bool = False               # the dense norm helper streams its residual adds (`norm_split`)


def _check(spec: ModelSpec) -> None:
    if spec.family != "qwen36moe":
        raise OpRangeError(f"qwen36moe recipe given a {spec.family!r} spec")
    quant_check(spec, "qwen36moe")
    if spec.quant_of("experts") == "q8":
        raise OpRangeError("qwen36moe: a q8 routed expert -- the expert stripe and down laws are q4_1 "
                           "only, and no published model ships one")
    if spec.quant_of("shared") == "q8":
        raise OpRangeError("qwen36moe: a q8 shared expert -- it rides the routed experts' call sites "
                           "(one nine-slot loop, for program memory), so it runs at the routed format; "
                           "the packer re-quantizes it, as it did before OPEN-QUANT-Q8")
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
        require_gemv(spec, "linear", spec.hidden, spec.lin_qkv_dim // n, PER_CALL)
        require_gemv(spec, "linear", spec.hidden, spec.lin_value_width // n, PER_CALL)
        require_gemv(spec, "linear_out", spec.lin_value_width, spec.hidden // n, PER_CALL)
    if spec.has_full:
        require("attn", head_dim=spec.head_dim, num_heads=spec.num_heads, num_kv_heads=spec.num_kv_heads,
                rotary_dim=spec.rotary_dim, rope_theta=spec.rope_theta, qk_norm=spec.qk_norm,
                attn_gate=spec.attn_gate)
        require_gemv(spec, "attn", spec.hidden, spec.attn_q_width // n, PER_CALL)
        require_gemv(spec, "attn", spec.hidden, spec.attn_kv_width // n, PER_CALL)
        require_gemv(spec, "attn", spec.attn_q_width, spec.hidden // n, PER_CALL)
    require("gemv_q4", K=spec.hidden, rs=4, rows_per_core=BAND_ROWS, per_call=PER_CALL)   # expert up / gate halves


# ---- the FFN tail's L1 budget (ffn="dense"): a main core's 64 KB data memory less IRON's
# bookkeeping, the same numbers recipes/dense.py checks -- plus the DeltaNet scratch, which
# the dense-only designs do not carry.
L1_BUDGET = 60 * 1024
STACK = 0x1800
DN_SCRATCH_FLOATS = 1280               # `ds` (dnx.h)
FFN_MS_FLOATS = 2 * BAND_ROWS          # `ms` for the dense tail: u[64] | g[64]


def xh_l1(spec: ModelSpec) -> int:
    """A rotated-basis (Hadamard) model's extra main-core L1: wht.h's fp32 scratch block and
    the attention output's sign bits (gen_kernels.xh_signs_h). 0 for every other model."""
    hd = spec.hadamard
    return 0 if not hd else 1024 * 4 + len(hd["og_signs"]) // 8


def core_l1(tab: int, ms_floats: int, ds_floats: int, pc: int = PER_CALL, extra: int = 0,
            chunk: int = CHUNK) -> int:
    """A main core's L1 bytes for a given scratch layout.

    Every main core carries the same six things and nothing else
    (`designs/layer_x/xcommon.py core_buffers`, and the fifo depths in `lx.py` / `dx.py`):
    the activation table, the `ms` and `ds` scratch buffers, then depth-2 fifos for the
    weight elements, the x elements and the y elements, over a 0x1800 stack. Both tails
    compute it here rather than each restating the sum, for the reason the band law is
    shared -- two copies of a budget drift, and the one that drifts is the one no shipped
    model exercises."""
    return (tab + ms_floats * 4 + ds_floats * 4
            + 2 * pc * chunk + 2 * ELEM + 2 * BAND_ROWS * 4 + STACK + extra)


def per_call(spec: ModelSpec, ffn: str = "moe") -> int:
    """Chunks per weight element. The MoE tail is frozen at 2 (the shipped 27B kernels);
    the dense tail takes 1 when the widest activation table leaves no room for two 10 KB
    elements beside the x elements and the two scratch buffers (the 9B: a 12288-wide down
    table is 27648 B, so 10 KB elements overflow by 7168 B)."""
    if ffn != "dense":
        return PER_CALL
    wide = kwide(spec, ffn)
    ds = DN_SCRATCH_FLOATS if spec.has_linear else 0
    if is_t2(spec):
        if core_l1(tab_bytes(wide), FFN_MS_FLOATS, ds, T2_PER_CALL, xh_l1(spec), T2_CHUNK) <= L1_BUDGET:
            return T2_PER_CALL
        raise OpRangeError(f"qwen35 t2: two {T2_CHUNK} B chunks per w element do not fit a core's L1 "
                           f"beside a {wide}-wide table")
    for pc in (2, 1):
        if core_l1(tab_bytes(wide), FFN_MS_FLOATS, ds, pc, xh_l1(spec)) <= L1_BUDGET:
            return pc
    raise OpRangeError(f"qwen35: a {wide}-wide activation table does not leave room for the streams "
                       f"in a core's L1 ({tab_bytes(wide)} B of table, {L1_BUDGET} B budget)")


def kwide(spec: ModelSpec, ffn: str = "moe") -> int:
    """The widest K a main core prepares a table for. The MoE keeps the expert hidden's table
    beside xm's (H_TAB_OFF); the dense tail runs its down GEMV after every up | gate band, so
    h's table replaces xm's and FF joins the max instead (designs/dense/dx.py) -- or, when
    the down GEMV is split along K, its widest piece."""
    wide = max(spec.hidden, spec.lin_value_width if spec.has_linear else 0,
               spec.attn_q_width if spec.has_full else 0)
    if ffn != "dense":
        return wide
    return max(wide, *(down_split(spec) or (spec.intermediate,)))


def down_split(spec: ModelSpec) -> tuple[int, ...]:
    """The dense down GEMV's K pieces, or () when FF's table fits a main core whole.

    The down GEMV reduces over FF, and its activation table is 2.25 FF bytes. At the 27B's
    FF 17408 that is 39 168 B, and the core is over its L1 even at 5 KB weight elements. So
    the reduction runs in two pieces, each a GEMV of its own with its own table, results to
    two act regions that the norm helper adds. The pieces cut at an f32 element of h (1024
    values, what `dense_prep_f32` prepares per call), the first the largest such cut at or
    below FF / 2. Nothing is repacked: inside a band, pool chunk c covers k-tile c / 2
    (gemv_q4.h's band law), so the first 2 K0 / 256 chunks of every FF-wide band ARE a
    K0-wide band and the rest a (FF - K0)-wide one, and each piece is a strided DMA tap over
    the pool the packer already writes."""
    if not spec.intermediate:
        return ()
    base = max(spec.hidden, spec.lin_value_width if spec.has_linear else 0,
               spec.attn_q_width if spec.has_full else 0)
    ds = DN_SCRATCH_FLOATS if spec.has_linear else 0
    ff = spec.intermediate
    if core_l1(tab_bytes(max(base, ff)), FFN_MS_FLOATS, ds, 1, xh_l1(spec)) <= L1_BUDGET:
        return ()
    e = ELEM // 4
    k0 = ff // 2 // e * e
    pieces = (k0, ff - k0)
    if not k0 or ff % e or any(core_l1(tab_bytes(max(base, k)), FFN_MS_FLOATS, ds, 1, xh_l1(spec)) > L1_BUDGET
                               for k in pieces):
        raise OpRangeError(f"qwen35: an FF of {ff} does not fit a main core's L1 in two pieces of "
                           f"whole f32 elements ({pieces}); a three-way split is not implemented")
    return pieces


# The dense norm helper (Tile(0, 3)): the residual + norm stage holds [x0 x1 w a0 a1] and one
# output element at once, six ELN-byte elements over a 0x1800 stack. At HID 4096 that is
# 55 296 B; at the 27B's 5120 it is 67 584 B against the core's 64 KB.
NORM_L1 = 64 * 1024


def norm_split(spec: ModelSpec) -> bool:
    """True when the norm helper's fused residual stage does not fit its core. The split
    form streams the residual adds half by half (`ln_add2` / `ln_add3`), sends the sum to DDR
    and normalizes it on the way back through `ln_nr`: never more than four elements held."""
    return 6 * spec.hidden * 2 + STACK > NORM_L1


def common(spec: ModelSpec, ffn: str = "moe") -> Common:
    if ffn == "dense":
        return _common_dense(spec)
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
    spp = ff // 128
    if not spp or spp > n or n % spp:
        raise OpRangeError(
            f"moe: an expert width of {ff} is {spp} stripes per projection over {n} cores. "
            f"`moe_sequence` splits ONE 128-row stripe across `n // stripes` cores -- its "
            f"`c // cps` and `c % cps` -- so the stripe count has to divide the core count. "
            f"Here each core would own {spp / n:g} stripes instead and `cps` is {n // spp if spp else 0}. "
            f"That is a different host sequence, not a different constant (OPEN-MOE-WIDE-FF).")
    wide = max(hid, spec.lin_value_width if spec.has_linear else 0, spec.attn_q_width if spec.has_full else 0)
    tab = tab_bytes(wide)
    h_tab = tab_bytes(hid)
    if h_tab + tab_bytes(ff) > tab:
        raise OpRangeError(
            f"moe: the hidden h's table does not fit past xm's in the core scratch -- "
            f"{h_tab} B for K={hid} plus {tab_bytes(ff)} B for K={ff} against a {tab} B "
            f"reservation. `tab` is sized for the WIDEST K a core prepares, which collapses "
            f"to the hidden itself on a family with neither linear-attention nor "
            f"full-attention layers (OPEN-MOE-WIDE-FF).")
    l1 = core_l1(tab, ms_floats, DN_SCRATCH_FLOATS)
    if l1 > L1_BUDGET:
        raise OpRangeError(
            f"moe: a main core's scratch comes to {l1} B against a {L1_BUDGET} B budget, "
            f"{l1 - L1_BUDGET} B over -- {tab} B of table, {ms_floats * 4} B of ms, "
            f"{DN_SCRATCH_FLOATS * 4} B of ds. Program memory only aiecc can measure, but "
            f"L1 is arithmetic, and until now nothing computed it for this tail.")
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
        DN_DIM=dn_dim, DN_PAD=dn_slices * dn_rows,
    )


def dn_slice_bytes(C: Common) -> int:
    """The bytes of S one streamed slice carries: DN_ROWS whole rows. Equal to the w element
    for every q4_1 / q8 element size (10 KB, 5 KB); smaller for t2's 2176 B element, whose
    4.25 rows round down to 4. Then the slices sit at this stride in the state buffer and each
    element also carries the next slice's first CALL_BYTES - this bytes, which nothing reads
    (xcommon.dn_sequence), and the state buffer ends in `dn_state_tail` bytes so the last
    head's last element stays inside it."""
    return C.DN_ROWS * C.DN_DIM * 4


def dn_state_tail(C: Common) -> int:
    return C.CALL_BYTES - dn_slice_bytes(C) if C.DN_DIM else 0


def _dn_geometry(spec: ModelSpec, call_bytes: int, n: int):
    """The DeltaNet slicing for a given weight-element size: S rows per element, slices per
    head, the padded row count dnx.h's kPad must be, heads per core. At 10 KB elements and
    dim 128 this is the 27B's 20 / 7 / 140 / 4; at 5 KB it is 10 / 13 / 130 / 4; at t2's
    2176 B it is 4 / 32 / 128 / 6 (4.25 rows round down: see dn_slice_bytes)."""
    dim = spec.lin_value_dim if spec.has_linear else 0
    if not dim:
        return 0, 0, 0, 0, 0
    rows = call_bytes // (dim * 4)
    slices = roundup(dim, rows) // rows
    return rows, slices, slices * rows, spec.lin_value_heads // n, dim


def _common_dense(spec: ModelSpec) -> Common:
    """The main-core geometry with the MoE tail replaced by the dense FFN (qwen35). Every
    DeltaNet field is `common()`'s law at this element size; the MoE-only fields are zero."""
    n = LIMITS["n_cols"]
    hid, ff = spec.hidden, spec.intermediate
    pc = per_call(spec, "dense")
    tile = T2_CHUNK if is_t2(spec) else CHUNK
    call_bytes = pc * tile
    wide = kwide(spec, "dense")
    dn_rows, dn_slices, dn_pad, dn_heads_pc, dn_dim = _dn_geometry(spec, call_bytes, n)
    return Common(
        N_CORES=n, HID=hid, FF=ff, NE=0, NX=0,
        TILE=tile, PER_CALL=pc, CALL_BYTES=call_bytes,
        STRIPE=0, HALF=0, PAIR=2 * tile, DOWN_BAND=0, UP_BYTES=0, DOWN_PER_CORE=0,
        BAND_ROWS=BAND_ROWS, BAND16=band_bytes(hid), BAND32=band_bytes(2 * hid), N_HDR=0,
        MS_FLOATS=FFN_MS_FLOATS, DS_FLOATS=DN_SCRATCH_FLOATS if dn_dim else 0,
        TAB_BYTES=tab_bytes(wide), H_TAB_OFF=0, KWIDE=wide,
        MS_RW=0, MS_XR=0, MS_ACC=0, MS_U=0, MS_G=BAND_ROWS, MS_YD=0,
        ROWS_PC=hid // n, HID_PC=ff // n,
        STRIPES_PER_PROJ=0, CORES_PER_STRIPE=0,
        UP_ELEMS=0, DOWN_ELEMS=0, OUT_ELEMS=0, W_ELEMS=0,
        DN_ROWS=dn_rows, DN_SLICES=dn_slices, DN_HEADS_PC=dn_heads_pc, DN_DIM=dn_dim, DN_PAD=dn_pad,
    )


def ffn_geometry(spec: ModelSpec) -> Ffn:
    n = LIMITS["n_cols"]
    hid, ff = spec.hidden, spec.intermediate
    return Ffn(
        FF=ff, UP_PC=ff // BAND_ROWS // n, DOWN_PC=hid // BAND_ROWS // n,
        MS_U=0, MS_G=BAND_ROWS, MS_FLOATS=FFN_MS_FLOATS,
        XN_ELEMS=roundup(hid * 2, ELEM) // ELEM, XM_ELEMS=roundup(hid * 2, ELEM) // ELEM,
        H_ELEMS=roundup(ff * 4, ELEM) // ELEM,
        DOWN_SPLIT=down_split(spec),
    )


def layout(spec: ModelSpec, max_ctx: int = 4096, ffn: str = "moe") -> Layout:
    if ffn == "dense":
        return _layout_dense(spec, max_ctx)
    n = LIMITS["n_cols"]
    hid, E, ff = spec.hidden, spec.num_experts, spec.moe_intermediate
    C = common(spec)
    rw_bytes = hid * E * 2                            # router W bf16 [HID, E]
    kv = {}

    # ---- consts, linear layer: [lnw][glue side minus xn][nw][postln][router W][sgw][out_proj q4]
    vw = spec.lin_value_width if spec.has_linear else 0
    nch = spec.lin_qkv_dim if spec.has_linear else 0
    if spec.has_linear:
        alpha = hid * ab_lanes(spec) * 2              # alpha / beta projections bf16 [HID, lanes]
        side = _Alloc()
        side.add("alpha", alpha)
        side.add("beta", alpha)
        side.add("small", ELEM)                       # [a f32[heads] | dt_bias f32[heads] | pad]
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
        q.add("qkv", role_bytes(spec, "linear", nch, hid))
        q.add("z", role_bytes(spec, "linear", vw, hid))
        kv.update(POOL_QKV=q.off["qkv"], POOL_Z=q.off["z"])
        end = max(end, q.n)
    else:
        kv.update(POOL_QKV=0, POOL_Z=0)
    if spec.has_full:
        q = _Alloc(); q.n = proj0
        q.add("q", role_bytes(spec, "attn", qw, hid))
        q.add("k", role_bytes(spec, "attn", kvw, hid))
        q.add("v", role_bytes(spec, "attn", kvw, hid))
        q.add("gate", role_bytes(spec, "attn", qw, hid))
        q.add("o", role_bytes(spec, "attn", hid, qw))
        kv.update(POOL_Q=q.off["q"], POOL_K=q.off["k"], POOL_V=q.off["v"], POOL_GATE=q.off["gate"], POOL_O=q.off["o"])
        end = max(end, q.n)
    else:
        kv.update(POOL_Q=0, POOL_K=0, POOL_V=0, POOL_GATE=0, POOL_O=0)
    # The BO is the shipped 512 MB for everything that fits it, byte for byte; a q8 variant
    # of the same shape needs more and rounds up to the next MB (OPEN-QUANT-Q8).
    kv["POOL_BYTES"] = max(POOL_BYTES, roundup(end, MB))

    # ---- lm_head q8: 128-row bands of HID. The pool BO holds a whole number of bands per core
    # (the closed engine's size, 517 MB for the 27B); the design streams exactly `bands`.
    band = 128 * hid // CHUNK_VALUES * Q8_CHUNK
    bands = spec.vocab // 128
    kv.update(KV_ROW=kv_row, PTAB_ROW=PTAB_ROW, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * PTAB_ROW,
              LMHEAD_POOL_BYTES=roundup(roundup(bands, n) * band, MB), LMHEAD_BAND_BYTES=band, LMHEAD_BANDS=bands,
              ELN=hid * 2, E_A=kv_row // 2)
    return Layout(**kv)


def _layout_dense(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    """The qwen35 composition's byte layouts: the MoE recipe's DeltaNet / attention regions
    with the router and MoE ones dropped, and the dense recipe's FFN regions added (an `h`
    stage and a second block output per layer type). Element sizes come from the widths, as
    recipes/dense.py sizes them: the norm helper's ELN = HID*2 (8 KB at HID 4096, so the
    split ln_y / ln_xn entries), the attention helper's E_A = one KV row half."""
    n = LIMITS["n_cols"]
    hid, ff = spec.hidden, spec.intermediate
    C, F = _common_dense(spec), ffn_geometry(spec)
    eln = hid * 2
    kv: dict[str, int] = {}
    vw = spec.lin_value_width if spec.has_linear else 0
    nch = spec.lin_qkv_dim if spec.has_linear else 0
    og_lin = roundup(vw * 2, ELEM) // ELEM * ELEM if vw else 0
    if spec.has_linear:
        alpha = hid * ab_lanes(spec) * 2              # the alpha / beta projections, bf16 [HID, lanes]
        side = _Alloc()
        side.add("alpha", alpha)
        side.add("beta", alpha)
        side.add("small", ELEM)                       # [a f32[heads] | dt_bias f32[heads] | pad]
        side.add("conv", spec.conv_kernel * nch * 2)
        glue_side = side.n
        c = _Alloc()
        c.add("lnw", eln)
        c.add("side", glue_side)
        c.add("nw", ELEM)
        c.add("postln", eln)
        # the region is the tensor: unlike the MoE's, nothing was captured at twice the size
        c.add("wout", role_bytes(spec, "linear_out", hid, vw))
        kv.update(C_LNW=c.off["lnw"], C_SIDE=c.off["side"], C_NW=c.off["nw"], C_POSTLN=c.off["postln"],
                  C_RW=0, C_SGW=0, C_WOUT=c.off["wout"], C_BYTES=c.n,
                  GLUE_SIDE_BYTES=glue_side, SIDE_ALPHA=side.off["alpha"], SIDE_BETA=side.off["beta"],
                  SIDE_SMALL=side.off["small"], SIDE_CONV=side.off["conv"])
        a = _Alloc()
        a.add("xn", F.XN_ELEMS * ELEM)
        a.add("qkv", nch * 4)
        a.add("z", vw * 4)
        a.add("vec", spec.lin_value_heads * DN_RECORD_FLOATS * 4)
        a.add("o", vw * 4)
        a.add("og", og_lin)
        a.add("out", hid * 4)
        a.add("res", hid * 4)
        a.add("xm", F.XM_ELEMS * ELEM)
        a.add("h", F.H_ELEMS * ELEM)
        a.add("out2", hid * 4)
        if F.DOWN_SPLIT:
            a.add("out2b", hid * 4)
        kv.update(A_XN=a.off["xn"], A_QKV=a.off["qkv"], A_Z=a.off["z"], A_VEC=a.off["vec"], A_O=a.off["o"],
                  A_OG=a.off["og"], A_OUT=a.off["out"], A_RES=a.off["res"], A_XM=a.off["xm"],
                  A_ROUT=0, A_HP=0, A_H=a.off["h"], A_OUT2=a.off["out2"], A_OUT2B=a.off.get("out2b", 0),
                  A_BYTES=roundup(a.n, ELEM))
        s_head = C.DN_PAD * C.DN_DIM * 4
        s_off = (spec.conv_kernel - 1) * nch * 2
        kv.update(S_ROWS=C.DN_PAD, S_HEAD_BYTES=s_head, STATE_S_OFF=s_off,
                  STATE_BYTES=s_off + spec.lin_value_heads * s_head + dn_state_tail(C))
    else:
        kv.update({k: 0 for k in ("C_LNW", "C_SIDE", "C_NW", "C_POSTLN", "C_RW", "C_SGW", "C_WOUT", "C_BYTES",
                                  "GLUE_SIDE_BYTES", "SIDE_ALPHA", "SIDE_BETA", "SIDE_SMALL", "SIDE_CONV",
                                  "A_XN", "A_QKV", "A_Z", "A_VEC", "A_O", "A_OG", "A_OUT", "A_RES", "A_XM",
                                  "A_ROUT", "A_HP", "A_H", "A_OUT2", "A_BYTES", "S_ROWS", "S_HEAD_BYTES",
                                  "STATE_S_OFF", "STATE_BYTES")})

    e_a, kv_row = 0, 0
    if spec.has_full:
        qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
        e_a, kv_row = kvw * 2, 4 * kvw
        og_att = roundup(qw * 2, ELEM) // ELEM * ELEM
        c = _Alloc()
        c.add("lnw", eln)
        c.add("postln", eln)
        meta = max(2048, e_a)                          # [qn bf16 HD @0 | kn @HD*2], one helper element
        c.add("meta", meta)
        if 2 * hd * 2 > meta:
            raise OpRangeError(f"attn: qn | kn ({2 * hd * 2} B) do not fit the {meta} B meta element")
        kv.update(CA_LNW=c.off["lnw"], CA_POSTLN=c.off["postln"], CA_META=c.off["meta"], CA_RW=0, CA_SGW=0,
                  CA_BYTES=c.n)
        a = _Alloc()
        a.add("xn", F.XN_ELEMS * ELEM)
        a.add("qg", 2 * qw * 4)                        # q | gate f32
        a.add("kvn", 2 * kvw * 4)                      # k | v f32
        a.add("og", og_att)
        a.add("out", hid * 4)
        a.add("res", hid * 4)
        a.add("xm", F.XM_ELEMS * ELEM)
        a.add("h", F.H_ELEMS * ELEM)
        a.add("out2", hid * 4)
        if F.DOWN_SPLIT:
            a.add("out2b", hid * 4)
        kv.update(AA_XN=a.off["xn"], AA_QG=a.off["qg"], AA_KVN=a.off["kvn"], AA_OG=a.off["og"],
                  AA_OUT=a.off["out"], AA_RES=a.off["res"], AA_XM=a.off["xm"], AA_ROUT=0, AA_HP=0,
                  AA_H=a.off["h"], AA_OUT2=a.off["out2"], AA_OUT2B=a.off.get("out2b", 0),
                  AA_BYTES=roundup(a.n, ELEM))
    else:
        kv.update({k: 0 for k in ("CA_LNW", "CA_POSTLN", "CA_META", "CA_RW", "CA_SGW", "CA_BYTES", "AA_XN",
                                  "AA_QG", "AA_KVN", "AA_OG", "AA_OUT", "AA_RES", "AA_XM", "AA_ROUT", "AA_HP",
                                  "AA_H", "AA_OUT2", "AA_BYTES")})

    # ---- the layer pool: the FFN first (both layer types see it at the same offsets), then
    # the per-type projections, which overlap because a layer is only ever one type.
    p = _Alloc()
    p.add("up", role_bytes(spec, "ffn", ff, hid))
    p.add("gate", role_bytes(spec, "ffn", ff, hid))
    p.add("down", role_bytes(spec, "ffn", hid, ff))
    proj0 = p.n
    kv.update(POOL_FFN_UP=p.off["up"], POOL_FFN_GATE=p.off["gate"], POOL_FFN_DOWN=p.off["down"],
              POOL_DOWN=0, POOL_SHARE_UP=0, POOL_SHARE_GATE=0, POOL_SHARE_DOWN=0)
    end = proj0
    if spec.has_linear:
        q = _Alloc(); q.n = proj0
        q.add("qkv", role_bytes(spec, "linear", nch, hid))
        q.add("z", role_bytes(spec, "linear", vw, hid))
        kv.update(POOL_QKV=q.off["qkv"], POOL_Z=q.off["z"])
        end = max(end, q.n)
    else:
        kv.update(POOL_QKV=0, POOL_Z=0)
    if spec.has_full:
        q = _Alloc(); q.n = proj0
        q.add("q", role_bytes(spec, "attn", qw, hid))
        q.add("k", role_bytes(spec, "attn", kvw, hid))
        q.add("v", role_bytes(spec, "attn", kvw, hid))
        q.add("gate", role_bytes(spec, "attn", qw, hid))
        q.add("o", role_bytes(spec, "attn", hid, qw))
        kv.update(POOL_Q=q.off["q"], POOL_K=q.off["k"], POOL_V=q.off["v"], POOL_GATE=q.off["gate"],
                  POOL_O=q.off["o"])
        end = max(end, q.n)
    else:
        kv.update(POOL_Q=0, POOL_K=0, POOL_V=0, POOL_GATE=0, POOL_O=0)
    kv["POOL_BYTES"] = roundup(end, MB)

    ptab_row = max(PTAB_ROW, e_a)
    if ptab_row != e_a:
        # ax.py acquires ONE element for the record, as the dense design did until
        # recipes/dense.py's _ptab_check. A wider record would leave the rest of it in
        # the stream to be read as q. No MoE geometry reaches this; say so rather than
        # let the next one find out on hardware.
        raise OpRangeError(f"qwen36moe: a {ptab_row}-byte position record does not fit one "
                           f"{e_a}-byte attention element (see recipes/dense.py _ptab_check)")
    band = 128 * hid // CHUNK_VALUES * Q8_CHUNK
    bands = spec.vocab // 128
    kv.update(KV_ROW=kv_row, PTAB_ROW=ptab_row, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row,
              PTAB_BYTES=max_ctx * ptab_row,
              LMHEAD_POOL_BYTES=roundup(roundup(bands, n) * band, MB), LMHEAD_BAND_BYTES=band,
              LMHEAD_BANDS=bands, ELN=eln, E_A=e_a)
    return Layout(**kv)


def linear(spec: ModelSpec) -> Linear | None:
    if not spec.has_linear:
        return None
    n = LIMITS["n_cols"]
    nch, vw = spec.lin_qkv_dim, spec.lin_value_width
    tile = 1024                                     # dn_glue's channel tile
    key_width = spec.lin_key_heads * spec.lin_key_dim
    lanes = ab_lanes(spec)
    xn_elems = roundup(spec.hidden * 2, ELEM) // ELEM
    return Linear(
        AB_LANES=lanes, AB_ROWS=ELEM // (2 * lanes),
        GLUE_HALF_OUTER=glue_fills(xn_elems, False) > LIMITS["shim_fills"],
        QKV_PC=nch // BAND_ROWS // n, Z_PC=vw // BAND_ROWS // n, OUT_PC=spec.hidden // BAND_ROWS // n,
        QKV_DIM=nch, VW=vw,
        NCH=nch, NHEAD=spec.lin_value_heads, TILE=tile, NT=nch // tile,
        AB_ELEMS=spec.hidden * ab_lanes(spec) * 2 // ELEM, G=tile, NG=vw // tile,
        KEY_WIDTH=key_width, HEADS_PER_TILE=tile // spec.lin_value_dim, VALUE_TILE0=2 * key_width // tile,
        RECORD_BYTES=DN_RECORD_FLOATS * 4, O_HEAD_BYTES=spec.lin_value_dim * 4,
        OUT_K=vw, QKV_K=spec.hidden,
        OG_ELEMS=roundup(vw * 2, ELEM) // ELEM,
        XN_SIDE_ELEMS=roundup(spec.hidden * 2, ELEM) // ELEM,
    )


def attn(spec: ModelSpec) -> Attn | None:
    if not spec.has_full:
        return None
    n = LIMITS["n_cols"]
    qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
    e_a = kvw * 2                                 # one cache-row half, attn.h's fifo element
    hpe, hpo = e_a // (hd * 4), e_a // (hd * 2)   # f32 heads per q/k/v/gate element; bf16 heads per og element
    A = attn_knobs(spec, spec.num_heads, hpo)
    return Attn(
        NH=spec.num_heads, KVH=spec.num_kv_heads, HD=hd, ROT=spec.rotary_dim,
        Q_PC=qw // BAND_ROWS // n, KV_PC=kvw // BAND_ROWS // n, O_PC=spec.hidden // BAND_ROWS // n,
        QW=qw, KVW=kvw, O_K=qw, QKV_K=spec.hidden,
        META_BYTES=2 * hd * 2, HEAD_BYTES=hd * 4,
        E_A=e_a, HPE=hpe, HPO=hpo,
        Q_AIN_ELEMS=spec.num_heads // hpe, K_AIN_ELEMS=spec.num_kv_heads // hpe,
        OG_AOUT_ELEMS=spec.num_heads // hpo,
        OG_ELEMS=roundup(qw * 2, ELEM) // ELEM,
        VEXP=A.VEXP, MLS=A.MLS, ACORES=A.ACORES, NHL=A.NHL, RB=A.RB,
    )


def recipe(spec: ModelSpec, max_ctx: int = 4096, ffn: str = "moe") -> Recipe:
    if ffn != "dense":
        _check(spec)
    return Recipe(spec=spec, layout=layout(spec, max_ctx, ffn), common=common(spec, ffn), linear=linear(spec),
                  attn=attn(spec), max_ctx=max_ctx, ffn=ffn_geometry(spec) if ffn == "dense" else None,
                  kind=ffn, q8=spec.q8_roles, ln_split=ffn == "dense" and norm_split(spec))


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
                              "ops": [{"op": "lmhead_q8", "tensor": "lm_head.weight", "chunk_bytes": Q8_CHUNK, "in_dim": hid, "dst": 0}]},
                  "embed": {"tensor": "model.embed_tokens.weight", "dim": hid},
                  "norm": {"tensor": "model.norm.weight", "bytes": hid * 2}}
    if spec.has_linear:
        vw, nch = spec.lin_value_width, spec.lin_qkv_dim
        side = L.C_SIDE
        plan["layer_types"][LINEAR] = {
            "pool": experts + [
                proj_op(spec, "linear", pre + "linear_attn.qkv_proj.weight", L.POOL_QKV, nch, hid, hid),
                proj_op(spec, "linear", pre + "self_attn.gate_proj.weight", L.POOL_Z, vw, hid, hid),
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
                proj_op(spec, "linear_out", pre + "linear_attn.ssm_out_proj.weight", L.C_WOUT,
                        hid, vw, vw),
            ],
        }
    if spec.has_full:
        qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
        nq = q4_chunks(qw, hid)
        plan["layer_types"][FULL] = {
            "pool": experts + [
                # q_proj is the fused [q | gate] rows; the pool splits the halves
                proj_op(spec, "attn", pre + "self_attn.q_proj.weight", L.POOL_Q, qw, hid, hid, chunk0=0),
                proj_op(spec, "attn", pre + "self_attn.k_proj.weight", L.POOL_K, kvw, hid, hid),
                proj_op(spec, "attn", pre + "self_attn.v_proj.weight", L.POOL_V, kvw, hid, hid),
                proj_op(spec, "attn", pre + "self_attn.q_proj.weight", L.POOL_GATE, qw, hid, hid, chunk0=nq),
                proj_op(spec, "attn", pre + "self_attn.o_proj.weight", L.POOL_O, hid, qw, qw),
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


# ---- the block prefill route (OPEN-PREFILL-BATCH): T tokens through a layer's projections
# as whole-array GEMM dispatches (designs/gemm_q4_prefill), the DeltaNet / attention / router
# stages on the host between them, the MoE block still per token -- on its own dispatch
# (designs/layer_x/mx.py), because lx1 / ax1 are the second half of a per-token core program
# and cannot run alone.
GEMM_T = 256          # gemm_q4_prefill.py's GQP_T: a multiple of tile_n * 8 columns
# moe_batch.py's token slots per expert visit, in bf16 MAC tiles of 8: the number of an
# expert's tokens that one streaming of its 1.97 MB serves. The driver reads it back out of
# the manifest, so the design and the host cannot drift apart, and mb_x / mb_h / mb_y below
# scale with it.
#
# It stays at 8 although 16 builds and passes (.claude/plans/prefill-parity.md, B(1)): the
# kernel turned out to be core-bound, not DMA-bound -- at 16 slots its weight stream alone is
# 0.86-0.91 ms whatever the width and the whole dispatch is already 1.105 ms at 8 -- so 16
# bought 4 % of the expert stage and nothing of the prefill, in exchange for doubling
# mb_x / mb_h / mb_y and spending the core's last 4 KB of L1. Raise it once the core rate is
# up (the plan's B(3)); the design and the driver are ready for it.
MB_NT = 8
ATTN_LMAX = 4096      # the widest attention GEMM stream (rows of window); a longer window is chunked on the host
# The projections the route streams out of the layer's own packed bytes, which the GEMM can
# read only at q4_1. `linear_out` is not among them: at q8 the route packs it as the exact
# sum of two q4_1 halves (a `from: pack` weight), which is what lets a Qwen3.5 container --
# whose ssm_out_proj ships q8 -- have a route at all.
GEMM_ROLES = ("attn", "linear", "shared")
DENSE_GEMM_ROLES = ("attn", "linear", "ffn")    # ... and with a dense FFN (Qwen3.5)


def _op_index(ops: list[dict], suffix: str, chunk0: int | None = None) -> int:
    """The position of the pack op for `suffix` (the fused q_proj appears twice; chunk0 picks)."""
    hits = [i for i, o in enumerate(ops) if o.get("tensor", "").endswith(suffix)
            and (chunk0 is None or o.get("chunk0", 0) == chunk0)]
    if len(hits) != 1:
        raise OpRangeError(f"gemm route: {len(hits)} pack ops end in {suffix!r}, want one")
    return hits[0]


def gemm_route(spec: ModelSpec, ffn: str = "moe", plan: dict | None = None) -> dict | None:
    """Per layer type the `gemm_block` the driver reads, plus the contexts / kernels / globals /
    builds the route adds. None when a projection is streamed at q8: the GEMM dequantises the
    q4_1 band law only, and the sequential path is then exactly what it was.

    `ffn="dense"` is Qwen3.5 (recipes/qwen35.py): the same linear / full halves, and in place of
    the router, the shared expert and the routed experts an `ffn_program` -- up|gate then down,
    the shared expert's two GEMMs without its sigmoid gate. `plan` is that family's pack plan's
    `layer_types`, since the op indices the weight buffers name are its own."""
    dense = ffn == "dense"
    if any(spec.quant_of(r) == "q8" for r in (DENSE_GEMM_ROLES if dense else GEMM_ROLES)):
        return None
    L, T = layout(spec, ffn=ffn), GEMM_T
    hid, ff = spec.hidden, spec.intermediate if dense else spec.moe_intermediate
    plan = plan if plan is not None else pack_plan(spec)["layer_types"]
    shapes: set[tuple[int, int]] = set()

    def ctx(N: int, K: int) -> str:
        if N % 256 or K % 256:
            raise OpRangeError(f"gemm route: [{N}, {K}] is not a multiple of 256 in both dims")
        shapes.add((N, K))
        return f"gemm_n{N}_k{K}"

    def run(N: int, K: int, w: str) -> dict:
        return {"op": "run", "kernel": ctx(N, K), "args": [w, f"gemm_x_k{K}", f"gemm_y_n{N}"]}

    # The routed experts stay on mx, one token at a time, until the token-batched expert
    # kernel lands (the plan's stage 2). The SHARED expert does not: it is the same 1.97 MB
    # for every token of the block, so the route runs it once as two GEMMs (up|gate are
    # contiguous in the pool and both std_perm, so the band law the GEMM reads is already
    # what is packed) and folds it into the residual mx is handed. mx closes on xres + acc.
    moe_args = ["pool", "xres", "consts", "state", "act", "ptab"]     # mx.py's six, ax's order
    if not dense:
        check_buffer_args("mx", moe_args)
    sff = spec.shared_expert_intermediate
    # The token-batched expert kernel (OPEN-MOE-BATCH, designs/moe_batch): one dispatch streams
    # every slot's expert once for up to MB_NT of its tokens. One xclbin, one instruction stream
    # per dispatch length; the driver takes the shortest stream that holds the experts still
    # owed tokens and patches the slots' expert offsets (moebatch). A set without it runs the
    # routed experts on mx, one token at a time.
    E = spec.num_experts
    # A binary ladder down from E. The driver takes the shortest stream that holds what is
    # left, so the rungs decide the padding: a 256-token block leaves ~306 visits a layer,
    # which without a 64 rounds up to 256 + 128 and streams 78 experts nobody asked for.
    mb_slots = [s for s in (E >> i for i in range(30)) if s >= 8 and s % 8 == 0]
    mb_args = ["pool", "mb_x", "mb_h", "mb_y"]
    if not dense:
        check_buffer_args("moe_batch", mb_args)
    moe_batch = {"kernels": {str(s): f"mb_s{s}" for s in mb_slots}, "args": mb_args, "nt": MB_NT}

    # Block attention as two bf16 GEMMs per kv head (OPEN-PREFILL-ATTN, designs/attn_block): the
    # group's 8 query heads x T tokens against the window, scores then values, the row softmax on
    # the host between. The GEMM takes M and K as runtime parameters, so one xclbin carries a
    # stream per 256 rows of window up to ATTN_LMAX; the host chunks a longer window.
    ag_m = (spec.num_heads // spec.num_kv_heads) * T
    ag_tiers = list(range(256, ATTN_LMAX + 1, 256))
    ag_args = ["ag_a", "ag_b", "ag_c"]
    check_buffer_args("attn_block", ag_args)
    attn_block = None
    if spec.has_full and ag_m % 256 == 0 and spec.head_dim % 256 == 0:
        attn_block = {"m": ag_m, "hd": spec.head_dim, "l_max": ATTN_LMAX, "args": ag_args,
                      "kernels_s": {str(L): f"ag_s{L}" for L in ag_tiers},
                      "kernels_pv": {str(L): f"ag_pv{L}" for L in ag_tiers}}

    def ffn_of(lt: str) -> dict:
        # up and gate are the pool's first two ops, contiguous and both std_perm, so one GEMM
        # gives [up | gate] -- the layout the shared expert's up|gate GEMM already has
        pool = plan[lt]["pool"]
        return {"ffn_program": [run(2 * ff, hid, "gffn_ug_w"), run(hid, ff, "gffn_down_w")],
                "ffn_weights": {
                    "gffn_ug_w": {"from": "pool", "ops": [_op_index(pool, "mlp.up_proj.weight"),
                                                          _op_index(pool, "mlp.gate_proj.weight")]},
                    "gffn_down_w": {"from": "pool", "ops": [_op_index(pool, "mlp.down_proj.weight")]}}}

    def moe_of(lt: str, a_xm: int, a_rout: int, a_res: int) -> tuple[dict, dict, dict]:
        """The MoE tail's keys, in the three places the 35B's manifest has always had them."""
        if dense:
            return {}, {}, {}
        kind = "linear" if lt == LINEAR else "full"
        return ({"moe_kernel": f"mx_{kind}", "moe_args": moe_args},
                {"a_xm": a_xm, "a_rout": a_rout, "a_res": a_res},
                {**shared_of(lt), "moe_batch": moe_batch})

    def shared_of(lt: str) -> dict:
        pool = plan[lt]["pool"]
        return {"shared_program": [run(2 * sff, hid, "gshare_w"), run(hid, sff, "gsdown_w")],
                "shared_ff": sff,
                "shared_weights": {
                    "gshare_w": {"from": "pool", "ops": [_op_index(pool, "share_up_exps_proj.weight"),
                                                         _op_index(pool, "share_gate_exps_proj.weight")]},
                    "gsdown_w": {"from": "pool", "ops": [_op_index(pool, "share_down_exps_proj.weight")]}}}

    types: dict[str, dict] = {}
    if spec.has_linear:
        pool, consts = plan[LINEAR]["pool"], plan[LINEAR]["consts"]
        nch, vw = spec.lin_qkv_dim, spec.lin_value_width
        out_i = _op_index(consts, "linear_attn.ssm_out_proj.weight")
        split = spec.quant_of("linear_out") == "q8"
        if split:
            # The sequential kernel streams this projection at q8 and the GEMM reads q4_1 only.
            # Re-quantising it costs what OPEN-QUANT-Q8 measured (the reason q8 is native), so
            # the route packs the q8 weight as the exact sum of two q4_1 ones (hi and lo codes,
            # pools split_q4_1_chunks) stacked in one buffer: one GEMM of 2 x hid rows, halves
            # added on the host.
            nq = q4_chunks(hid, vw)
            op = {"op": "std_perm", "tensor": consts[out_i]["tensor"], "nch": nq, "in_dim": vw}
            gout = {"from": "pack", "pack": [{**op, "dst": 0, "split": "hi"},
                                             {**op, "dst": nq * CHUNK, "split": "lo"}]}
        else:
            gout = {"from": "consts", "ops": [out_i]}
        m0, m1, m2 = moe_of(LINEAR, L.A_XM, L.A_ROUT, L.A_RES)
        types[LINEAR] = {
            "kind": "linear", "t": T, "eps": spec.norm_eps,
            "program": [run(nch + vw, hid, "gqkvz_w"), run(2 * hid if split else hid, vw, "gout_w")], **m0,
            "weights": {"gqkvz_w": {"from": "pool", "ops": [_op_index(pool, "linear_attn.qkv_proj.weight"),
                                                           _op_index(pool, "self_attn.gate_proj.weight")]},
                        "gout_w": gout},
            "qkv_dim": nch, "vw": vw, "key_heads": spec.lin_key_heads, "value_heads": spec.lin_value_heads,
            "head_dim": spec.lin_value_dim, "conv_kernel": spec.conv_kernel, "ff": ff, **m1,
            **({"out_split": True} if split else {}),
            "state_s_off": L.STATE_S_OFF, "s_head_bytes": L.S_HEAD_BYTES, "s_rows": L.S_ROWS,
            **m2, **(ffn_of(LINEAR) if dense else {}),
        }
    if spec.has_full:
        pool = plan[FULL]["pool"]
        qw, kvw = spec.attn_q_width, spec.attn_kv_width
        nq = q4_chunks(qw, hid)
        m0, m1, m2 = moe_of(FULL, L.AA_XM, L.AA_ROUT, L.AA_RES)
        types[FULL] = {
            "kind": "full", "t": T, "eps": spec.norm_eps,
            "program": [run(2 * qw + 2 * kvw, hid, "gqkvg_w"), run(hid, qw, "go_w")], **m0,
            "weights": {"gqkvg_w": {"from": "pool", "ops": [_op_index(pool, "self_attn.q_proj.weight", 0),
                                                           _op_index(pool, "self_attn.k_proj.weight"),
                                                           _op_index(pool, "self_attn.v_proj.weight"),
                                                           _op_index(pool, "self_attn.q_proj.weight", nq)]},
                        "go_w": {"from": "pool", "ops": [_op_index(pool, "self_attn.o_proj.weight")]}},
            "qw": qw, "kvw": kvw, "nh": spec.num_heads, "kvh": spec.num_kv_heads, "hd": spec.head_dim,
            "rot": spec.rotary_dim, "ff": ff, **m1, **m2, **(ffn_of(FULL) if dense else {}),
        }
        if attn_block:
            types[FULL]["attn_block"] = attn_block
    if is_t2(spec):
        for g in types.values():
            g["y_tn"] = True            # OPEN-GEMM-T2: the GEMM writes y [T, N] ...
            g["x_tile_k"] = 128         # ... and streams x in 128-k tiles ...
            g["x_bfp"] = True           # ... of bfp16 blocks the host makes (GQP_XBFP)
    out = {"layer_types": types, "contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    # Hardware contexts are the scarce thing (every design here takes all eight
    # columns, so contexts time-share the array, and changing one costs ~2.5 ms):
    # the two MoE streams share one xclbin, and the GEMM core program depends on
    # neither N nor K, so every projection shape is an instruction stream over one
    # xclbin. The context points at the first build; a kernel set carries one
    # final.xclbin for the whole GEMM route and one insts.bin per shape.
    qh = spec.quant_hash()
    sfx = f"_q{qh}" if qh else ""
    # The attention GEMMs read q / k / v / P rows only -- no weight format, no rotation -- so a
    # rotated-basis or t2 spec shares the plain one's build directories; a q8 one keeps its own.
    asfx = sfx if spec.q8_roles else ""
    kinds = [k for lt, k in ((LINEAR, "linear"), (FULL, "full")) if lt in types and not dense]
    for kind in kinds:
        name = f"mx_{kind}"
        if "mx" not in out["contexts"]:
            out["contexts"]["mx"] = f"{name}/final.xclbin"
        out["kernels"][name] = {"context": "mx", "insts": f"{name}/insts.bin", "patch": "moeroute2", "build": name}
        out["builds"][name] = {"design": "layer_x/mx.py", "build_dir": f"layer_x/build_mx_{kind}{sfx}", "env": {"MX_KIND": kind}}
    # the expert streams share one xclbin (the core program does not depend on the slot count);
    # the x / h / y globals are sized for the longest
    for s in ([] if dense else mb_slots):
        out["contexts"].setdefault("mb", f"mb_s{mb_slots[0]}/final.xclbin")
        name = f"mb_s{s}"
        out["kernels"][name] = {"context": "mb", "insts": f"{name}/insts.bin", "patch": "moebatch", "build": name}
        out["builds"][name] = {"design": "moe_batch/moe_batch.py", "build_dir": f"moe_batch/build_s{s}{sfx}",
                               "env": {"MB_SLOTS": str(s), "MB_NT": str(MB_NT), "MB_HID": str(hid), "MB_FF": str(ff),
                                       "MB_EXPERTS": str(E), "MB_POOL_DOWN": str(L.POOL_DOWN),
                                       "MB_POOL_BYTES": str(L.POOL_BYTES)}}
    if not dense:
        out["globals"]["mb_x"] = mb_slots[0] * hid * MB_NT * 2
        out["globals"]["mb_h"] = mb_slots[0] * ff * MB_NT * 2
        out["globals"]["mb_y"] = mb_slots[0] * hid * MB_NT * 4
    if attn_block:
        hd = spec.head_dim
        # AG_M is baked into the build; the 35B's directories keep the names they have always had
        msfx = "" if ag_m == 2048 else f"_m{ag_m}"
        out["contexts"]["ag"] = f"ag_s{ag_tiers[0]}/final.xclbin"
        for Lw in ag_tiers:
            for tag, K, N in (("s", hd, Lw), ("pv", Lw, hd)):
                name = f"ag_{tag}{Lw}"
                out["kernels"][name] = {"context": "ag", "insts": f"{name}/insts.bin", "build": name}
                out["builds"][name] = {"design": "attn_block/attn_gemm.py", "build_dir": f"attn_block/build_{tag}{Lw}{msfx}{asfx}",
                                       "env": {"AG_M": str(ag_m), "AG_K": str(K), "AG_N": str(N)}}
        # a: Q or P rows [m, K] bf16; b: the tiled K^T or V [K, N] bf16; c: [m, N] f32 -- sized for the widest
        out["globals"]["ag_a"] = ag_m * ATTN_LMAX * 2
        out["globals"]["ag_b"] = ATTN_LMAX * hd * 2
        out["globals"]["ag_c"] = ag_m * ATTN_LMAX * 4
    # OPEN-GEMM-T2: a ternary (t2) model's GEMM runs the bf16 matmul on aie2p's bfp16 datapath,
    # ~2.6x the native bf16 form on this array. Its weights are exact there (three values per
    # 128-k scale block, so every 8-k block shares one exponent), and the activations take a
    # block-of-8 shared-exponent rounding. A q4_1 model's 16-level weights are not exact in bfp16,
    # so every other spec keeps the native bf16 build, byte for byte.
    bfp = is_t2(spec)
    for N, K in sorted(shapes):
        name, ctx = f"gemm_n{N}_k{K}", "gemm"
        if ctx not in out["contexts"]:
            out["contexts"][ctx] = f"{name}/final.xclbin"
        out["kernels"][name] = {"context": ctx, "insts": f"{name}/insts.bin", "build": name}
        out["globals"][f"gemm_x_k{K}"] = K * T * 9 // 8 if bfp else K * T * 2
        out["globals"][f"gemm_y_n{N}"] = N * T * 4
        env = {"GQP_N": str(N), "GQP_K": str(K), "GQP_T": str(T)}
        if bfp:
            # OPEN-GEMM-T2: and it reads the decode pool's 2-bit chunks (copied per weight, as the
            # q4_1 route copies its own), 128 k per matmul call, and writes y token-major
            # -- and (GQP_XBFP) both operands reach the matmul as bfp16 blocks: the activation from
            # the host at 9 B per 8 values, the weight from the dequant, so the core converts nothing
            env.update(GQP_BFP16="1", GQP_WFMT="t2", GQP_KT="128", GQP_YT="1", GQP_T2_STRIDE=str(T2_CHUNK),
                       GQP_XBFP="1")
        out["builds"][name] = {"design": "gemm_q4_prefill/gemm_q4_prefill.py",
                               "build_dir": f"gemm_q4_prefill/build_n{N}_k{K}_t{T}" + ("_t2kyx" if bfp else ""),
                               "env": env}
    return out


# ---- the step program (what the driver runs per layer type), and the kernel sets that serve it
def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
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
    r = gemm_route(spec)
    if r:
        for k in ("contexts", "kernels", "globals"):
            out[k].update(r[k])
        for lt, gb in r["layer_types"].items():
            out["layer_types"][lt]["gemm_block"] = gb
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
    # A q8 variant bakes different pool offsets and fill sizes into its instruction streams,
    # so it is a different kernel set. Only then does the directory name change: a model with
    # no q8 role keeps the name every shipped build already uses (OPEN-QUANT-Q8).
    qh = spec.quant_hash()
    sfx = f"_q{qh}" if qh else ""
    if spec.has_linear:
        b["lx0"] = {"design": "layer_x/lx.py", "build_dir": f"layer_x/build_lx0{sfx}", "env": {"LX_PART": "0"}}
        b["lx1"] = {"design": "layer_x/lx.py", "build_dir": f"layer_x/build_lx1{sfx}", "env": {"LX_PART": "1"}}
    if spec.has_full:
        b["ax0"] = {"design": "layer_x/ax.py", "build_dir": f"layer_x/build_ax0{sfx}", "env": {"AX_PART": "0"}}
        b["ax1"] = {"design": "layer_x/ax.py", "build_dir": f"layer_x/build_ax1{sfx}", "env": {"AX_PART": "1"}}
    b["ln"] = {"design": "ln/ln.py", "build_dir": "ln/build", "env": {}}
    b["lm_head_q8"] = {"design": "lm_head_q8/lm_head_q8.py", "build_dir": "lm_head_q8/build_full",
                       "env": {"LMHEAD_N": str(spec.vocab), "LMHEAD_K": str(spec.hidden),
                               "LMHEAD_CORES": str(LIMITS["n_cols"])}}
    r = gemm_route(spec)
    if r:
        b.update(r["builds"])
    return b


# the design sources a build of this recipe depends on (for the build key), relative to open_kernels/
GEN_KERNELS = "designs/layer_x/gen_kernels.py"      # the design's kernel-TU generator (export_qwen36_kernels.py runs it per spec)
KERNEL_SOURCES = [
    "designs/layer_x/*.py", "designs/layer_x/*.h",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h", "designs/gemv_q4/gemv_q4.py",
    "designs/attn/*.cc", "designs/attn/*.h",
    "designs/dn_glue/*.cc", "designs/dn_glue/*.h", "designs/dn_post/*.cc",
    "designs/router/*.cc", "designs/router/*.h",
    "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py", "designs/lin_layer/ln_nr.cc",
    "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.cc", "designs/lm_head_q8/*.h",
    "designs/gemm_q4_prefill/*.py", "designs/gemm_q4_prefill/*.cc", "designs/gemm_q4_prefill/*.h",
    "designs/moe_batch/moe_batch.py", "designs/moe_batch/*.cc", "designs/moe_batch/*.h",
    "designs/attn_block/attn_gemm.py", "../npu_offload/gemm_rtp/gemm_pretiled.py", "../npu_offload/gemm_rtp/npue.py",
    "include/vecmath.h", "ironutil.py", "build_design.py",
]
# compiled only when a role is q8, so listing it here does not move a shipped build key
KERNEL_SOURCES_Q8 = ["designs/gemv_q4/gemv_q8.h"]
# the roles this family's designs can stream at q8 (OPEN-QUANT-Q8). The routed experts have
# their own stripe laws, and the shared expert shares their call sites, so both stay q4_1
# and the packer re-quantizes them; `recipes/load.py` downgrades the rest of the container's
# map to match rather than refusing the model.
Q8_ROLES = frozenset({"attn", "linear", "linear_out"})
