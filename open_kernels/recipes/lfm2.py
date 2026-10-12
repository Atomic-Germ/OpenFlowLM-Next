"""LFM2: ten of sixteen layers replace attention with a short depthwise causal conv.

The attention layers ARE the dense recipe's block - same two norms, same silu-gated FFN,
GQA with q/k RMSNorm and no gate - so everything here composes `dense` rather than copying
it. What is new is the other layer type.

Per token, from `Lfm2ShortConv`:

    h  = W_in @ x                            # 3 x hidden rows, in the order [B | C | u]
    Bx = B * u
    conv[c] = sum_k state[c, k] * w[c, k]    # state[:, taps-1] is this token's Bx
    out = W_out @ (C * conv)

The cache holds `Bx`, the gated product the convolution reduces over, and this token pairs
with the LAST tap. `model/replica_lfm2.py` is the fp64 reference and asserts that
orientation directly - a transposed tap weight is the one error that still produces
plausible numbers.

The conv weight ships as [hidden, taps] and goes into consts TAP-major, [taps, hidden]: a
core loading tap k for 32 consecutive channels wants those 32 contiguous, and at
[hidden, taps] they are a stride-taps gather. `designs/short_conv/sc.h` reads it that way.

The fused `shortconv.in_proj.weight` is [3 * hidden, hidden] with B, C and u down the rows.
The conv core wants B[c], C[c] and u[c] together, which in band order sit a third of the
tensor apart, so the plan emits THREE std_perm ops off the one tensor at source chunk
offsets 0, n, 2n - the same `chunk0` split `qwen35.py` uses on its fused [q | gate]. That
turns one awkward 6144-row projection into three ordinary 2048-row ones the GEMV already
runs, and no new gemv_q4 point is needed anywhere in this model.

See `.claude/plans/lfm2-short-conv.md` for the element accounting and the hardware
procedure.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import dense as D
from .attnknobs import probe_env  # noqa: F401  (cache.py reads it off the family module)
from .catalogue import OpRangeError, check_buffer_args, require
from .qwen36moe import (CHUNK, ELEM, MB, mixed_check, proj_op, q4_chunks, quant_check,
                        require_gemv, role_bytes, roundup)
from .spec import FULL, SHORT_CONV, ModelSpec

# The attention layers ARE dense layers (designs/dense/dx.py runs them unchanged), so the
# chunk geometry a design reads off the family module is the dense one.
band_bytes = D.band_bytes   # noqa: F401  (dx.py: band_bytes(K, quant))
chunk_bytes = D.chunk_bytes  # noqa: F401  (dx.py: chunk_bytes(quant))

TAPS = 3                      # LFM2's conv_L_cache; the [hidden, 3] weight confirms it
EMBED = "model.token_embd.weight"


@dataclass(frozen=True)
class Lfm2Layout:
    """The dense layout, plus what a short-conv layer needs on top of it."""
    dense: D.DenseLayout
    # pool: the three thirds of in_proj, then out_proj, then this layer type's own FFN. The
    # conv block is WIDER than the attention block it replaces (four hidden-square
    # projections against q + o square and k + v narrow), so the FFN cannot sit at the dense
    # offsets and each layer type carries its own set. The pack plan is per layer type
    # already, so that costs nothing but the larger pool both types are allocated.
    POOL_SC_B: int; POOL_SC_C: int; POOL_SC_U: int; POOL_SC_OUT: int
    POOL_SC_UP: int; POOL_SC_GATE: int; POOL_SC_DOWN: int
    POOL_BYTES: int           # max of the two layer types; every layer gets a pool this size
    CD_CONV: int; CD_BYTES: int
    AD_B: int; AD_C: int; AD_U: int; AD_Y: int; AD_BYTES: int
    STATE_BYTES: int          # (taps - 1) x hidden f32, the conv window still in reach
    SC_CHUNKS: int            # chunks in one third of in_proj

    def __getattr__(self, name: str):
        """Anything not defined here comes from the dense layout underneath -- POOL_Q, the
        KV and ptab rows, the lm_head bands. An LFM2 attention layer runs `dx.py` unchanged
        and reads exactly those, while POOL_BYTES / CD_BYTES / AD_BYTES are defined above
        and so shadow the dense ones: both layer types are allocated the larger of the two.
        (A frozen dataclass blocks __setattr__, not __getattr__.)"""
        try:
            return getattr(object.__getattribute__(self, "dense"), name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no {name!r}") from None

    def constants(self) -> dict[str, int]:
        out = dict(self.dense.__dict__)
        out.update({k: v for k, v in self.__dict__.items() if k != "dense"})
        return out


def _check(spec: ModelSpec) -> None:
    """LFM2's own, not `dense._check`: its attention layers are `full_attention`, which that
    one refuses outright ("every layer must be a dense layer with an FFN"). The projection
    checks below are the same ones, asked about the layers that have those projections."""
    n = D.cores_for(spec)
    if spec.family != "lfm2":
        raise OpRangeError(f"lfm2 recipe on a {spec.family!r} spec")
    bad = sorted(set(spec.layer_types) - {SHORT_CONV, FULL})
    if bad:
        raise OpRangeError(f"lfm2: layer types {bad} are neither short_conv nor full_attention")
    if SHORT_CONV not in spec.layer_types:
        raise OpRangeError("lfm2: no short_conv layer in layer_types; this is the dense recipe's model")
    if FULL not in spec.layer_types:
        raise OpRangeError("lfm2: no full_attention layer in layer_types")
    if spec.conv_kernel != TAPS:
        raise OpRangeError(f"lfm2: conv_kernel {spec.conv_kernel} is not {TAPS}; the conv core "
                           f"holds taps - 1 rows of state and the design is built for {TAPS}")
    if spec.activation != "silu":
        raise OpRangeError(f"lfm2: activation {spec.activation!r}, not silu")
    if spec.attn_gate or spec.has_linear or spec.has_dense or spec.intermediate == 0:
        raise OpRangeError("lfm2: expected a short_conv + full_attention hybrid with an FFN on "
                           "every layer, no attention gate and no DeltaNet layer")
    quant_check(spec, "lfm2")
    mixed_check(spec, "lfm2", ("attn", "ffn", "linear", "linear_out"))
    for what, v in (("hidden", spec.hidden), ("intermediate", spec.intermediate),
                    ("q width", spec.attn_q_width), ("kv width", spec.attn_kv_width)):
        if v % (D.BAND_ROWS * n):
            raise OpRangeError(f"lfm2: {what} {v} is not a multiple of {D.BAND_ROWS * n} "
                               f"(64-row bands over {n} cores)")
    if spec.hidden % 256 or spec.intermediate % 256:
        raise OpRangeError("lfm2: hidden and intermediate must be multiples of 256 (one q4 k-tile)")
    if spec.num_kv_heads % 2:
        raise OpRangeError("lfm2: an odd kv-head count does not split into ain elements")
    require("ln", width=spec.hidden)
    require("attn", head_dim=spec.head_dim, num_heads=spec.num_heads, num_kv_heads=spec.num_kv_heads,
            rotary_dim=spec.rotary_dim, rope_theta=spec.rope_theta, qk_norm=spec.qk_norm,
            attn_gate=spec.attn_gate, qk_norm_post_rope=False, qkv_bias=False)
    require("short_conv", taps=TAPS, width=spec.hidden)
    pc = D.per_call(spec)
    require_gemv(spec, "attn", spec.hidden, spec.attn_q_width // n, pc)
    require_gemv(spec, "attn", spec.attn_q_width, spec.hidden // n, pc)
    require_gemv(spec, "ffn", spec.hidden, spec.intermediate // n, pc)
    require_gemv(spec, "ffn", spec.intermediate, spec.hidden // n, pc)
    # the three thirds of in_proj and out_proj are all [hidden, hidden]
    require_gemv(spec, "linear", spec.hidden, spec.hidden // n, pc)
    require_gemv(spec, "linear_out", spec.hidden, spec.hidden // n, pc)
    require("lm_head_q4", K=spec.hidden, vocab=D.lm_rows(spec))


def geometry(spec: ModelSpec) -> D.DenseGeometry:
    """The attention layers' geometry IS the dense one; the conv layer adds no knob."""
    return D.geometry(spec)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Lfm2Layout:
    _check(spec)
    L, G = D.layout(spec, max_ctx), geometry(spec)
    hid = spec.hidden
    ff = spec.intermediate
    p: dict[str, int] = {}
    off = 0
    for name, role, rows, cols in (("b", "linear", hid, hid), ("c", "linear", hid, hid),
                                   ("u", "linear", hid, hid), ("out", "linear_out", hid, hid),
                                   ("up", "ffn", ff, hid), ("gate", "ffn", ff, hid),
                                   ("down", "ffn", hid, ff)):
        p[name] = off
        off += role_bytes(spec, role, rows, cols)
    pool_bytes = max(roundup(off, MB), L.POOL_BYTES)
    # consts: the dense block, then the [hidden, taps] bf16 conv weight
    cd_conv = L.CD_BYTES
    cd_bytes = roundup(cd_conv + hid * TAPS * 2, ELEM)
    # act: the dense bounce, then B / C / u / y (f32 hidden each)
    off = L.AD_BYTES
    a = {}
    for name, width in (("b", 4), ("c", 4), ("u", 4), ("y", 2)):
        a[name] = off
        off += hid * width          # y is bf16: the out projection's GEMV input
    ad_bytes = roundup(off, ELEM)
    return Lfm2Layout(
        dense=L,
        POOL_SC_B=p["b"], POOL_SC_C=p["c"], POOL_SC_U=p["u"], POOL_SC_OUT=p["out"],
        POOL_SC_UP=p["up"], POOL_SC_GATE=p["gate"], POOL_SC_DOWN=p["down"],
        POOL_BYTES=pool_bytes, CD_CONV=cd_conv, CD_BYTES=cd_bytes,
        AD_B=a["b"], AD_C=a["c"], AD_U=a["u"], AD_Y=a["y"], AD_BYTES=ad_bytes,
        STATE_BYTES=(TAPS - 1) * hid * 4,
        SC_CHUNKS=q4_chunks(hid, hid),
    )


@dataclass(frozen=True)
class Lfm2Recipe:
    spec: ModelSpec
    layout: Lfm2Layout
    geo: D.DenseGeometry
    max_ctx: int = 4096


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> Lfm2Recipe:
    _check(spec)
    return Lfm2Recipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def pack_plan(spec: ModelSpec) -> dict:
    """Two layer types. `full_attention` is the dense plan's entry unchanged; `short_conv`
    swaps the four attention projections for the three thirds of in_proj plus out_proj, and
    adds the conv taps to consts. The head is 4-bit here, not q8."""
    Lx, L, G = layout(spec), layout(spec).dense, geometry(spec)
    hid, ff = spec.hidden, spec.intermediate
    pre = "model.layers.{l}."
    base = D.pack_plan(spec)
    attn = base["layer_types"][FULL] if FULL in base["layer_types"] else next(iter(base["layer_types"].values()))
    ffn_ops = [
        proj_op(spec, "ffn", pre + "mlp.up_proj.weight", Lx.POOL_SC_UP, ff, hid, hid),
        proj_op(spec, "ffn", pre + "mlp.gate_proj.weight", Lx.POOL_SC_GATE, ff, hid, hid),
        proj_op(spec, "ffn", pre + "mlp.down_proj.weight", Lx.POOL_SC_DOWN, hid, ff, ff),
    ]
    n = Lx.SC_CHUNKS
    sc = {
        "pool": [
            # one tensor, three regions: B, C and u are [hid, hid] slabs down its rows
            proj_op(spec, "linear", pre + "shortconv.in_proj.weight", Lx.POOL_SC_B, hid, hid, hid, chunk0=0),
            proj_op(spec, "linear", pre + "shortconv.in_proj.weight", Lx.POOL_SC_C, hid, hid, hid, chunk0=n),
            proj_op(spec, "linear", pre + "shortconv.in_proj.weight", Lx.POOL_SC_U, hid, hid, hid, chunk0=2 * n),
            proj_op(spec, "linear_out", pre + "shortconv.out_proj.weight", Lx.POOL_SC_OUT, hid, hid, hid),
        ] + ffn_ops,
        "consts": [
            {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CD_LNW, "cap": L.ELN},
            {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CD_POSTLN, "cap": L.ELN},
            # The container stores [hidden, taps]; the core wants TAP-major, so that loading
            # tap k for 32 consecutive channels is 32 contiguous values instead of a
            # stride-taps gather. [hidden, taps] -> [taps, hidden].
            {"op": "transpose", "tensor": pre + "shortconv.conv.weight", "dst": Lx.CD_CONV,
             "rows": hid, "cols": TAPS, "elem": 2},
        ],
    }
    return {
        "pool_bytes": Lx.POOL_BYTES, "chunk_bytes": CHUNK,
        "layer_types": {FULL: attn, SHORT_CONV: sc},
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "std_perm", "tensor": "lm_head.weight", "dst": 0,
                             "nch": q4_chunks(D.lm_rows(spec), hid), "in_dim": hid}]},
        # LFM2 containers name the table token_embd, not embed_tokens
        "embed": {"tensor": EMBED, "dim": hid},
        "norm": {"tensor": "model.norm.weight", "bytes": hid * 2},
    }


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    """`dx` for the attention layers, `cx` for the conv ones. A conv layer takes no position
    table - it has no rotation - and its state is a fixed-size BO, the `linear` kind the
    engine already parses, rather than a KV cache."""
    Lx = layout(spec, max_ctx)
    L = Lx.dense
    out = D.programs(spec, max_ctx)
    out["contexts"]["cx"] = "short_conv/final.xclbin"
    cx_args = ["pool", "xres", "consts", "state", "act"]
    check_buffer_args("cx", cx_args)
    out["kernels"]["cx"] = {"context": "cx", "insts": "short_conv/insts.bin", "build": "short_conv"}
    out["layer_types"][SHORT_CONV] = {
        "buffers": {"consts": Lx.CD_BYTES, "act": Lx.AD_BYTES,
                    "state": {"kind": "linear", "bytes": Lx.STATE_BYTES}},
        "program": [{"op": "run", "kernel": "cx", "args": cx_args}],
    }
    # the attention type's buffers grow to the shared sizes, so one consts / act BO serves both
    for lt, e in out["layer_types"].items():
        if lt != SHORT_CONV:
            e["buffers"]["consts"] = Lx.CD_BYTES
            e["buffers"]["act"] = Lx.AD_BYTES
    out["globals"]["lmpool"] = L.LMHEAD_POOL_BYTES
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    out = D.builds(spec)
    out["short_conv"] = {"design": "short_conv/cx.py",
                         "build_dir": f"short_conv/build_h{spec.hidden}_t{TAPS}",
                         "env": {"SC_HID": str(spec.hidden), "SC_TAPS": str(TAPS)}}
    return out


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    out = D.manifest_layout(spec, max_ctx)
    Lx = layout(spec, max_ctx)
    out["pool_bytes"] = Lx.POOL_BYTES
    return out


def hf_config_check(spec: ModelSpec) -> dict:
    return D.hf_config_check(spec)


GEN_KERNELS = "designs/dense/gen_kernels.py"
KERNEL_SOURCES = list(D.KERNEL_SOURCES) + [
    "designs/short_conv/cx.py",
    "designs/short_conv/sc.h",
]
