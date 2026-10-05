"""The Qwen3.5 dense recipe: two families composed, not a third one written.

A Qwen3.5 dense layer is a Qwen3.6-MoE layer with the MoE block replaced by a
silu-gated dense FFN. Everything else -- the 3 linear : 1 full layer pattern,
the gated DeltaNet, the attention output gate, head_dim 256 with a partial
RoPE of 64, 16 linear key heads of 128, conv kernel 4, the q8 lm_head -- is the
MoE's. So this module is a thin surface:

    layout / common / linear / attn   qwen36moe's, with ffn="dense"
                                      (the MoE block, router and shared expert dropped;
                                      the FFN's pool block and `h` / `out2` act stages added)
    ffn                               qwen36moe.ffn_geometry: recipes/dense.py's arithmetic
                                      for the up | gate | act | down tail at this width
    pack_plan                         the MoE plan's ops for qkv / z / q / k / v / gate / o and
                                      the linear consts, the dense plan's std_perm for
                                      up / gate / down, plus the two ops this container needs:
                                      `requant_q4_1` for the q8 ssm_out_proj and `transpose`
                                      for the bf16 alpha / beta copies
    programs                          ONE run per layer type -- nothing is routed, so there is
                                      no part split (the MoE needs two streams only because
                                      the router's output patches the second one)
    gemm_route                        the MoE's block prefill route with ffn="dense": the same
                                      linear / full halves, and the FFN as the shared expert's
                                      two GEMMs without its gate (OPEN-PREFILL-BATCH)

The container (probed on Qwen3.8-Distilled-9B-NPU2, 2026-09-06) names its
tensors `model.layers.N.` and stores every projection as q4_1 / 5120 EXCEPT
three per linear layer: `ssm_out_proj` is q8 / 8704 (the MoE stores it q4_1),
and `ssm_{alpha,beta}_proj` are q8 with a bf16 `[heads, hidden]` copy beside
them (the MoE stores `[hidden, heads]`, which is what `glue_ab` reads).
The q8 out projection is re-quantised to q4_1 on the host: it is the only q8
GEMV in the model, the main cores have no q8 entry point, and the 35B already
runs this exact projection at q4_1. See OPEN-FAMILY-QWEN35 in
specs/open-engine/spec.md and the plan's risk R3 for the measured cost.

Images are refused as on the other VLM families: the open engine has no vision
path, so a request carrying one routes to the closed engine.
"""
from __future__ import annotations

from .catalogue import LIMITS, OpRangeError, check_buffer_args, require
from . import qwen36moe as M
import os

from .attnknobs import probe_env as _attn_probe_env
from .qwen36moe import (BAND_ROWS, CHUNK, ELEM, Q8_CHUNK, Recipe, ab_lanes, ffn_geometry, mixed_check,
                        per_call, proj_op, q4_chunks, quant_check, require_gemv, roundup, t2_check)
from .spec import FULL, LINEAR, ModelSpec

FAMILY = "qwen35"
FFN = "dense"
# OPEN-QUANT-T2's rotated head: a 2-bit chunk (32 rows x 256 K) at its unpadded size -- the head
# has its own pool and its own design (designs/lm_head_t2), so none of the layer pools' padding.
LM_T2_CHUNK = 2176
LM_T2_BAND_ROWS = 64


def head_t2(spec: ModelSpec) -> bool:
    """The container's head is PrismML's rotated ternary weight itself (prism_hadamard.lm_head
    "rotated", q4nx-build's exact q4_1 copy): it runs as t2 through lm_head_t2, with the FWHT on
    its input, whatever format the layers stream. Every other spec keeps the q8 head."""
    return bool(spec.hadamard) and spec.hadamard.get("lm_head") == "rotated"


def lm_t2_chunks(spec: ModelSpec) -> int:
    """t2 chunks in the rotated head: 32-row blocks x 256-wide k-tiles."""
    return spec.vocab // 32 * (spec.hidden // 256)


def lm_t2_pool_bytes(spec: ModelSpec) -> int:
    """The head pool: exactly the chunks lm_head_t2 streams (pack_lmhead zeroes and fills it)."""
    return lm_t2_chunks(spec) * LM_T2_CHUNK


def _check(spec: ModelSpec) -> None:
    n = LIMITS["n_cols"]
    if spec.family != FAMILY:
        raise OpRangeError(f"qwen35 recipe given a {spec.family!r} spec")
    quant_check(spec, "qwen35")
    mixed_check(spec, "qwen35", ("attn", "linear", "linear_out", "ffn"))
    t2_check(spec, "qwen35", ("attn", "linear", "linear_out", "ffn"))
    if spec.quant_of("experts") == "q8" or spec.quant_of("shared") == "q8":
        raise OpRangeError("qwen35: this family has no experts; the 'experts' / 'shared' roles "
                           "cannot be set")
    if spec.num_experts or spec.moe_intermediate or spec.shared_expert_intermediate:
        raise OpRangeError("qwen35: a MoE spec belongs to the qwen36moe recipe")
    if spec.intermediate == 0:
        raise OpRangeError("qwen35: no dense FFN (intermediate is 0)")
    if spec.activation != "silu":
        raise OpRangeError(f"qwen35: activation {spec.activation!r} (silu only)")
    if spec.sandwich_norms or spec.has_local:
        raise OpRangeError("qwen35: sandwich norms / sliding windows are not this family")
    if not spec.has_linear and not spec.has_full:
        raise OpRangeError("qwen35: every layer must be a linear-attention or full-attention layer")
    for what, v in (("hidden", spec.hidden), ("intermediate", spec.intermediate)):
        if v % (BAND_ROWS * n):
            raise OpRangeError(f"qwen35: {what} {v} is not a multiple of {BAND_ROWS * n} "
                               f"(64-row bands over {n} cores)")
    pc = per_call(spec, FFN)
    require("ln", width=spec.hidden)
    if head_t2(spec):
        require("lm_head_t2", K=spec.hidden, vocab=spec.vocab)
    else:
        require("lm_head_q8", K=spec.hidden, vocab=spec.vocab)
    # the FFN tail (recipes/dense.py's GEMV points, at this family's widths); a split down
    # GEMV asks for each piece's K, which is what the core actually runs
    require_gemv(spec, "ffn", spec.hidden, spec.intermediate // n, pc)
    for k in M.down_split(spec) or (spec.intermediate,):
        require_gemv(spec, "ffn", k, spec.hidden // n, pc)
    if spec.has_linear:
        require("deltanet", heads=spec.lin_value_heads, dim=spec.lin_value_dim,
                key_heads=spec.lin_key_heads, conv_kernel=spec.conv_kernel)
        if spec.lin_key_dim != spec.lin_value_dim:
            raise OpRangeError("qwen35: DeltaNet key and value head dims must match")
        require_gemv(spec, "linear", spec.hidden, spec.lin_qkv_dim // n, pc)
        require_gemv(spec, "linear", spec.hidden, spec.lin_value_width // n, pc)
        require_gemv(spec, "linear_out", spec.lin_value_width, spec.hidden // n, pc)
        fills = glue_side_fills(spec)
        if fills > LIMITS["shim_fills"]:
            raise OpRangeError(
                f"qwen35: the glue's side channel needs {fills} fills at hidden {spec.hidden} "
                f"(each xn half, then both accumulators' weight tiles for it, then small and conv), "
                f"over the {LIMITS['shim_fills']} a whole-layer design's shim budget allows. The "
                f"fallback is a second side-class fifo for the xn halves (a design change, not a knob).")
    if spec.has_full:
        require("attn", head_dim=spec.head_dim, num_heads=spec.num_heads, num_kv_heads=spec.num_kv_heads,
                rotary_dim=spec.rotary_dim, rope_theta=spec.rope_theta, qk_norm=spec.qk_norm,
                attn_gate=spec.attn_gate)
        require_gemv(spec, "attn", spec.hidden, spec.attn_q_width // n, pc)
        require_gemv(spec, "attn", spec.hidden, spec.attn_kv_width // n, pc)
        require_gemv(spec, "attn", spec.attn_q_width, spec.hidden // n, pc)
    if M.norm_split(spec) and 4 * spec.hidden * 2 + M.STACK > M.NORM_L1:
        raise OpRangeError(f"qwen35: the norm helper cannot hold three {spec.hidden * 2} B elements and "
                           f"one output at hidden {spec.hidden}, even split")


def xn_side_elems(spec: ModelSpec) -> int:
    """4 KB elements the layer-entry norm output arrives in on the glue's `side` channel."""
    return roundup(spec.hidden * 2, ELEM) // ELEM


def ab_tiles_per_half(spec: ModelSpec) -> list[int]:
    """Alpha (or beta) weight tiles that belong to each 4 KB half of the xn: a tile is one
    4 KB element of the projection -- 64 rows at 32 lanes, 32 rows at the 27B's 64 -- and a
    half carries min(2048, HID - h*2048) rows. Equal halves only when HID is a multiple of
    2048 -- at HID 2560 the two halves are 32 and 8 tiles."""
    rows = ELEM // 2                                  # bf16 rows in one 4 KB element
    per_tile = ELEM // (2 * ab_lanes(spec))
    return [min(rows, spec.hidden - h * rows) // per_tile for h in range(xn_side_elems(spec))]


def glue_side_fills(spec: ModelSpec) -> int:
    """DMA fills the glue's `side` channel issues in one linear-attention dispatch: each xn
    half and its weight tiles, then `small` and the conv taps. Walked accumulator-outer (each
    half carried once per accumulator) one half (HID <= 2048) makes 6 and two (the 4B and the
    9B) make 10; three (the 27B) would make 14, so there the walk turns half-outer, carrying
    each half once for both accumulators, and makes 11 (`qwen36moe.glue_fills`).
    designs/layer_x/lx.py's `dense_sequence` issues exactly these, throttled through
    ironutil.Pipeline: a shim channel's start queue is 4 BDs deep, so they cannot go into
    one TaskGroup."""
    x = xn_side_elems(spec)
    return M.glue_fills(x, M.glue_fills(x, False) > LIMITS["shim_fills"])


def layout(spec: ModelSpec, max_ctx: int = 4096):
    """The MoE recipe's, with the dense tail selected -- NOT a re-derivation: every
    DeltaNet and attention constant is the one `qwen36moe.layout` gives for the same
    attention geometry (tests/test_qwen35.py asserts that against a MoE twin spec)."""
    return M.layout(spec, max_ctx, FFN)


def common(spec: ModelSpec):
    return M.common(spec, FFN)


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> Recipe:
    _check(spec)
    return M.recipe(spec, max_ctx, FFN)


# ---- the packing plan: tensor -> offset -> chunk order. `{l}` is the layer index.
def pack_plan(spec: ModelSpec) -> dict:
    L, F = layout(spec), ffn_geometry(spec)
    hid, ff = spec.hidden, spec.intermediate
    pre = "model.layers.{l}."
    ffn_pool = [
        proj_op(spec, "ffn", pre + "mlp.up_proj.weight", L.POOL_FFN_UP, ff, hid, hid),
        proj_op(spec, "ffn", pre + "mlp.gate_proj.weight", L.POOL_FFN_GATE, ff, hid, hid),
        proj_op(spec, "ffn", pre + "mlp.down_proj.weight", L.POOL_FFN_DOWN, hid, ff, ff),
    ]
    plan: dict = {"pool_bytes": L.POOL_BYTES, "chunk_bytes": CHUNK, "layer_types": {},
                  "lm_head": lm_head_plan(spec),
                  "embed": {"tensor": "model.embed_tokens.weight", "dim": hid},
                  "norm": {"tensor": "model.norm.weight", "bytes": hid * 2}}
    if spec.has_linear:
        vw, nch, heads = spec.lin_value_width, spec.lin_qkv_dim, spec.lin_value_heads
        side = L.C_SIDE
        # dn_glue's accumulator is 32 lanes wide whatever the head count, so a 16-head model's
        # projection is written [hid, 32] with columns 16..31 zero. `dst_rows` appears ONLY
        # when it differs from `rows`: an extra key would move every existing family's plan,
        # its manifest and its build key for a value they already have.
        lanes = ab_lanes(spec)
        pad = {"dst_rows": lanes} if lanes != heads else {}
        plan["layer_types"][LINEAR] = {
            "pool": ffn_pool + [
                proj_op(spec, "linear", pre + "linear_attn.qkv_proj.weight", L.POOL_QKV, nch, hid, hid),
                proj_op(spec, "linear", pre + "self_attn.gate_proj.weight", L.POOL_Z, vw, hid, hid),
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.C_LNW, "cap": L.ELN},
                # the container's q8 alpha / beta come with a bf16 [heads, hidden] copy; the glue
                # reads [hidden, heads], which is what the 35B's container already stores (R4)
                {"op": "transpose", "tensor": pre + "linear_attn.ssm_alpha_proj.bf16.weight",
                 "dst": side + L.SIDE_ALPHA, "rows": heads, "cols": hid, "elem": 2, **pad},
                {"op": "transpose", "tensor": pre + "linear_attn.ssm_beta_proj.bf16.weight",
                 "dst": side + L.SIDE_BETA, "rows": heads, "cols": hid, "elem": 2, **pad},
                {"op": "put", "tensor": pre + "linear_attn.ssm_a", "dst": side + L.SIDE_SMALL,
                 "cap": heads * 4},
                {"op": "put", "tensor": pre + "linear_attn.ssm_dt.bias",
                 "dst": side + L.SIDE_SMALL + heads * 4, "cap": heads * 4},
                {"op": "conv_transpose", "tensor": pre + "linear_attn.ssm_conv1d.weight",
                 "dst": side + L.SIDE_CONV, "taps": spec.conv_kernel, "groups": nch // 1024, "width": 1024},
                {"op": "put", "tensor": pre + "linear_attn.ssm_norm.weight", "dst": L.C_NW,
                 "cap": spec.lin_value_dim * 2},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.C_POSTLN,
                 "cap": L.ELN},
                # `ssm_out_proj` is stored q8 here and q4_1 in the 35B's container; the plan is
                # the same either way -- std_perm re-quantises a q8 source on the way into the
                # pool, because both formats hold the same 32 x 256 tile (OPEN-PACK-PLAN, R3).
                proj_op(spec, "linear_out", pre + "linear_attn.ssm_out_proj.weight", L.C_WOUT,
                        hid, vw, vw),
            ],
        }
    if spec.has_full:
        qw, kvw, hd = spec.attn_q_width, spec.attn_kv_width, spec.head_dim
        nq = q4_chunks(qw, hid)
        plan["layer_types"][FULL] = {
            "pool": ffn_pool + [
                # q_proj is the fused [q | gate] rows; the pool splits the halves
                proj_op(spec, "attn", pre + "self_attn.q_proj.weight", L.POOL_Q, qw, hid, hid, chunk0=0),
                proj_op(spec, "attn", pre + "self_attn.k_proj.weight", L.POOL_K, kvw, hid, hid),
                proj_op(spec, "attn", pre + "self_attn.v_proj.weight", L.POOL_V, kvw, hid, hid),
                proj_op(spec, "attn", pre + "self_attn.q_proj.weight", L.POOL_GATE, qw, hid, hid, chunk0=nq),
                proj_op(spec, "attn", pre + "self_attn.o_proj.weight", L.POOL_O, hid, qw, qw),
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CA_LNW, "cap": L.ELN},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CA_POSTLN,
                 "cap": L.ELN},
                {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CA_META, "cap": hd * 2},
                {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CA_META + hd * 2,
                 "cap": hd * 2},
            ],
        }
    return plan


def lm_head_plan(spec: ModelSpec) -> dict:
    """The lmpool's pack: the q8 head's supertile order, or -- for a rotated ternary head --
    `t2_perm` (std_perm's 64-row band order, the order lm_head_t2 streams) into unpadded
    LM_T2_CHUNK chunks, re-derived at load from the container's exact q4_1 copy."""
    hid = spec.hidden
    if head_t2(spec):
        return {"pool_bytes": lm_t2_pool_bytes(spec),
                "ops": [{"op": "t2_perm", "tensor": "lm_head.weight", "dst": 0, "nch": lm_t2_chunks(spec),
                         "in_dim": hid, "chunk_bytes": LM_T2_CHUNK}]}
    return {"pool_bytes": layout(spec).LMHEAD_POOL_BYTES,
            "ops": [{"op": "lmhead_q8", "tensor": "lm_head.weight", "chunk_bytes": Q8_CHUNK, "in_dim": hid, "dst": 0}]}


def gemm_route(spec: ModelSpec) -> dict | None:
    """The block prefill route (OPEN-PREFILL-BATCH): `qwen36moe.gemm_route` over this family's
    pack plan, with the dense FFN in place of the MoE block. None means the sequential set,
    byte for byte what it was: a projection at q8, or a size whose projections the GEMM cannot
    tile (not a multiple of 256) -- refused here rather than failing the whole export, since
    the sequential path serves that size either way."""
    plan = pack_plan(spec)["layer_types"]
    try:
        route = M.gemm_route(spec, FFN, plan)
    except OpRangeError:
        return None
    if M.is_t2(spec):
        _t2_gemm_weights(route, plan)
    return route


def _t2_gemm_weights(route: dict, plan: dict) -> None:
    """OPEN-GEMM-T2: the block route's GEMM reads the 2-bit chunks the decode pool holds, but the
    t2 pack plan keeps q4_1's 5120 B slot per chunk (each op fills the first half of its run), so
    a fused weight (qkv | z, up | gate, ...) is not one contiguous region of the pool. Every GEMM
    weight that pointed into the pool or the consts therefore becomes its own `pack` buffer of the
    same t2_perm ops (their `chunk_bytes` stride, T2_CHUNK), packed back to back -- the route copies
    each weight into a buffer of its own either way."""
    for lt, g in route["layer_types"].items():
        for key in ("weights", "ffn_weights", "shared_weights"):
            for name, w in list(g.get(key, {}).items()):
                if w.get("from") not in ("pool", "consts"):
                    continue
                ops, dst = [], 0
                for i in w["ops"]:
                    op = dict(plan[lt][w["from"]][i])
                    assert op["op"] == "t2_perm", op
                    op["dst"] = dst
                    ops.append(op)
                    dst += op["nch"] * M.T2_CHUNK
                g[key][name] = {"from": "pack", "pack": ops}


# ---- OPEN-DECODE-ONE-CONTEXT: both layer types on one image (designs/layer_x/dux.py)
def one_context(spec: ModelSpec) -> bool:
    """The decode layer loop as ONE hardware context. OPEN_LAYER_ONE_CTX=1 at export forces it
    on and =0 off. Unset, it is on for an all-t2 spec (Ternary Bonsai 2 27B, the one model it
    was measured and validated on) and off for every other dense model until each is."""
    env = os.environ.get("OPEN_LAYER_ONE_CTX")
    if env is not None:
        return env != "0"
    return bool(spec.quant_map) and all(f == "t2" for f in spec.quant_map.values())


def merged_image(spec: ModelSpec) -> bool:
    """Whether this spec's layer kernels are dux.py's merged image: it needs both layer types,
    no q8 role (the merged main core holds one GEMV entry), the split norm helper, and the two
    og projections at one shape -- dux.py's own asserts, checked here first so a spec that
    cannot merge keeps lx / ax instead of failing its build."""
    if not (one_context(spec) and spec.has_linear and spec.has_full and not spec.q8_roles):
        return False
    R = recipe(spec)
    D, A = R.linear, R.attn
    return (bool(R.ln_split) and (D.OUT_PC, D.OUT_K, D.OG_ELEMS) == (A.O_PC, A.O_K, A.OG_ELEMS)
            and D.XN_SIDE_ELEMS == R.ffn.XN_ELEMS and A.ACORES > 1 and A.NHL == A.HPO)


def tail_in_layer(spec: ModelSpec) -> bool:
    """Whether the final norm and the head run as the merged image's third stream (dux.py part 2)
    instead of their own ln / lm_head_t2 images, so a decode step never leaves the layer context
    (OPEN-DECODE-ONE-CONTEXT-DENSE). It needs the merged image, the rotated ternary head (the
    main cores' t2 GEMV streams it: the layers' own chunk size and band law) and a head that
    splits into whole bands per core. OPEN_LAYER_TAIL=0 at export keeps the ln / lm_head_t2 images
    (an explicit value is in the build key, as OPEN_LAYER_ONE_CTX's)."""
    if os.environ.get("OPEN_LAYER_TAIL") == "0":
        return False
    return (merged_image(spec) and head_t2(spec) and M.T2_CHUNK == LM_T2_CHUNK
            and (spec.vocab // LM_T2_BAND_ROWS) % LIMITS["n_cols"] == 0)


def probe_env() -> dict[str, str]:
    """The build key's probe variables (cache.py). An explicit OPEN_LAYER_ONE_CTX joins them: it
    changes what `builds` compiles, and nothing else in the key sees it. Unset, the flavour is a
    function of the spec, which the key already hashes."""
    e = dict(_attn_probe_env())
    for k in ("OPEN_LAYER_ONE_CTX", "OPEN_LAYER_TAIL"):
        if os.environ.get(k) is not None:
            e[k] = os.environ[k]
    return e


# ---- the step program: ONE run per layer type (nothing is routed, so no part split)
def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    L = layout(spec)
    lm = "lm_head_t2" if head_t2(spec) else "lm_head_q8"
    out: dict = {
        "contexts": {"ln": "ln/final.xclbin", "lm": f"{lm}/final.xclbin"},
        "kernels": {"ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
                    "lm": {"context": "lm", "insts": f"{lm}/insts.bin", "build": lm}},
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": spec.vocab * 4,
                    "lmpool": lm_head_plan(spec)["pool_bytes"],
                    "ptab": {"per_row": L.PTAB_ROW, "inv_freq": spec.rope_inv_freq()}},
    }
    # The merged image (dux.py) carries both layer types, so both run in the context named here
    # and the linear stream takes the attention layer's six buffer arguments -- one image, one
    # kernel signature; its `ptab` is never touched (OPEN-DECODE-ONE-CONTEXT).
    merged = merged_image(spec)
    lin_ctx, full_ctx = ("layer", "layer") if merged else ("lx", "ax")
    if tail_in_layer(spec):
        # The tail is the image's third stream (dux.py part 2): the final norm on the norm helper,
        # the head on the main cores, in the layer context. Its six arguments in the image's
        # order: the head pool, xres, the final norm's weight (as consts), logits (as state), a
        # scratch act (hn, and the norm helper's junk stages), ptab (untouched).
        del out["contexts"]["ln"], out["contexts"]["lm"], out["kernels"]["ln"]
        out["kernels"]["lm"] = {"context": "layer", "insts": "lm/insts.bin", "build": "lm"}
        out["tail"] = [{"op": "run", "kernel": "lm", "args": ["lmpool", "xres", "normw", "logits", "lmact", "ptab"]}]
        for g in ("zero", "xresf", "hn"):
            del out["globals"][g]
        out["globals"]["lmact"] = 64 * 1024
    if spec.has_linear:
        args = ["pool", "xres", "consts", "state", "act", "ptab"] if merged else ["pool", "xres", "consts", "state", "act"]
        check_buffer_args("lx", args)
        out["contexts"][lin_ctx] = "lx/final.xclbin"
        out["kernels"]["lx"] = {"context": lin_ctx, "insts": "lx/insts.bin", "build": "lx"}
        out["layer_types"][LINEAR] = {
            "buffers": {"consts": L.C_BYTES, "act": L.A_BYTES,
                        "state": {"kind": "linear", "bytes": L.STATE_BYTES}},
            "program": [{"op": "run", "kernel": "lx", "args": args}],
        }
    if spec.has_full:
        args = ["pool", "xres", "consts", "state", "act", "ptab"]
        check_buffer_args("ax", args)
        out["contexts"].setdefault(full_ctx, "ax/final.xclbin")
        out["kernels"]["ax"] = {"context": full_ctx, "insts": "ax/insts.bin", "patch": "attnpos", "build": "ax"}
        A = recipe(spec).attn
        if merged and A.RB > 1:
            # attn.h ATTN_BLOCK_WIN (dux.py): the host streams the window as whole blocks of RB
            # rows, padded past `pos`; a kernel and a driver that disagree deadlock the fifo
            out["kernels"]["ax"]["rb_win"] = A.RB
        out["layer_types"][FULL] = {
            "buffers": {"consts": L.CA_BYTES, "act": L.AA_BYTES,
                        "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": "ax", "args": args}],
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
            "intermediate_size": spec.intermediate, "head_dim": spec.head_dim,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "linear_num_value_heads": spec.lin_value_heads, "layer_types": list(spec.layer_types)}


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    """The manifest's `layout` block. No `moe`, no `rout_idx_off`: the engine's
    `has_moe` goes false and nothing asks for the router record."""
    L = layout(spec, max_ctx)
    return {
        "hidden": spec.hidden, "vocab": spec.vocab, "real_vocab": spec.real_vocab,
        "chunk_bytes": CHUNK, "pool_bytes": L.POOL_BYTES,
        "lmhead_pool_bytes": lm_head_plan(spec)["pool_bytes"],
        "lmhead_chunk_bytes": LM_T2_CHUNK if head_t2(spec) else Q8_CHUNK,
        "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim,
        "rope_theta": spec.rope_theta, "rope_inv_freq": spec.rope_inv_freq(),
    }


def builds(spec: ModelSpec) -> dict[str, dict]:
    """name -> {design, build_dir, env}. One build per layer type (one instruction stream
    each), plus the norm at this width and the q8 head at this K."""
    n = LIMITS["n_cols"]
    b: dict[str, dict] = {}
    qh = spec.quant_hash()
    sfx = f"_q{qh}" if qh else ""          # a q8 variant is a different kernel set (OPEN-QUANT-Q8)
    if merged_image(spec):
        # Same set names (lx, ax), so the manifest's kernel names and an export's --only list
        # do not move: one design, two parts, one image (OPEN-DECODE-ONE-CONTEXT).
        for name, part in (("lx", 0), ("ax", 1)) + ((("lm", 2),) if tail_in_layer(spec) else ()):
            b[name] = {"design": "layer_x/dux.py",
                       "build_dir": f"layer_x/build_{spec.family}_dux{part}_h{spec.hidden}{sfx}",
                       "env": {"DUX_PART": str(part)}}
        if tail_in_layer(spec):
            return {**b, **(gemm_route(spec) or {}).get("builds", {})}
    else:
        if spec.has_linear:
            b["lx"] = {"design": "layer_x/lx.py",
                       "build_dir": f"layer_x/build_{spec.family}_lx_h{spec.hidden}{sfx}",
                       "env": {"LX_PART": "0"}}
        if spec.has_full:
            b["ax"] = {"design": "layer_x/ax.py",
                       "build_dir": f"layer_x/build_{spec.family}_ax_h{spec.hidden}{sfx}",
                       "env": {"AX_PART": "0"}}
    b["ln"] = {"design": "ln/ln.py", "build_dir": f"ln/build_{spec.hidden}_{spec.norm_eps:g}",
               "env": {"LN_N": str(spec.hidden), "LN_EPS": f"{spec.norm_eps:g}"}}
    if head_t2(spec):
        b["lm_head_t2"] = {"design": "lm_head_t2/lm_head_t2.py",
                           "build_dir": f"lm_head_t2/build_{spec.vocab}_k{spec.hidden}",
                           "env": {"LMHEAD_N": str(spec.vocab), "LMHEAD_K": str(spec.hidden),
                                   "LMHEAD_CORES": str(n)}}
    else:
        b["lm_head_q8"] = {"design": "lm_head_q8/lm_head_q8.py",
                           "build_dir": f"lm_head_q8/build_{spec.vocab}_k{spec.hidden}",
                           "env": {"LMHEAD_N": str(spec.vocab), "LMHEAD_K": str(spec.hidden),
                                   "LMHEAD_CORES": str(n)}}
    r = gemm_route(spec)
    if r:
        b.update(r["builds"])
    return b


GEN_KERNELS = "designs/layer_x/gen_kernels.py"
KERNEL_SOURCES = [
    "designs/layer_x/*.py", "designs/layer_x/*.h",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h", "designs/gemv_q4/gemv_q4.py",
    "designs/attn/*.cc", "designs/attn/*.h",
    "designs/dn_glue/*.cc", "designs/dn_glue/*.h", "designs/dn_post/*.cc",
    "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py", "designs/lin_layer/ln_nr.cc",
    "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.cc", "designs/lm_head_q8/*.h",
    "designs/gemm_q4_prefill/*.py", "designs/gemm_q4_prefill/*.cc", "designs/gemm_q4_prefill/*.h",
    "designs/attn_block/attn_gemm.py", "../npu_offload/gemm_rtp/gemm_pretiled.py", "../npu_offload/gemm_rtp/npue.py",
    "include/vecmath.h", "include/scalar_fp.h", "ironutil.py", "build_design.py",
]
KERNEL_SOURCES_Q8 = ["designs/gemv_q4/gemv_q8.h"]
# a rotated ternary head (head_t2) compiles these; listed only for such a spec (recipes/cache.py),
# so no other kernel set's build key moves
KERNEL_SOURCES_T2_HEAD = ["designs/lm_head_t2/*.py", "designs/lm_head_t2/*.cc",
                          "designs/gemv_q4/gemv_t2.h", "designs/gemv_q4/wht.h"]
# every projection this family runs goes through `gemv_q4_gy` or `gemv_q4_gms`, both of
# which have q8 twins, so every role it has can be streamed at q8 (OPEN-QUANT-Q8)
Q8_ROLES = frozenset({"attn", "linear", "linear_out", "ffn"})
