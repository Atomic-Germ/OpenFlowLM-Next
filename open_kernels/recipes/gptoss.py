"""GPT-OSS (OPEN-FAMILY-GPTOSS) on the open engine, from the GGUF.

A layer is two dispatches with the router on the host between them:

  gx_att  (designs/gptoss/gx.py, GX_PART=att; attnpos):
          ln -> q | k | v -> YaRN RoPE -> attention with the learned sink (a 128-row window on
          the sliding layers) -> o -> res = x + o; xm = post_attention_norm(res)
  gptoss_route (host): the router (F32 [32, 2880] + bias) over xm, top four, softmax over the
          four; the weights go into act beside xm, the four experts' weighted down biases into
          act as `bsum`, and the four experts into gx_moe's weight fills (moeroute2's patch)
  gx_moe  (GX_PART=moe; moeroute2): per expert, up | gate -> the clamped SwiGLU scaled by the
          expert's weight -> h_e; down sums the four experts in each core's accumulator;
          xres = res + down + bsum

then the final norm (`ln`) and the q8 head (`lm_head_q8`, f32 scales).

A prompt prefills on its own block route (prefill_route; OPEN-PREFILL-GPTOSS; GX_PREFILL=0 leaves
it out): the whole prompt through each layer before the next, q|k|v and o as gemm_bfp16 dispatches
of PREFILL_T tokens (the GGUF's Q8_0 rows as bfp16), each routed expert's gate|up and down on
gemm_bfp16's MXFP4 variant (GB_B=mx4: the GGUF's nibbles, converted on the core) over its own
tokens, and the norms, rotation, attention with the sinks, router and SwiGLU on the host.

Weights. The GGUF (ggml-org's MXFP4 file) stores attention, embedding and head at Q8_0 and
the experts at MXFP4. Both become the q8 GEMV's half-tiles (designs/gemv_q4/gemv_q8.h)
exactly, but for Q8_0's fp16 scale narrowing to bf16 (the same narrowing the GGUF q4 path's
f32 planes take in the kernel): an MXFP4 code is twice an e2m1 level, an integer in
[-12, 12], and its block scale 2^(e-128) is exact in bf16. A q/k/v/o/gate/up bias is f32[16]
in each k-tile-0 half-tile's pad (GEMV_Q8_PAD_BIAS), where the accumulator starts.

Widths. 2880 is not a whole number of 256-wide k-tiles, and 64-row bands over 8 cores want
a multiple of 512, so hidden and the expert width are padded to 3072 with zero weights; the
residual's padding stays exactly zero, and the norms take their mean over 2880 (LN_DIV).

Experts in MXFP4 (the default; GX_MX4=0 at export keeps the q8 experts). gx_moe's GEMVs read
the GGUF's own nibbles as mx4 half-tiles (designs/gemv_q4/gemv_mx4.h): the q8 half-tile's 16 x
256 shape and band law at 2368 B instead of 5120, decoded on the core into the same integer codes
and scales, so the products are the q8 route's to the bit. 0.53 GiB of pool per layer instead of
1.09, ~1.6 GB streamed per token instead of ~3.4.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import BAND_ROWS, ELEM, MB, roundup, tab_bytes
from .spec import DENSE, DENSE_LOCAL, ModelSpec

CH = 5120                 # a q8 half-tile: 16 rows x 256 K (gemv_q8.h)
PAD_BIAS = 4352           # f32[16] bias in a k-tile-0 half-tile's pad (GEMV_Q8_PAD_BIAS)
PER_CALL = 2              # half-tiles per w element
MX_CH = 2368              # an mx4 half-tile: the same 16 rows x 256 K at MXFP4 (gemv_mx4.h)
MX_PER_CALL = 4           # mx4 half-tiles per gx_moe w element (9472 B)
N_CORES = 8
TOPK_MAX = 8
LM_TILE = 9216            # lm_head_q8.h's f32-scale chunk: 32 rows x 256 K
LM_BAND_ROWS = 128
LM_CORES = 8
ATTN_CORES = 4            # attnknobs.attn_cores(64): the largest divisor of 64 heads <= 6
STACK = 0x1800


def padded(w: int) -> int:
    """The width a 64-row-band x 8-core pool is built at."""
    return roundup(w, BAND_ROWS * N_CORES)


def band_bytes(K: int, ch: int = CH) -> int:
    """64 rows x K: K/256 k-tiles x 4 sixteen-row parts of `ch` bytes (q8 by default)."""
    return K // 64 * ch


def mx4_on() -> bool:
    """The experts as mx4 half-tiles (the GGUF's MXFP4): on unless GX_MX4=0."""
    return os.environ.get("GX_MX4", "1") != "0"


def expert_ch() -> int:
    return MX_CH if mx4_on() else CH


def expert_per_call() -> int:
    return MX_PER_CALL if mx4_on() else PER_CALL


@dataclass(frozen=True)
class Geometry:
    HID: int; HR: int; FF: int; FR: int
    NH: int; KVH: int; HD: int; ROT: int; QW: int; KVW: int
    N_CORES: int; PER_CALL: int; CALL_BYTES: int; EXPERT_CH: int; EXPERT_PER_CALL: int; EXPERT_CALL_BYTES: int
    Q_PC: int; KV_PC: int; O_PC: int; UP_PC: int; DOWN_PC: int
    E_A: int; HPE: int; HPO: int; Q_AIN_ELEMS: int; K_AIN_ELEMS: int
    XN_ELEMS: int; XM_ELEMS: int; OG_ELEMS: int; H_ELEMS: int
    TAB_BYTES: int; MS_U: int; MS_G: int; MS_FLOATS: int
    EPS: float; WINDOW: int; TOPK: int; EXPERTS: int
    ACORES: int; NHL: int; VEXP: int; MLS: int; RB: int
    PTAB_ELEMS: int; PTAB_CS_ELEM: int


@dataclass(frozen=True)
class Layout:
    CD_LNW: int; CD_POSTLN: int; CD_META: int; CD_SINK: int; CD_BYTES: int
    AD_XN: int; AD_Q: int; AD_KVN: int; AD_OG: int; AD_OUT: int; AD_RES: int; AD_XM: int; AD_W: int
    AD_H: int; AD_OUT2: int; AD_BSUM: int; AD_JUNK: int; AD_BYTES: int
    UP_BYTES: int; EXPERT_UG: int; POOL_DOWN: int; POOL_ATTN: int
    POOL_Q: int; POOL_K: int; POOL_V: int; POOL_O: int; POOL_BYTES: int
    KV_ROW: int; PTAB_ROW: int; MAX_CTX: int; KV_BYTES: int; PTAB_BYTES: int
    LM_ROWS: int; LMHEAD_BANDS: int; LMHEAD_BAND_BYTES: int; LMHEAD_POOL_BYTES: int
    ELN: int


@dataclass(frozen=True)
class GptOssRecipe:
    spec: ModelSpec
    layout: Layout
    geo: Geometry
    max_ctx: int
    kind: str = "gptoss"


def _check(spec: ModelSpec) -> None:
    if spec.family != "gptoss":
        raise OpRangeError(f"gptoss recipe given a {spec.family!r} spec")
    if set(spec.layer_types) - {DENSE, DENSE_LOCAL}:
        raise OpRangeError("gptoss: layers must be sliding (dense_local) or full (dense)")
    if spec.activation != "clamped_swiglu" or not spec.num_experts or not spec.moe_intermediate:
        raise OpRangeError("gptoss: expected routed clamped-SwiGLU experts")
    if spec.experts_per_tok > TOPK_MAX:
        raise OpRangeError(f"gptoss: top-{spec.experts_per_tok} routing (the route record holds {TOPK_MAX})")
    if spec.qk_norm or spec.attn_gate:
        raise OpRangeError("gptoss: q/k norms and an attention gate are not GPT-OSS")
    if spec.rotary_dim != spec.head_dim:
        raise OpRangeError("gptoss: a partial rotation is not GPT-OSS")
    if spec.num_kv_heads % 2 or spec.num_heads % spec.num_kv_heads or spec.num_heads % ATTN_CORES:
        raise OpRangeError(f"gptoss: {spec.num_heads} q / {spec.num_kv_heads} kv heads do not split "
                           f"over the attention elements and {ATTN_CORES} cores")
    qw, kvw = spec.num_heads * spec.head_dim, spec.num_kv_heads * spec.head_dim
    for what, w in (("q width", qw), ("kv width", kvw)):
        if w % (BAND_ROWS * N_CORES):
            raise OpRangeError(f"gptoss: {what} {w} is not whole 64-row bands over {N_CORES} cores")
    if 4 * spec.head_dim + 2 * spec.num_heads > 2 * kvw:
        raise OpRangeError("gptoss: qn | kn | sinks do not fit one meta element (attn.h ATTN_SINK)")


def geometry(spec: ModelSpec) -> Geometry:
    _check(spec)
    hid, ff = padded(spec.hidden), padded(spec.moe_intermediate)
    nh, kvh, hd = spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    e_a = kvw * 2
    ptab_row = max(1024, e_a)
    acores = ATTN_CORES
    nhl = nh // acores
    return Geometry(
        HID=hid, HR=spec.hidden, FF=ff, FR=spec.moe_intermediate,
        NH=nh, KVH=kvh, HD=hd, ROT=spec.rotary_dim, QW=qw, KVW=kvw,
        N_CORES=N_CORES, PER_CALL=PER_CALL, CALL_BYTES=PER_CALL * CH,
        EXPERT_CH=expert_ch(), EXPERT_PER_CALL=expert_per_call(), EXPERT_CALL_BYTES=expert_per_call() * expert_ch(),
        Q_PC=qw // BAND_ROWS // N_CORES, KV_PC=kvw // BAND_ROWS // N_CORES, O_PC=hid // BAND_ROWS // N_CORES,
        UP_PC=ff // BAND_ROWS // N_CORES, DOWN_PC=hid // BAND_ROWS // N_CORES,
        E_A=e_a, HPE=e_a // (hd * 4), HPO=e_a // (hd * 2), Q_AIN_ELEMS=nh // (e_a // (hd * 4)),
        K_AIN_ELEMS=kvh // (e_a // (hd * 4)),
        XN_ELEMS=roundup(hid * 2, ELEM) // ELEM, XM_ELEMS=roundup(hid * 2 + 4 * TOPK_MAX, ELEM) // ELEM,
        OG_ELEMS=roundup(qw * 2, ELEM) // ELEM, H_ELEMS=roundup(ff * 4, ELEM) // ELEM,
        TAB_BYTES=tab_bytes(max(hid, qw, ff)), MS_U=0, MS_G=BAND_ROWS, MS_FLOATS=2 * BAND_ROWS,
        EPS=spec.norm_eps, WINDOW=spec.sliding_window, TOPK=spec.experts_per_tok, EXPERTS=spec.num_experts,
        # RB 4: Peano 2026-10 (llvm-aie in target/npu-dev/env) hits an UNREACHABLE in RegBankSelect
        # compiling attn_row_impl at head dim 64 with 16 heads a core under ATTN_VEXP and RB 1; the
        # blocked walk compiles, and is the faster one besides (granite runs it).
        ACORES=acores, NHL=nhl, VEXP=1, MLS=roundup(nhl, 32), RB=4,
        PTAB_ELEMS=ptab_row // e_a, PTAB_CS_ELEM=512 // e_a,
    )


def lm_rows(spec: ModelSpec) -> int:
    return roundup(spec.vocab, LM_BAND_ROWS)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Layout:
    G = geometry(spec)
    eln = G.HID * 2
    cd = {"lnw": 0, "postln": eln, "meta": 2 * eln}
    cd_bytes = roundup(2 * eln + G.E_A, ELEM)
    a: dict[str, int] = {}
    off = 0
    for name, size in (("xn", G.XN_ELEMS * ELEM), ("q", G.QW * 4), ("kvn", 2 * G.KVW * 4), ("og", G.OG_ELEMS * ELEM),
                       ("out", G.HID * 4), ("res", G.HID * 4), ("xm", G.XM_ELEMS * ELEM),
                       ("h", G.TOPK * G.FF * 4), ("out2", G.HID * 4), ("bsum", G.HID * 4), ("junk", eln)):
        a[name] = off
        off += size
    ad_bytes = roundup(off, ELEM)
    up = (G.FF // BAND_ROWS) * band_bytes(G.HID, G.EXPERT_CH)    # one expert's up (= gate) projection
    down = (G.HID // BAND_ROWS) * band_bytes(G.FF, G.EXPERT_CH)
    if up != down:
        raise OpRangeError("gptoss: up and down must be the same size for the routed-expert patch (moe2)")
    pool_down = spec.num_experts * 2 * up
    p_attn = pool_down + spec.num_experts * up
    p = {"q": p_attn}
    p["k"] = p["q"] + (G.QW // BAND_ROWS) * band_bytes(G.HID)
    p["v"] = p["k"] + (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    p["o"] = p["v"] + (G.KVW // BAND_ROWS) * band_bytes(G.HID)
    pool_bytes = roundup(p["o"] + (G.HID // BAND_ROWS) * band_bytes(G.QW), MB)
    if pool_bytes >= 1 << 32:
        raise OpRangeError("gptoss: a layer pool past 4 GiB does not fit a 32-bit fill offset")
    rows = lm_rows(spec)
    bands = rows // LM_BAND_ROWS
    lm_band = 4 * (G.HID // 256) * LM_TILE
    kv_row = 2 * G.E_A
    ptab_row = G.PTAB_ELEMS * G.E_A
    return Layout(
        CD_LNW=cd["lnw"], CD_POSTLN=cd["postln"], CD_META=cd["meta"], CD_SINK=cd["meta"] + 4 * G.HD,
        CD_BYTES=cd_bytes,
        AD_XN=a["xn"], AD_Q=a["q"], AD_KVN=a["kvn"], AD_OG=a["og"], AD_OUT=a["out"], AD_RES=a["res"],
        AD_XM=a["xm"], AD_W=a["xm"] + G.HID * 2, AD_H=a["h"], AD_OUT2=a["out2"], AD_BSUM=a["bsum"],
        AD_JUNK=a["junk"], AD_BYTES=ad_bytes,
        UP_BYTES=up, EXPERT_UG=2 * up, POOL_DOWN=pool_down, POOL_ATTN=p_attn,
        POOL_Q=p["q"], POOL_K=p["k"], POOL_V=p["v"], POOL_O=p["o"], POOL_BYTES=pool_bytes,
        KV_ROW=kv_row, PTAB_ROW=ptab_row, MAX_CTX=max_ctx, KV_BYTES=max_ctx * kv_row, PTAB_BYTES=max_ctx * ptab_row,
        LM_ROWS=rows, LMHEAD_BANDS=bands, LMHEAD_BAND_BYTES=lm_band, LMHEAD_POOL_BYTES=roundup(bands * lm_band, MB),
        ELN=eln,
    )


PREFILL_T = 256           # tokens a prefill GEMM dispatch takes (gemm_bfp16's M: 4 rows of 64-row tiles)
PREFILL_T_SMALL = 128     # the experts' second variant: 4 rows of 32-row tiles, for the chunks that half fill PREFILL_T
GEMM_WIDE = 1024          # gemm_bfp16's N tiling: 128-row tiles over 8 array columns
MX4_GEMM_TILE = 4352      # gemm_bfp16 GB_B=mx4's B tile: 128 x 64 MXFP4 (designs/gemm_bfp16/mx4_b.cc)
PREFILL_MIN = 12          # the prompt length from which the block route beats one token a step (measured)


def prefill_on() -> bool:
    """The block prefill route (OPEN-PREFILL-GPTOSS): on unless GX_PREFILL=0."""
    return os.environ.get("GX_PREFILL", "1") != "0"


def prefill_hilo() -> bool:
    """q|k|v's and o's weights stacked [bfp16(W) ; bfp16(W - bfp16(W))] (the GGUF's Q8_0 to ~2^-16 instead of
    bfp16's 2^-8, the largest of the route's roundings): on unless GX_PREFILL_HILO=0."""
    return os.environ.get("GX_PREFILL_HILO", "1") != "0"


def prefill_shapes(spec: ModelSpec) -> dict[str, tuple[int, int]]:
    """(N, K) of each prefill GEMM. gate|up stacks the gate's rows at 0 and the up's at N / 2; under
    prefill_hilo q|k|v and o stack their weight's hi rows at 0 and its lo rows at N / 2."""
    G = geometry(spec)
    hr, fr = spec.hidden, spec.moe_intermediate
    w = 2 if prefill_hilo() else 1
    return {"qkv": (w * roundup(G.QW + 2 * G.KVW, GEMM_WIDE), hr), "o": (w * roundup(hr, GEMM_WIDE), G.QW),
            "glu": (2 * roundup(fr, GEMM_WIDE // 2), hr), "down": (roundup(hr, GEMM_WIDE), fr)}


def prefill_expert_bytes(n: int, k: int) -> int:
    return n // 128 * (k // 64) * MX4_GEMM_TILE


def prefill_builds(spec: ModelSpec) -> dict[str, dict]:
    out = {}
    for name, (n, k) in prefill_shapes(spec).items():
        mx4 = name in ("glu", "down")
        out[f"gp_{name}"] = {"design": "gemm_bfp16/gemm_bfp16.py",
                             "build_dir": f"gemm_bfp16/build_m{PREFILL_T}_n{n}_k{k}{'_mx4' if mx4 else ''}",
                             "env": {"GB_M": str(PREFILL_T), "GB_K": str(k), "GB_N": str(n),
                                     **({"GB_B": "mx4"} if mx4 else {})}}
        if mx4:
            # the same GEMM at 32-row tiles: another core program (DIM_M), so a context of its own
            out[f"gp_{name}_s"] = {"design": "gemm_bfp16/gemm_bfp16.py",
                                   "build_dir": f"gemm_bfp16/build_m{PREFILL_T_SMALL}_tm32_n{n}_k{k}_mx4",
                                   "env": {"GB_M": str(PREFILL_T_SMALL), "GB_TM": "32", "GB_K": str(k), "GB_N": str(n),
                                           "GB_B": "mx4"}}
    return out


def prefill_route(spec: ModelSpec) -> dict:
    """The manifest's `gptoss_prefill` and the contexts / kernels / globals it runs on. One context per
    core program: q|k|v and o share gemm_bfp16's (it depends on neither N nor K), gate|up and down the
    MXFP4 variant's, whose streams the engine re-points at each expert (gxexpert)."""
    G = geometry(spec)
    S = prefill_shapes(spec)
    b = "blk.{l}."
    (n_qkv, k_qkv), (n_o, k_o), (n_glu, k_glu), (n_down, k_down) = S["qkv"], S["o"], S["glu"], S["down"]
    xb_glu, xb_down = prefill_expert_bytes(n_glu, k_glu), prefill_expert_bytes(n_down, k_down)
    hl = prefill_hilo()

    def halves(op, n_total):
        """the op, or under hi / lo its hi copy at row0 and its lo copy N / 2 further"""
        return [op] if not hl else [dict(op, split="hi"), dict(op, split="lo", row0=op["row0"] + n_total // 2)]
    weights = {
        "gpw_qkv": [h for r, row0 in (("q", 0), ("k", G.QW), ("v", G.QW + G.KVW))
                    for h in halves({"op": "gp_bfp16", "tensor": b + f"attn_{r}.weight", "row0": row0,
                                     "n_total": n_qkv, "in_dim": k_qkv}, n_qkv)],
        "gpw_o": halves({"op": "gp_bfp16", "tensor": b + "attn_output.weight", "row0": 0, "n_total": n_o,
                         "in_dim": k_o}, n_o),
        "gpw_glu": [{"op": "gp_mx4", "gate": b + "ffn_gate_exps.weight", "up": b + "ffn_up_exps.weight",
                     "experts": spec.num_experts, "n_total": n_glu, "in_dim": k_glu, "expert_bytes": xb_glu}],
        "gpw_down": [{"op": "gp_mx4", "tensor": b + "ffn_down_exps.weight", "experts": spec.num_experts,
                      "n_total": n_down, "in_dim": k_down, "expert_bytes": xb_down}],
    }
    tensors = {"attn_norm": "attn_norm.weight", "post_norm": "post_attention_norm.weight", "q_bias": "attn_q.bias",
               "k_bias": "attn_k.bias", "v_bias": "attn_v.bias", "o_bias": "attn_output.bias",
               "sinks": "attn_sinks.weight", "router": "ffn_gate_inp.weight", "router_bias": "ffn_gate_inp.bias",
               "gate_bias": "ffn_gate_exps.bias", "up_bias": "ffn_up_exps.bias", "down_bias": "ffn_down_exps.bias"}
    section = {
        # the shortest prompt it takes: 2026-10-31, 16 tokens 1.3 s on it against ~2.0 s one token a step
        "t": PREFILL_T, "min_prompt": PREFILL_MIN, "x": "gp_x", "y": "gp_y",
        "qkv": {"kernel": "gpq_qkv", "w": "gpw_qkv", "n": n_qkv, "k": k_qkv, **({"split": True} if hl else {})},
        "o": {"kernel": "gpq_o", "w": "gpw_o", "n": n_o, "k": k_o, **({"split": True} if hl else {})},
        "glu": {"kernel": "gpx_glu", "w": "gpw_glu", "n": n_glu, "k": k_glu, "expert_bytes": xb_glu},
        "down": {"kernel": "gpx_down", "w": "gpw_down", "n": n_down, "k": k_down, "expert_bytes": xb_down},
        # the experts again at PREFILL_T_SMALL rows (32-row tiles): a chunk of an expert's tokens that fits half of it
        # costs ~2/3 of a PREFILL_T dispatch there (2026-10-31: gate|up 0.79 against 1.17 ms)
        "t_small": PREFILL_T_SMALL,
        "glu_small": {"kernel": "gpx_glu_s", "w": "gpw_glu", "n": n_glu, "k": k_glu, "expert_bytes": xb_glu},
        "down_small": {"kernel": "gpx_down_s", "w": "gpw_down", "n": n_down, "k": k_down, "expert_bytes": xb_down},
        "weights": weights,
        "tensors": {role: b + name for role, name in tensors.items()},
        "window": {lt: (spec.sliding_window if lt == DENSE_LOCAL else 0) for lt in sorted(set(spec.layer_types))},
        "hidden_real": spec.hidden, "ff_real": spec.moe_intermediate, "heads": G.NH, "kv_heads": G.KVH,
        "head_dim": G.HD, "experts": spec.num_experts, "topk": spec.experts_per_tok, "eps": spec.norm_eps,
        "rope_scale": spec.rope_scale(), "inv_freq": spec.rope_inv_freq(), "alpha": 1.702, "limit": 7.0,
    }
    kmax, nmax = max(k for _, k in S.values()), max(n for n, _ in S.values())
    return {
        "contexts": {"gpgemm": "gp_qkv/final.xclbin", "gpmx4": "gp_glu/final.xclbin", "gpmx4s": "gp_glu_s/final.xclbin"},
        "kernels": {"gpq_qkv": {"context": "gpgemm", "insts": "gp_qkv/insts.bin", "build": "gp_qkv"},
                    "gpq_o": {"context": "gpgemm", "insts": "gp_o/insts.bin", "build": "gp_o"},
                    "gpx_glu": {"context": "gpmx4", "insts": "gp_glu/insts.bin", "patch": "gxexpert", "build": "gp_glu"},
                    "gpx_down": {"context": "gpmx4", "insts": "gp_down/insts.bin", "patch": "gxexpert",
                                 "build": "gp_down"},
                    "gpx_glu_s": {"context": "gpmx4s", "insts": "gp_glu_s/insts.bin", "patch": "gxexpert",
                                  "build": "gp_glu_s"},
                    "gpx_down_s": {"context": "gpmx4s", "insts": "gp_down_s/insts.bin", "patch": "gxexpert",
                                   "build": "gp_down_s"}},
        "globals": {"gp_x": PREFILL_T * kmax // 8 * 9, "gp_y": PREFILL_T * nmax * 4},
        "section": section,
    }


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> GptOssRecipe:
    return GptOssRecipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec), max_ctx=max_ctx)


def attn_flags(G: Geometry, probe_env=()) -> list[str]:
    """attn.h for GPT-OSS: no q/k norm, no gate, the sink, the vector softmax over 4 cores."""
    flags = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
             "-DATTN_QKNORM=0", "-DATTN_QKNORM_POST=0", f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}",
             f"-DATTN_NHL={G.NHL}", "-DATTN_SINK=1"]
    if G.RB > 1:
        flags.append(f"-DATTN_RB={G.RB}")
    if G.PTAB_ELEMS > 1:
        flags.append("-DATTN_PTAB_SPLIT=1")
    for k, v in probe_env:
        flags.append(f"-D{k}={v}")
    return flags


def ln_flags(G: Geometry) -> list[str]:
    return [f"-DLN_N={G.HID}", f"-DLN_EPS={G.EPS:g}f", f"-DLN_DIV={G.HR}"]


PROBE_VARS = ("ATTN_NULL", "GX_NULL")


def probe_env() -> dict[str, str]:
    return {k: os.environ[k] for k in PROBE_VARS if os.environ.get(k)}


def gguf_quant(spec: ModelSpec) -> str:
    """The GGUF view is the only view: the same program, nested (core.cpp picks it for model.gguf),
    under the format its pools actually hold -- which is also what stops the nesting."""
    return "q8"


def pack_plan(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    b = "blk.{l}."
    sfx = "_mx4" if mx4_on() else ""      # the experts' pack ops: mx4 half-tiles or q8 ones
    one = {
        "pool": [
            {"op": f"gx_ug{sfx}", "up": b + "ffn_up_exps.weight", "gate": b + "ffn_gate_exps.weight",
             "up_bias": b + "ffn_up_exps.bias", "gate_bias": b + "ffn_gate_exps.bias", "dst": 0,
             "experts": spec.num_experts, "rows": G.FF, "in_dim": G.HID, "expert_bytes": L.EXPERT_UG},
            {"op": f"gx_down{sfx}", "tensor": b + "ffn_down_exps.weight", "dst": L.POOL_DOWN,
             "experts": spec.num_experts, "rows": G.HID, "in_dim": G.FF, "expert_bytes": L.UP_BYTES},
            {"op": "gx_q8", "tensor": b + "attn_q.weight", "bias": b + "attn_q.bias", "dst": L.POOL_Q,
             "rows": G.QW, "in_dim": G.HID},
            {"op": "gx_q8", "tensor": b + "attn_k.weight", "bias": b + "attn_k.bias", "dst": L.POOL_K,
             "rows": G.KVW, "in_dim": G.HID},
            {"op": "gx_q8", "tensor": b + "attn_v.weight", "bias": b + "attn_v.bias", "dst": L.POOL_V,
             "rows": G.KVW, "in_dim": G.HID},
            {"op": "gx_q8", "tensor": b + "attn_output.weight", "bias": b + "attn_output.bias", "dst": L.POOL_O,
             "rows": G.HID, "in_dim": G.QW},
        ],
        "consts": [
            {"op": "put", "tensor": b + "attn_norm.weight", "dst": L.CD_LNW, "cap": L.ELN},
            {"op": "put", "tensor": b + "post_attention_norm.weight", "dst": L.CD_POSTLN, "cap": L.ELN},
            {"op": "put", "tensor": b + "attn_sinks.weight", "dst": L.CD_SINK, "cap": 2 * G.NH},
        ],
    }
    return {
        "pool_bytes": L.POOL_BYTES, "chunk_bytes": CH,
        "layer_types": {lt: one for lt in sorted(set(spec.layer_types))},
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "gx_lmhead", "tensor": "output.weight", "dst": 0, "rows": L.LM_ROWS,
                             "in_dim": G.HID}]},
        "embed": {"tensor": "token_embd.weight", "dim": G.HID},
        "norm": {"tensor": "output_norm.weight", "bytes": L.ELN},
    }


def route_step(spec: ModelSpec) -> dict:
    L, G = layout(spec), geometry(spec)
    b = "blk.{l}."
    return {"op": "gptoss_route", "kernel": "gx_moe", "a_xm": L.AD_XM, "a_w": L.AD_W, "a_bsum": L.AD_BSUM,
            "hidden_real": G.HR, "router": b + "ffn_gate_inp.weight", "router_bias": b + "ffn_gate_inp.bias",
            "down_bias": b + "ffn_down_exps.bias"}


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    out = {
        "contexts": {"gx_att": "gx_att/final.xclbin", "gx_moe": "gx_moe/final.xclbin",
                     "ln": "ln/final.xclbin", "lm": "lm_head_q8/final.xclbin"},
        "kernels": {
            "gx_moe": {"context": "gx_moe", "insts": "gx_moe/insts.bin", "patch": "moeroute2", "build": "gx_moe"},
            "ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
            "lm": {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"},
        },
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": G.HID * 4, "zero": G.HID * 4, "normw": G.HID * 2, "xresf": G.HID * 4, "hn": G.HID * 2,
                    "logits": L.LM_ROWS * 4, "lmpool": L.LMHEAD_POOL_BYTES},
    }
    if spec.quant != gguf_quant(spec):
        out["gguf"] = gguf_quant(spec)    # manifest.py nests the GGUF view under this quant
    if prefill_on():
        r = prefill_route(spec)
        for key in ("contexts", "kernels", "globals"):
            out[key].update(r[key])
        out["gptoss_prefill"] = r["section"]
    for lt in sorted(set(spec.layer_types)):
        local = lt == DENSE_LOCAL
        kn = "gx_att_local" if local else "gx_att"
        tab = "ptab_local" if local else "ptab"
        window = spec.sliding_window if local else 0
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        out["kernels"][kn] = {"context": "gx_att", "insts": "gx_att/insts.bin", "patch": "attnpos", "build": "gx_att",
                              "window": window}
        # one rotation for both layer types: GPT-OSS's sliding layers turn at the full layers' YaRN
        # frequencies, where `rope_inv_freq(local=True)` is Gemma's separate unscaled local table
        out["globals"][tab] = {"per_row": L.PTAB_ROW, "inv_freq": spec.rope_inv_freq(), "window": window}
        if spec.rope_scale() != 1.0:
            out["globals"][tab]["scale"] = spec.rope_scale()
        out["layer_types"][lt] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": kn, "args": args}, route_step(spec),
                        {"op": "run", "kernel": "gx_moe", "args": ["pool", "xres", "act"]}],
        }
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    G, L = geometry(spec), layout(spec)
    return (prefill_builds(spec) if prefill_on() else {}) | {
        # gx_att reads the attention past the experts in the same pool, so it follows the experts' form too
        "gx_att": {"design": "gptoss/gx.py", "build_dir": f"gptoss/build_att_h{G.HID}{'_mx4' if mx4_on() else ''}",
                   "env": {"GX_PART": "att", "GX_MX4": "1" if mx4_on() else "0"}},
        "gx_moe": {"design": "gptoss/gx.py", "build_dir": f"gptoss/build_moe_h{G.HID}{'_mx4' if mx4_on() else ''}",
                   "env": {"GX_PART": "moe", "GX_MX4": "1" if mx4_on() else "0"}},
        "ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{G.HID}_{G.EPS:g}_d{G.HR}",
               "env": {"LN_N": str(G.HID), "LN_EPS": f"{G.EPS:g}", "LN_DIV": str(G.HR)}},
        "lm_head_q8": {"design": "lm_head_q8/lm_head_q8.py",
                       "build_dir": f"lm_head_q8/build_gguf_n{L.LM_ROWS}_k{G.HID}_c{LM_CORES}",
                       "env": {"LMHEAD_N": str(L.LM_ROWS), "LMHEAD_K": str(G.HID), "LMHEAD_CORES": str(LM_CORES),
                               "LMHEAD_SCALES_F32": "1"}},
    }


def moe_geometry(spec: ModelSpec) -> dict:
    """stream_patch::MoeGeometry for moeroute2 over gx_moe's fills: slot j is compiled as expert j."""
    L = layout(spec)
    return {"stripe": L.UP_BYTES, "up_bytes": L.UP_BYTES, "down_core": L.UP_BYTES, "pool_down": L.POOL_DOWN,
            "share_up": L.POOL_ATTN, "share_gate": L.POOL_ATTN, "share_down": L.POOL_ATTN,
            "experts": spec.num_experts, "topk": spec.experts_per_tok}


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L, G = layout(spec, max_ctx), geometry(spec)
    return {"hidden": G.HID, "vocab": L.LM_ROWS, "real_vocab": spec.real_vocab, "chunk_bytes": CH,
            "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES, "lmhead_chunk_bytes": LM_TILE,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": spec.rope_inv_freq(), "moe": moe_geometry(spec)}


def hf_config_check(spec: ModelSpec) -> dict:
    sc = spec.rope_scaling or {}
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim, "intermediate_size": spec.moe_intermediate,
            "num_local_experts": spec.num_experts, "num_experts_per_tok": spec.experts_per_tok,
            "sliding_window": spec.sliding_window, "rope_theta": spec.rope_theta,
            "rope_scaling": {"rope_type": "yarn", "factor": sc.get("factor"),
                             "original_max_position_embeddings": sc.get("original_max_position_embeddings"),
                             "beta_fast": sc.get("beta_fast"), "beta_slow": sc.get("beta_slow"),
                             "truncate": sc.get("truncate")}}


GEN_KERNELS = "designs/gptoss/gen_kernels.py"
KERNEL_SOURCES = [
    "designs/gptoss/*.py", "designs/gemv_q4/gemv_q8.h", "designs/gemv_q4/gemv_mx4.h", "designs/gemv_q4/mx4_codes.h",
    "designs/gemm_bfp16/*.py", "designs/gemm_bfp16/*.cc", "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h",
    "designs/attn/*.cc", "designs/attn/*.h", "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py",
    "designs/lin_layer/ln_nr.cc", "designs/lm_head_q8/*.py", "designs/lm_head_q8/*.h", "designs/lm_head_q8/*.cc",
    "include/vecmath.h", "include/scalar_fp.h", "ironutil.py", "build_design.py",
]
