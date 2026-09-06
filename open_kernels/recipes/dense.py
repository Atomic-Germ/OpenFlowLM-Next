"""The dense recipe (Qwen3 dense, Llama 3, Gemma 3, HunYuan dense): ModelSpec -> everything
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
theta the spec builder folds the NTK alpha into.

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

from dataclasses import dataclass

from .catalogue import LIMITS, OpRangeError, check_buffer_args, require
from .qwen36moe import BAND_ROWS, CHUNK, ELEM, MB, band_bytes, q4_bytes, q4_chunks, roundup, tab_bytes
from .spec import DENSE, DENSE_LOCAL, ModelSpec


@dataclass(frozen=True)
class DenseLayout:
    # consts: [lnw (ELN)][postln (ELN)][meta: qn bf16 HD @0 | kn @HD*2 (E_A)][sandwich: preffn (ELN)][postffn (ELN)]
    CD_LNW: int; CD_POSTLN: int; CD_META: int; CD_PREFFN: int; CD_POSTFFN: int; CD_BYTES: int
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


@dataclass(frozen=True)
class DenseRecipe:
    spec: ModelSpec
    layout: DenseLayout
    geo: DenseGeometry
    max_ctx: int = 4096


# families whose q/k RMSNorm weight multiplies AFTER the rotation (HunYuan's
# query_layernorm(apply_rotary_pos_emb(q))); everyone else norms first.
QKNORM_POST_ROPE = ("hunyuan",)
DENSE_FAMILIES = ("qwen3", "llama3", "gemma3", "hunyuan")


def lm_rows(spec: ModelSpec) -> int:
    """lm_head rows: the vocabulary rounded up to a whole 64-row band. Every family
    but HunYuan already publishes a padded vocab_size; HunYuan's 128167 is not one,
    and the head's bands (and the logits buffer) are sized by the rounded count while
    `real_vocab` still bounds the ids."""
    return roundup(spec.vocab, BAND_ROWS)


def _check(spec: ModelSpec) -> None:
    n = LIMITS["n_cols"]
    if spec.family not in DENSE_FAMILIES:
        raise OpRangeError(f"dense recipe given a {spec.family!r} spec")
    if spec.activation not in ("silu", "gelu_tanh"):
        raise OpRangeError(f"dense: activation {spec.activation!r} (silu | gelu_tanh)")
    if spec.has_local and spec.sliding_window <= 0:
        raise OpRangeError("dense: dense_local layers need a positive sliding_window")
    if spec.quant != "q4_1":
        raise OpRangeError(f"dense: quant={spec.quant!r}; the gemv_q4 template reads q4_1 chunks only")
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
            qk_norm_post_rope=spec.qk_norm and spec.family in QKNORM_POST_ROPE)
    pc = per_call(spec)
    require("gemv_q4", K=spec.hidden, rs=2, rows_per_core=spec.attn_q_width // n, per_call=pc)
    require("gemv_q4", K=spec.attn_q_width, rs=2, rows_per_core=spec.hidden // n, per_call=pc)
    require("gemv_q4", K=spec.hidden, rs=2, rows_per_core=spec.intermediate // n, per_call=pc)
    require("gemv_q4", K=spec.intermediate, rs=2, rows_per_core=spec.hidden // n, per_call=pc)
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


def geometry(spec: ModelSpec) -> DenseGeometry:
    n = LIMITS["n_cols"]
    hid, ff, nh, kvh, hd = spec.hidden, spec.intermediate, spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    e_a = kvw * 2
    hpe = e_a // (hd * 4)                     # q/k/v heads (f32) per ain element = KVH/2
    hpo = e_a // (hd * 2)                     # og heads (bf16) per aout element = KVH
    wide = max(hid, qw, ff)
    pc = per_call(spec)
    return DenseGeometry(
        N_CORES=n, HID=hid, FF=ff, NH=nh, KVH=kvh, HD=hd, ROT=spec.rotary_dim, GATE=spec.attn_gate,
        QKNORM=spec.qk_norm, QKNORM_POST=spec.qk_norm and spec.family in QKNORM_POST_ROPE,
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
    )


def layout(spec: ModelSpec, max_ctx: int = 4096) -> DenseLayout:
    n = LIMITS["n_cols"]
    hid, ff = spec.hidden, spec.intermediate
    G = geometry(spec)
    eln, e_a = hid * 2, G.KVW * 2
    if 2 * G.HD * 2 > e_a or 512 + 4 * spec.rotary_dim > max(1024, e_a):
        raise OpRangeError("dense: qn | kn or the RoPE record do not fit the attention element")
    # consts
    c = {"lnw": 0, "postln": eln, "meta": 2 * eln, "preffn": 2 * eln + e_a, "postffn": 3 * eln + e_a}
    cd_bytes = roundup((4 * eln + e_a) if spec.sandwich_norms else (2 * eln + e_a), ELEM)
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
    for name, rows, cols in (("q", G.QW, hid), ("k", G.KVW, hid), ("v", G.KVW, hid), ("o", hid, G.QW),
                             ("up", ff, hid), ("gate", ff, hid), ("down", hid, ff)):
        p[name] = off
        off += q4_bytes(rows, cols)
    pool_bytes = roundup(off, MB)
    kv_row = 2 * e_a
    ptab_row = max(1024, e_a)
    band = band_bytes(hid)
    bands = lm_rows(spec) // BAND_ROWS
    return DenseLayout(
        CD_LNW=c["lnw"], CD_POSTLN=c["postln"], CD_META=c["meta"], CD_PREFFN=c["preffn"], CD_POSTFFN=c["postffn"],
        CD_BYTES=cd_bytes,
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
                {"op": "std_perm", "tensor": pre + "self_attn.q_proj.weight", "dst": L.POOL_Q, "nch": q4_chunks(G.QW, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.k_proj.weight", "dst": L.POOL_K, "nch": q4_chunks(G.KVW, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.v_proj.weight", "dst": L.POOL_V, "nch": q4_chunks(G.KVW, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "self_attn.o_proj.weight", "dst": L.POOL_O, "nch": q4_chunks(hid, G.QW), "in_dim": G.QW},
                {"op": "std_perm", "tensor": pre + "mlp.up_proj.weight", "dst": L.POOL_UP, "nch": q4_chunks(ff, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "mlp.gate_proj.weight", "dst": L.POOL_GATE, "nch": q4_chunks(ff, hid), "in_dim": hid},
                {"op": "std_perm", "tensor": pre + "mlp.down_proj.weight", "dst": L.POOL_DOWN, "nch": q4_chunks(hid, ff), "in_dim": ff},
            ],
            "consts": [
                {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CD_LNW, "cap": L.ELN},
                {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CD_POSTLN, "cap": L.ELN},
            ] + ([
                {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CD_META, "cap": G.HD * 2},
                {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CD_META + G.HD * 2, "cap": G.HD * 2},
            ] if spec.qk_norm else []) + ([
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


def programs(spec: ModelSpec) -> dict:
    """One design serves every dense layer type; a layer type with a sliding window gets its own
    kernel entry (the same instruction stream, its own instruction BO, patched with its window)
    and its own position table (its RoPE frequencies, its window's row counts)."""
    L, G = layout(spec), geometry(spec)
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
        out["layer_types"][lt] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": kn, "args": args}],
        }
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    n = LIMITS["n_cols"]
    return {
        "dx": {"design": "dense/dx.py", "build_dir": f"dense/build_{spec.family}_h{spec.hidden}", "env": {}},
        "ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{spec.hidden}_{spec.norm_eps:g}", "env": {"LN_N": str(spec.hidden), "LN_EPS": f"{spec.norm_eps:g}"}},
        "lm_head_q4": {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{lm_rows(spec)}",
                       "env": {"LMHEAD_N": str(lm_rows(spec)), "LMHEAD_K": str(spec.hidden), "LMHEAD_CORES": str(n)}},
    }


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    L = layout(spec, max_ctx)
    return {"hidden": spec.hidden, "vocab": lm_rows(spec), "real_vocab": spec.real_vocab,
            "chunk_bytes": CHUNK, "pool_bytes": L.POOL_BYTES, "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": spec.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": spec.rope_inv_freq()}     # the global table's; each ptab global carries its own


def hf_config_check(spec: ModelSpec) -> dict:
    d = {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
         "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
         "intermediate_size": spec.intermediate}
    if spec.family in ("qwen3", "gemma3", "hunyuan"):
        d["head_dim"] = spec.head_dim          # Llama configs may omit it (hidden / heads)
    if spec.family == "gemma3":
        d["sliding_window"] = spec.sliding_window
    return d


GEN_KERNELS = "designs/dense/gen_kernels.py"      # the design's kernel-TU generator (export_qwen36_kernels.py runs it per spec)
KERNEL_SOURCES = [
    "designs/dense/*.py", "designs/dense/*.cc", "designs/dense/*.h",
    "designs/gemv_q4/gemv_q4.h", "designs/gemv_q4/gemv_tab.h", "designs/gemv_q4/gemv_q4_prep_rt.cc",
    "designs/gemv_q4/gemv_q4_prep_f32_rt.cc",
    "designs/attn/*.cc", "designs/attn/*.h",
    "designs/ln/ln.h", "designs/ln/*.cc", "designs/ln/ln.py", "designs/lin_layer/ln_nr.cc",
    "designs/lm_head_q4/*.py", "designs/lm_head_q4/*.cc",
    "include/vecmath.h", "ironutil.py", "build_design.py",
]
