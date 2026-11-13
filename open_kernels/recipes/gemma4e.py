"""Gemma 4 E2B / E4B (text), family gemma4e: Gemma 3's dense layer -- sandwich norms, GeGLU-tanh, sliding
and global layers -- with the 12B's attention arithmetic (recipes/gemma4.py: unscaled scores, a weightless v
norm, 512-wide global heads with proportional RoPE) and two things the 12B does not have:

- the last layers compute no K or V: each attends over the cache of the last earlier layer of its own kind,
  that layer's newest row included (`kv_sources`), so their dx build has no k / v projection at all;
- each layer ends with a per-layer input gated into the residual and a per-layer scalar (`ple`,
  `layer_scalar`), on the one-context image in gx.py's layer program (`ple_layout`), else on the host.
E2B also doubles the FFN width in its KV-shared layers and has one kv head (attn.h ATTN_KVH 1).

GGUF only (Google ships the QAT Q4_0 file; there is no q4nx container), so every pool is GGUF-direct. The
pools are the bf16-scale 5120-byte chunk (`std_perm_gguf` narrows the fp16 block scales at pack time), not
the f32-scale twin: the memory rule keeps f32 out of device memory, and the kernel narrows f32 to bf16 on
load anyway, so the numbers are the same (open_kernels/gguf_scale_width.py).

One build per layer KIND: (local | global) x (own | shared KV), as four parts of gx.py -- one image in one
context (`image_cores`, `image_geometry`, `rtp_words`). `G4_PART` names the kind a design build is for, and
recipe(spec) returns that kind's dense recipe; with no part set it returns the first kind's, which is what
the kernel-TU generator reads (it only needs the chunks-per-element, the same in every kind: _check).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

from . import dense as D
from .attnknobs import probe_env as _attn_probe_env
from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import CHUNK, GEMM_T, MB, q4_chunks, roundup
from .spec import DENSE, DENSE_LOCAL, DENSE_LOCAL_SHARED, DENSE_SHARED, ModelSpec

band_bytes = D.band_bytes    # noqa: F401  (dx.py: band_bytes(K, quant))
chunk_bytes = D.chunk_bytes  # noqa: F401  (dx.py: chunk_bytes(quant))

KINDS = (DENSE_LOCAL, DENSE, DENSE_LOCAL_SHARED, DENSE_SHARED)
OWN = {DENSE_LOCAL_SHARED: DENSE_LOCAL, DENSE_SHARED: DENSE}          # a shared kind -> whose cache it reads
PART = {DENSE_LOCAL: "local", DENSE: "global", DENSE_LOCAL_SHARED: "local_shared", DENSE_SHARED: "global_shared"}
KERNEL = {DENSE_LOCAL: "dx_local", DENSE: "dx", DENSE_LOCAL_SHARED: "dx_local_shared", DENSE_SHARED: "dx_shared"}

# llama.cpp's gemma4 tensor names, used directly: a GGUF-only family has no other naming to translate from
BLK = "blk.{l}."
T_Q, T_K, T_V, T_O = BLK + "attn_q.weight", BLK + "attn_k.weight", BLK + "attn_v.weight", BLK + "attn_output.weight"
T_UP, T_GATE, T_DOWN = BLK + "ffn_up.weight", BLK + "ffn_gate.weight", BLK + "ffn_down.weight"
T_QN, T_KN = BLK + "attn_q_norm.weight", BLK + "attn_k_norm.weight"
T_LN, T_POSTLN = BLK + "attn_norm.weight", BLK + "post_attention_norm.weight"
T_PREFFN, T_POSTFFN = BLK + "ffn_norm.weight", BLK + "post_ffw_norm.weight"
T_PLE_GATE, T_PLE_PROJ, T_PLE_NORM = BLK + "inp_gate.weight", BLK + "proj.weight", BLK + "post_norm.weight"
T_SCALAR = BLK + "layer_output_scale.weight"
T_EMBED, T_NORM = "token_embd.weight", "output_norm.weight"
T_PLE_EMBED, T_PLE_MODEL, T_PLE_MODEL_NORM = ("per_layer_token_embd.weight", "per_layer_model_proj.weight",
                                              "per_layer_proj_norm.weight")


def kinds(spec: ModelSpec) -> list[str]:
    return [k for k in KINDS if k in spec.layer_types]


def ple_on_npu() -> bool:
    """The per-layer input inside gx.py's layer program. On by default; OPEN_G4_PLE_NPU=0 at export keeps
    it on the host, between the layer dispatches."""
    return os.environ.get("OPEN_G4_PLE_NPU", "1") != "0"


def probe_env() -> dict[str, str]:
    """The build key's probe variables (cache.py reads this off the family module); the per-layer-input
    flavour changes what `builds` compiles, so an export of one never skips the other's."""
    e = dict(_attn_probe_env())
    if ple_on_npu():
        e["OPEN_G4_PLE_NPU"] = "1"
    return e


def _kind_spec(spec: ModelSpec, kind: str) -> ModelSpec:
    glob = kind in (DENSE, DENSE_SHARED)
    hd = (spec.global_head_dim or spec.head_dim) if glob else spec.head_dim
    rot = (spec.global_rotary_dim or hd) if glob else spec.rotary_dim
    kvh = (spec.global_kv_heads or spec.num_kv_heads) if glob else spec.num_kv_heads
    ff = (spec.intermediate_shared or spec.intermediate) if kind in OWN else spec.intermediate
    return replace(spec, layer_types=(kind,), num_layers=1, head_dim=hd, rotary_dim=rot, num_kv_heads=kvh,
                   intermediate=ff, global_head_dim=0, global_rotary_dim=0, intermediate_shared=0,
                   global_kv_heads=0, attn_k_eq_v=False, extra={})


def geometry(s: ModelSpec) -> D.DenseGeometry:
    """A kind's dense geometry with the 12B's attention knobs (recipes/gemma4.py): V norm, score scale 1.0,
    and the proportional rotation's pair distance on a partly rotated head."""
    return replace(D.geometry(s), VNORM=1, SCALE=float(s.attn_scale),
                   ROPE_PAIR=s.head_dim // 2 if s.rotary_dim < s.head_dim else 0)


def per_call_cap(spec: ModelSpec) -> int | None:
    """The smallest chunks-per-element of the kinds when they differ (E2B's 12288-wide shared FFN), else None."""
    pcs = {D.per_call(_kind_spec(spec, k)) for k in kinds(spec)}
    return min(pcs) if len(pcs) > 1 else None


def sub_spec(spec: ModelSpec, kind: str) -> ModelSpec:
    """One layer kind as the dense recipe sees it: its head width, rotation and FFN width."""
    s = _kind_spec(spec, kind)
    cap = per_call_cap(spec)
    return replace(s, extra={"per_call": cap}) if cap else s


def kv_sources(spec: ModelSpec) -> list[int]:
    """Per layer: -1 for a layer with its own cache, else the layer whose cache it attends over -- the last
    layer of its own kind before the first shared one (transformers' store_full_length_kv)."""
    lt = spec.layer_types
    first = next((i for i, t in enumerate(lt) if t in OWN), len(lt))
    out = []
    for t in lt:
        if t in OWN:
            src = [j for j in range(first) if lt[j] == OWN[t]]
            if not src:
                raise OpRangeError(f"gemma4e: a {t} layer has no earlier {OWN[t]} layer to share the cache of")
            out.append(src[-1])
        else:
            out.append(-1)
    if any(t not in OWN for t in lt[first:]):
        raise OpRangeError("gemma4e: a layer with its own cache after the first KV-shared one")
    return out


def image_cores(spec: ModelSpec) -> int:
    """Main cores of the one-context image (gx.py): the most that every kind's hidden, q and FFN widths split
    over in whole bands; a kv width may instead fill exactly half of them (dense.qkv_segments)."""
    subs = [sub_spec(spec, k) for k in kinds(spec)]
    for n in range(D.LIMITS["n_cols"], 0, -1):
        if all(w % (D.BAND_ROWS * n) == 0 for s in subs for w in (s.hidden, s.attn_q_width, s.intermediate)) and \
           all((s.attn_kv_width // D.BAND_ROWS) % n == 0 or 2 * (s.attn_kv_width // D.BAND_ROWS) == n for s in subs):
            return n
    raise OpRangeError("gemma4e: no main-core count splits every layer kind")


def image_geometry(spec: ModelSpec, kind: str, max_ctx: int = 4096):
    """(layout, geometry) of one kind on the one-context image: its own layout, with its bands spread over
    image_cores() main cores rather than the cores its widths alone would allow."""
    s = sub_spec(spec, kind)
    L, G = D.layout(s, max_ctx), geometry(s)
    n = image_cores(spec)
    return L, replace(G, N_CORES=n, Q_PC=G.QW // D.BAND_ROWS // n, KV_PC=G.KVW // D.BAND_ROWS // n,
                      O_PC=G.HID // D.BAND_ROWS // n, UP_PC=G.FF // D.BAND_ROWS // n, DOWN_PC=G.HID // D.BAND_ROWS // n)


@dataclass(frozen=True)
class PleLayout:
    """The per-layer input's regions on the one-context image, after the kind's own layout: the gate
    (ple_dim x hidden) and projection (hidden x ple_dim) in the pool; post_norm and the f32 layer scalar
    in consts; in act y (the FFN residual), [g | ple_in] (the gate output, then the host's input), z,
    p (the projection) and r (its norm)."""
    POOL_PG: int; POOL_PP: int; POOL_BYTES: int
    CD_PLEN: int; CD_PLES: int; CD_BYTES: int
    AD_Y: int; AD_G: int; AD_IN: int; AD_Z: int; AD_P: int; AD_R: int; AD_BYTES: int
    Y_ELEMS: int                                   # x-stream elements of y (f32, hidden)
    PG_BANDS: int; PP_PC: int                      # gate bands (one per core from core 0); projection bands per core


def ple_layout(spec: ModelSpec, kind: str, max_ctx: int = 4096) -> PleLayout:
    L, G = image_geometry(spec, kind, max_ctx)
    hid, d = spec.hidden, spec.ple_dim
    if d % D.BAND_ROWS or d // D.BAND_ROWS > G.N_CORES or d > D.ELEM // 4:
        raise OpRangeError(f"gemma4e: a {d}-wide per-layer input does not fit one band per main core and one x element")
    pg = L.POOL_BYTES
    pp = pg + d // D.BAND_ROWS * D.band_bytes(hid, "q4_1")
    pool = roundup(pp + hid // D.BAND_ROWS * D.band_bytes(d, "q4_1"), MB)
    cd_n = L.CD_BYTES
    cd_s = cd_n + L.ELN
    a, off = {}, L.AD_BYTES
    for name, size in (("y", roundup(hid * 4, D.ELEM)), ("g", roundup(max(L.ELN, 2 * d * 4), D.ELEM)),
                       ("z", roundup(max(L.ELN, D.ELEM), D.ELEM)), ("p", roundup(hid * 4, D.ELEM)),
                       ("r", roundup(hid * 4, D.ELEM))):
        a[name] = off
        off += size
    return PleLayout(POOL_PG=pg, POOL_PP=pp, POOL_BYTES=pool, CD_PLEN=cd_n, CD_PLES=cd_s,
                     CD_BYTES=roundup(cd_s + L.ELN, D.ELEM), AD_Y=a["y"], AD_G=a["g"], AD_IN=a["g"] + d * 4,
                     AD_Z=a["z"], AD_P=a["p"], AD_R=a["r"], AD_BYTES=off, Y_ELEMS=roundup(hid * 4, D.ELEM) // D.ELEM,
                     PG_BANDS=d // D.BAND_ROWS, PP_PC=hid // D.BAND_ROWS // G.N_CORES)


# RTP words the one-context image's main cores read, one set per core; each kind's stream writes its own
RTP = ("qkv_bands", "og_elems", "o_k", "o_groups", "o_chunks", "up_bands", "h_elems", "down_k", "down_groups",
       "down_chunks")
RTP_WORDS = 16


def rtp_words(spec: ModelSpec, kind: str) -> dict[str, int]:
    L, G = image_geometry(spec, kind)
    nkv = G.KVW // D.BAND_ROWS
    kv = 0 if G.KV_SHARED else (2 * G.KV_PC if nkv % G.N_CORES == 0 else 1)

    def groups(k):
        return D.band_bytes(k, "q4_1") // G.CALL_BYTES

    def chunks(k):
        return D.band_bytes(k, "q4_1") // CHUNK
    return {"qkv_bands": G.Q_PC + kv, "og_elems": G.OG_ELEMS, "o_k": G.QW, "o_groups": groups(G.QW),
            "o_chunks": chunks(G.QW), "up_bands": G.UP_PC, "h_elems": G.H_ELEMS, "down_k": G.FF,
            "down_groups": groups(G.FF), "down_chunks": chunks(G.FF)}


def _check(spec: ModelSpec) -> None:
    if spec.family != "gemma4e":
        raise OpRangeError(f"gemma4e recipe on a {spec.family!r} spec")
    if not spec.ple_dim or spec.attn_k_eq_v or not (spec.v_norm and spec.attn_scale == 1.0):
        raise OpRangeError("gemma4e: needs per-layer inputs, its own v projection, a v norm and score scale 1.0 "
                           "(the 12B is recipes/gemma4.py)")
    bad = sorted(set(spec.layer_types) - set(KINDS))
    if bad:
        raise OpRangeError(f"gemma4e: layer types {bad}")
    kv_sources(spec)
    pcs = {D.per_call(sub_spec(spec, k)) for k in kinds(spec)}
    if len(pcs) != 1:
        raise OpRangeError(f"gemma4e: the layer kinds disagree on chunks per weight element {sorted(pcs)}; "
                           f"the dense kernel TUs are generated once per export")


def _part() -> str | None:
    p = os.environ.get("G4_PART")
    if p and p not in PART.values():
        raise OpRangeError(f"G4_PART={p!r} (one of {sorted(PART.values())})")
    return p


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> D.DenseRecipe:
    _check(spec)
    p = _part()
    kind = next((k for k, v in PART.items() if v == p), None) if p else kinds(spec)[0]
    if kind not in spec.layer_types:
        raise OpRangeError(f"gemma4e: G4_PART={p} but the spec has no {kind} layer")
    s = sub_spec(spec, kind)
    return replace(D.recipe(s, max_ctx), geo=geometry(s))


def layout(spec: ModelSpec, max_ctx: int = 4096) -> D.DenseLayout:
    return recipe(spec, max_ctx).layout


def pool_bytes(spec: ModelSpec, max_ctx: int = 4096) -> int:
    if ple_on_npu():
        return max(ple_layout(spec, k, max_ctx).POOL_BYTES for k in kinds(spec))
    return max(D.layout(sub_spec(spec, k), max_ctx).POOL_BYTES for k in kinds(spec))


def _gguf_op(op: dict) -> dict:
    """A dense plan op, aimed at the GGUF file: projections pack through std_perm_gguf (bf16 scales into
    5120-byte chunks), and the HF-style tensor name becomes the GGUF one."""
    names = {"model.layers.{l}.self_attn.q_proj.weight": T_Q, "model.layers.{l}.self_attn.k_proj.weight": T_K,
             "model.layers.{l}.self_attn.v_proj.weight": T_V, "model.layers.{l}.self_attn.o_proj.weight": T_O,
             "model.layers.{l}.mlp.up_proj.weight": T_UP, "model.layers.{l}.mlp.gate_proj.weight": T_GATE,
             "model.layers.{l}.mlp.down_proj.weight": T_DOWN,
             "model.layers.{l}.self_attn.q_norm.weight": T_QN, "model.layers.{l}.self_attn.k_norm.weight": T_KN,
             "model.layers.{l}.input_layernorm.weight": T_LN,
             "model.layers.{l}.post_attention_layernorm.weight": T_POSTLN,
             "model.layers.{l}.pre_feedforward_layernorm.weight": T_PREFFN,
             "model.layers.{l}.post_feedforward_layernorm.weight": T_POSTFFN}
    o = dict(op)
    if o["tensor"] not in names:
        raise OpRangeError(f"gemma4e: no GGUF name for {o['tensor']}")
    o["tensor"] = names[o["tensor"]]
    if o["op"] == "std_perm":
        o["op"] = "std_perm_gguf"
    return o


def pack_plan(spec: ModelSpec) -> dict:
    L0 = D.layout(sub_spec(spec, kinds(spec)[0]))
    hid = spec.hidden
    types = {}
    for k in kinds(spec):
        plan = D.pack_plan(sub_spec(spec, k))["layer_types"][k]
        pool = [_gguf_op(o) for o in plan["pool"]]
        types[k] = {"pool": pool, "consts": [_gguf_op(o) for o in plan["consts"]]}
        if ple_on_npu():                   # the per-layer input runs in gx.py's layer program
            P, d = ple_layout(spec, k), spec.ple_dim
            if d:
                types[k]["pool"] += [
                    {"op": "std_perm_gguf", "tensor": T_PLE_GATE, "dst": P.POOL_PG, "nch": q4_chunks(d, hid), "in_dim": hid},
                    {"op": "std_perm_gguf", "tensor": T_PLE_PROJ, "dst": P.POOL_PP, "nch": q4_chunks(hid, d), "in_dim": d}]
                types[k]["consts"].append({"op": "put", "tensor": T_PLE_NORM, "dst": P.CD_PLEN, "cap": hid * 2})
            types[k]["consts"].append({"op": "put_f32", "tensor": T_SCALAR, "dst": P.CD_PLES, "cap": 4})
    return {
        "pool_bytes": pool_bytes(spec), "chunk_bytes": CHUNK,
        "layer_types": types,
        # the head is the tied embedding (token_embd), re-quantized to q4_1 when the file is not 4-bit
        "lm_head": {"pool_bytes": L0.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "std_perm_gguf", "tensor": T_EMBED, "dst": 0,
                             "nch": q4_chunks(D.lm_rows(spec), hid), "in_dim": hid}]},
        "embed": {"tensor": T_EMBED, "dim": hid, "scale": hid ** 0.5},
        "norm": {"tensor": T_NORM, "bytes": hid * 2},
    }


def attn_wide_on() -> bool:
    """OPEN-PREFILL-ATTN-WIDE: the block attention on the NPU (designs/fa_wide); G4_ATTN_WIDE=0 exports the host-only set."""
    return os.environ.get("G4_ATTN_WIDE", "1") != "0"


def proportional_inv_freq(theta: float, rot: int, hd: int) -> list[float]:
    """transformers' _compute_proportional_rope_parameters: rot/2 rotated pairs at theta^(-2i/hd)."""
    return [theta ** (-2.0 * i / hd) for i in range(rot // 2)]


def gemm_route(spec: ModelSpec, max_ctx: int = 4096) -> dict | None:
    """The block prefill route (OPEN-PREFILL-G4): per layer kind a `gemma4` gemm_block of seven GEMM steps over
    the layer's decode pool -- input projection (q|k|v, or q alone on a KV-shared layer), o, gate, up, down, and
    the per-layer input's gate and projection -- plus the contexts / kernels / globals / builds they need. None
    when the per-layer input is not in the pool (OPEN_G4_PLE_NPU=0)."""
    if not ple_on_npu():
        return None
    T, d, hid = GEMM_T, spec.ple_dim, spec.hidden
    plans = pack_plan(spec)["layer_types"]
    shapes: set[tuple[int, int]] = set()

    def ctx(N: int, K: int) -> str:
        if N % 256 or K % 256:
            raise OpRangeError(f"gemma4e gemm route: [{N}, {K}] is not a multiple of 256 in both dims")
        shapes.add((N, K))
        return f"gemm_n{N}_k{K}"

    def run(N: int, K: int, w: str) -> dict:
        return {"op": "run", "kernel": ctx(N, K), "args": [w, f"gemm_x_k{K}", f"gemm_y_n{N}"]}

    out: dict = {"layer_types": {}, "contexts": {}, "kernels": {}, "globals": {}, "builds": {}}
    wide: dict[int, tuple[int, int, int]] = {}      # head dim -> (heads, kv heads, cache row in bf16) its fa_wide stream reads
    for k in kinds(spec):
        s = sub_spec(spec, k)
        G = geometry(s)
        pool = plans[k]["pool"]

        def at(*tensors):
            out_, start = [], 0
            for t in tensors:
                out_.append(next(i for i in range(start, len(pool)) if pool[i]["tensor"] == t))
                start = out_[-1] + 1
            return out_
        shared = k in OWN
        qw, kvw, ff = G.QW, 0 if shared else G.KVW, s.intermediate
        local = k in (DENSE_LOCAL, DENSE_LOCAL_SHARED)
        first = ("gq_w", [T_Q]) if shared else ("gqkv3_w", [T_Q, T_K, T_V])
        weights = {first[0]: first[1], "go_w": [T_O], "ggate_w": [T_GATE], "gup_w": [T_UP], "gdown_w": [T_DOWN]}
        program = [run(qw + 2 * kvw, hid, first[0]), run(hid, qw, "go_w"), run(ff, hid, "ggate_w"),
                   run(ff, hid, "gup_w"), run(hid, ff, "gdown_w")]
        weights.update({"gpg_w": [T_PLE_GATE], "gpp_w": [T_PLE_PROJ]})
        program += [run(d, hid, "gpg_w"), run(hid, d, "gpp_w")]
        out["layer_types"][k] = {
            "kind": "gemma4", "t": T, "eps": spec.norm_eps, "program": program,
            "weights": {n: {"from": "pool", "ops": at(*ts)} for n, ts in weights.items()},
            "qw": qw, "kvw": kvw, "ff": ff, "hd": G.HD, "kvh": G.KVH, "rot": s.rotary_dim,
            "window": spec.sliding_window if local else 0, "ple_dim": d,
            "ptab": "ptab_local" if local else "ptab", "sandwich": True, "act": spec.activation}
        # OPEN-PREFILL-ATTN-WIDE: the block's attention as one fa_wide dispatch (scores unscaled, the kind's window); a
        # shared kind reads its owner's cache rows, which have the owner's geometry and so the same stream
        if attn_wide_on():
            geo = (G.QW // G.HD, G.KVH, D.layout(s, max_ctx).KV_ROW // 2)
            if wide.setdefault(G.HD, geo) != geo:
                raise OpRangeError(f"gemma4e: two kinds at head dim {G.HD} disagree on heads / kv heads / cache row")
            out["layer_types"][k]["attn_wide"] = {"kernel": f"fw{G.HD}", "q": "fwq", "o": "fwo", "hd": G.HD,
                                                  "window": spec.sliding_window if local else 0}
    for hd, (nh, kvh, kv_ld) in sorted(wide.items()):
        name = f"fw{hd}"
        out["contexts"][f"fwa{hd}"] = f"{name}/final.xclbin"
        out["kernels"][name] = {"context": f"fwa{hd}", "insts": f"{name}/insts.bin", "build": name, "patch": "fawpos",
                                "hd": hd, "kv_ld": kv_ld, "passes": T // (256 if hd == 256 else 128)}
        out["builds"][name] = {"design": "fa_wide/fa_wide.py",
                               "build_dir": f"fa_wide/build_d{hd}_h{nh}_kv{kvh}_ld{kv_ld}_s1",
                               "env": {"FW_D": str(hd), "FW_L": str(T), "FW_HEADS": str(nh), "FW_KV_HEADS": str(kvh),
                                       "FW_Q_LD": str(nh * hd), "FW_KV_LD": str(kv_ld), "FW_V_COL": str(kv_ld // 2),
                                       "FW_O_LD": str(nh * hd), "FW_SCALE": "1.0"}}
    if wide:
        qmax = max(nh * hd for hd, (nh, _, _) in wide.items())
        out["globals"]["fwq"] = T * qmax * 2
        out["globals"]["fwo"] = T * qmax * 2
    for N, K in sorted(shapes):
        name = f"gemm_n{N}_k{K}"
        out["contexts"].setdefault("gemm", f"{name}/final.xclbin")
        out["kernels"][name] = {"context": "gemm", "insts": f"{name}/insts.bin", "build": name}
        out["globals"][f"gemm_x_k{K}"] = K * T * 2
        out["globals"][f"gemm_y_n{N}"] = N * T * 4
        out["builds"][name] = {"design": "gemm_q4_prefill/gemm_q4_prefill.py",
                               "build_dir": f"gemm_q4_prefill/build_n{N}_k{K}_t{T}",
                               "env": {"GQP_N": str(N), "GQP_K": str(K), "GQP_T": str(T)}}
    return out


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    """One dx kernel per kind, all in context `dx` (gx.py: one image, one stream per kind; OPEN-DECODE-ONE-
    CONTEXT); one position table per attention kind (the shared kinds read their owner's), and the host
    stages' keys."""
    L0 = D.layout(sub_spec(spec, kinds(spec)[0]), max_ctx)
    out = {
        "contexts": {"ln": "ln/final.xclbin", "lm": "lm_head_q4/final.xclbin"},
        "kernels": {"ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
                    "lm": {"context": "lm", "insts": "lm_head_q4/insts.bin", "build": "lm_head_q4"}},
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": D.lm_rows(spec) * 4,
                    "lmpool": L0.LMHEAD_POOL_BYTES},
    }
    for k in kinds(spec):
        s = sub_spec(spec, k)
        L, G = D.layout(s, max_ctx), geometry(s)
        local = k in (DENSE_LOCAL, DENSE_LOCAL_SHARED)
        kn, part, tab = KERNEL[k], PART[k], "ptab_local" if local else "ptab"
        window = spec.sliding_window if local else 0
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        ctx = "dx"
        out["contexts"].setdefault(ctx, f"dx_{part}/final.xclbin")
        out["kernels"][kn] = {"context": ctx, "insts": f"dx_{part}/insts.bin", "patch": "attnpos",
                              "build": f"dx_{part}", "window": window, "kv_row": L.KV_ROW}
        if k not in OWN:
            inv = ([spec.rope_local_theta ** (-2.0 * i / s.rotary_dim) for i in range(s.rotary_dim // 2)] if local
                   else proportional_inv_freq(spec.rope_theta, s.rotary_dim, s.head_dim))
            out["globals"][tab] = {"per_row": L.PTAB_ROW, "inv_freq": inv, "window": window,
                                   "cs_off": G.PTAB_CS_OFF, "rotary_dim": s.rotary_dim}
        P = ple_layout(spec, k, max_ctx) if ple_on_npu() else None
        out["layer_types"][k] = {
            "buffers": {"consts": P.CD_BYTES if P else L.CD_BYTES, "act": P.AD_BYTES if P else L.AD_BYTES,
                        "state": ({"kind": "kv_shared", "row": L.KV_ROW} if k in OWN
                                  else {"kind": "kv", "row": L.KV_ROW})},
            "program": [{"op": "run", "kernel": kn, "args": args}],
        }
        if P:                              # the host writes this layer's per-layer input here each token
            out["layer_types"][k]["ple_in"] = P.AD_IN
    r = gemm_route(spec, max_ctx)
    if r:
        for key in ("contexts", "kernels", "globals"):
            out[key].update(r[key])
        for k, gb in r["layer_types"].items():
            out["layer_types"][k]["gemm_block"] = gb
    return out


def host_stages(spec: ModelSpec) -> dict:
    """What the engine computes on the host around the dispatches (Core's gemma4e host stages)."""
    out = {"kv_source": kv_sources(spec)}
    if spec.ple_dim:
        out["ple"] = {"dim": spec.ple_dim, "embed": T_PLE_EMBED, "embed_scale": spec.ple_dim ** 0.5,
                      "model_proj": T_PLE_MODEL, "model_proj_scale": spec.hidden ** -0.5,
                      "model_norm": T_PLE_MODEL_NORM, "input_scale": 2.0 ** -0.5, "gate": T_PLE_GATE,
                      "proj": T_PLE_PROJ, "norm": T_PLE_NORM, "act": spec.activation, "eps": spec.norm_eps}
    out["layer_scalar"] = T_SCALAR
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    n = D.cores_for(spec)
    tag = "gxp_" if ple_on_npu() else "gx_"
    b = {f"dx_{PART[k]}": {"design": "dense/gx.py",
                           "build_dir": f"dense/build_gemma4e_h{spec.hidden}_{tag}{PART[k]}",
                           "env": {"G4_PART": PART[k]}} for k in kinds(spec)}
    b["ln"] = {"design": "ln/ln.py", "build_dir": f"ln/build_{spec.hidden}_{spec.norm_eps:g}",
               "env": {"LN_N": str(spec.hidden), "LN_EPS": f"{spec.norm_eps:g}"}}
    b["lm_head_q4"] = {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{D.lm_rows(spec)}",
                       "env": {"LMHEAD_N": str(D.lm_rows(spec)), "LMHEAD_K": str(spec.hidden),
                               "LMHEAD_CORES": str(n)}}
    r = gemm_route(spec)
    if r:
        b.update(r["builds"])
    return b


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    s = sub_spec(spec, kinds(spec)[0])
    L = D.layout(s, max_ctx)
    return {"hidden": spec.hidden, "vocab": D.lm_rows(spec), "real_vocab": spec.real_vocab,
            "chunk_bytes": CHUNK, "pool_bytes": pool_bytes(spec, max_ctx),
            "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": s.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": [spec.rope_local_theta ** (-2.0 * i / s.rotary_dim) for i in range(s.rotary_dim // 2)],
            "final_logit_softcapping": spec.final_softcap}


def hf_config_check(spec: ModelSpec) -> dict:
    c = {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
         "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
         "intermediate_size": spec.intermediate, "head_dim": spec.head_dim,
         "global_head_dim": spec.global_head_dim or spec.head_dim, "sliding_window": spec.sliding_window,
         "hidden_size_per_layer_input": spec.ple_dim, "final_logit_softcapping": spec.final_softcap}
    return c


GEN_KERNELS = D.GEN_KERNELS
KERNEL_SOURCES = list(D.KERNEL_SOURCES)
Q8_ROLES = frozenset()          # Google's file is Q4_0 throughout the layers; no q8 role
GGUF_ONLY = True                # recipes/manifest.py: the manifest's own view is the GGUF one
