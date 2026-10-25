"""Gemma 4 12B (OPEN-FAMILY-GEMMA4): the dense recipe, twice, in one model.

The 40 sliding layers ARE Gemma 3 12B's dense_local layer -- hidden 3840, 16 q / 8 kv heads at
head dim 256, a 1024-row window, GeGLU-tanh 15360, sandwich norms -- and run designs/dense/dx.py
unchanged in shape. The 8 global layers have a geometry of their own: head dim 512 and ONE kv
head. `ModelSpec`, `dense.geometry()` and `dense.layout()` all describe one geometry per spec,
so this module derives a second spec for the global layers and builds dx.py once per layer type
(G4_KIND, below): two xclbins, two hardware contexts, two layouts, two position tables. The
engine runs a different context per layer type already (LFM2, Qwen3.5); what it needed was the
attention patch's KV / position-record row per KERNEL rather than per model (manifest
`kernels.*.kv_row` / `ptab_row`).

What Gemma 4 changes inside a layer, every one from transformers 5.15 `modeling_gemma4.py`
and checked in the fp64-free reference before anything here relied on it
(NpuEmbeddings tasks/0184 `g4ref.py`):

* score scale 1.0, not 1/sqrt(HD): q_norm / k_norm carry it (attn.h ATTN_SCALE).
* a weightless RMSNorm on V, every layer (attn.h ATTN_VNORM).
* the global layers have no v_proj (`attention_k_eq_v`): V is the RAW k projection, normed
  without weight and not rotated, while K is k_norm'd and rotated. Packing k_proj into the V
  slot as well gives exactly that, since the attention core already norms (V) and norms +
  rotates (K) the two slots separately.
* one kv head is an odd count, and the attention element carries KVH/2 heads; the head is
  DUPLICATED (two pack ops of the same tensor), which is exact -- every q head of the group
  reads an identical row -- for one extra 512-wide K and V row on 8 layers.
* proportional RoPE on the global layers: 64 pairs turn, pairing dim i with i + 256, at
  theta^(-2i/512) (spec.rope_inv_freq; attn.h ATTN_ROPE_PAIR). The record holds 64 cos / sin.
* a per-layer output scalar, folded into the two post norms by the packer (`div_running`).
* a tanh cap on the final logits (the manifest's layout.final_logit_softcapping, on the host).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

from . import dense as D
from .attnknobs import probe_env  # noqa: F401  (cache.py reads it off the family module)
from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import MB, proj_op, q4_chunks, role_bytes, roundup
from .spec import DENSE, DENSE_LOCAL, ModelSpec

band_bytes = D.band_bytes    # noqa: F401  (dx.py: band_bytes(K, quant))
chunk_bytes = D.chunk_bytes  # noqa: F401  (dx.py: chunk_bytes(quant))

KIND_ENV = "G4_KIND"          # which layer type a dx.py build is for: "local" | "global"
KINDS = {"local": DENSE_LOCAL, "global": DENSE}
LAYER_SCALE = "model.layers.{j}.layer_output_scale.weight"   # GGUF blk.{j}.layer_output_scale.weight


def _kind(kind: str | None) -> str:
    k = kind or os.environ.get(KIND_ENV, "local")
    if k not in KINDS:
        raise OpRangeError(f"gemma4: {KIND_ENV}={k!r} (local | global)")
    return k


def sub_spec(spec: ModelSpec, kind: str) -> ModelSpec:
    """The dense spec one layer type is built from. Local: the model's own head fields, every
    layer sliding. Global: head dim 512, the kv head doubled (see the module docstring),
    the proportional rotation's 128 dims, no window."""
    n = spec.num_layers
    if kind == "local":
        return replace(spec, layer_types=(DENSE_LOCAL,) * n)
    return replace(spec, layer_types=(DENSE,) * n, head_dim=spec.global_head_dim or spec.head_dim,
                   num_kv_heads=2 * spec.global_kv_heads if spec.global_kv_heads % 2 else spec.global_kv_heads,
                   rotary_dim=spec.global_rotary_dim or spec.global_head_dim, sliding_window=0)


def _check(spec: ModelSpec) -> None:
    if spec.family != "gemma4":
        raise OpRangeError(f"gemma4 recipe given a {spec.family!r} spec")
    if set(spec.layer_types) != {DENSE, DENSE_LOCAL}:
        raise OpRangeError("gemma4: expected sliding (dense_local) and global (dense) layers")
    if not (spec.v_norm and spec.attn_scale == 1.0 and spec.global_head_dim and spec.global_kv_heads):
        raise OpRangeError("gemma4: the spec lacks the Gemma 4 attention fields (v_norm, attn_scale 1.0, "
                           "global_head_dim, global_kv_heads)")
    if spec.global_kv_heads not in (1, 2):
        raise OpRangeError(f"gemma4: {spec.global_kv_heads} global kv heads (the duplication covers 1)")
    if not spec.attn_k_eq_v:
        raise OpRangeError("gemma4: global layers with their own v_proj are not packed here")
    for kind in KINDS:
        s = sub_spec(spec, kind)
        D._check(s)
        if D.cores_for(s) != D.cores_for(sub_spec(spec, "local")):
            raise OpRangeError("gemma4: the two layer types split over different core counts")


def _require(spec: ModelSpec, kind: str, G: D.DenseGeometry) -> None:
    """The attention tuple, with Gemma 4's three knobs IN it: the catalogue's Gemma 3 12B point
    (256, 16, 8, 256) is not this kernel, because the same shape with a V norm, scale 1.0 and
    another pair distance computes something else."""
    D.require("attn", head_dim=G.HD, num_heads=G.NH, num_kv_heads=G.KVH, rotary_dim=G.ROT,
              rope_theta=spec.rope_theta, qk_norm=True, attn_gate=False, qk_norm_post_rope=False,
              qkv_bias=False, v_norm=bool(G.VNORM), attn_scale=G.SCALE, rope_pair=G.ROPE_PAIR)


def geometry(spec: ModelSpec, kind: str | None = None) -> D.DenseGeometry:
    k = _kind(kind)
    s = sub_spec(spec, k)
    G = D.geometry(s)
    return replace(G, VNORM=1, SCALE=float(spec.attn_scale), QLOCAL=1,
                   ROPE_PAIR=(s.head_dim // 2) if (k == "global" and s.rotary_dim < s.head_dim) else 0)


def layout(spec: ModelSpec, max_ctx: int = 4096, kind: str | None = None) -> D.DenseLayout:
    return D.layout(sub_spec(spec, _kind(kind)), max_ctx)


def pool_bytes(spec: ModelSpec) -> int:
    """Both layer types are allocated the larger pool (the global q / o are twice as wide)."""
    return max(layout(spec, kind=k).POOL_BYTES for k in KINDS)


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> D.DenseRecipe:
    """The recipe dx.py builds: the layer type G4_KIND names (local when unset)."""
    _check(spec)
    k = _kind(None)
    G = geometry(spec, k)
    _require(spec, k, G)
    return D.DenseRecipe(spec=sub_spec(spec, k), layout=layout(spec, max_ctx, k), geo=G, max_ctx=max_ctx)


def _layer_plan(spec: ModelSpec, kind: str) -> dict:
    s = sub_spec(spec, kind)
    L, G = D.layout(s), geometry(spec, kind)
    hid, ff = spec.hidden, spec.intermediate
    pre = "model.layers.{l}."
    if kind == "local":
        pool = D.pack_plan(s)["layer_types"][DENSE_LOCAL]["pool"]
    else:
        # k_proj is one 512-row head. The attention core wants KVH = 2 heads per row half, so
        # both slots get the head twice, back to back (the pool is band-major in row order:
        # rows [k; k] are two ordinary 8-band projections end to end). The V slot is the SAME
        # tensor: attention_k_eq_v.
        kv1 = spec.global_kv_heads * s.head_dim
        dup = G.KVW // kv1
        step = role_bytes(s, "attn", kv1, hid)
        k_ops = [proj_op(s, "attn", pre + "self_attn.k_proj.weight", L.POOL_K + i * step, kv1, hid, hid)
                 for i in range(dup)]
        v_ops = [proj_op(s, "attn", pre + "self_attn.k_proj.weight", L.POOL_V + i * step, kv1, hid, hid)
                 for i in range(dup)]
        pool = ([proj_op(s, "attn", pre + "self_attn.q_proj.weight", L.POOL_Q, G.QW, hid, hid)] + k_ops + v_ops +
                [proj_op(s, "attn", pre + "self_attn.o_proj.weight", L.POOL_O, hid, G.QW, G.QW),
                 proj_op(s, "ffn", pre + "mlp.up_proj.weight", L.POOL_UP, ff, hid, hid),
                 proj_op(s, "ffn", pre + "mlp.gate_proj.weight", L.POOL_GATE, ff, hid, hid),
                 proj_op(s, "ffn", pre + "mlp.down_proj.weight", L.POOL_DOWN, hid, ff, ff)])
    # layer_output_scale folded in: the residual is carried divided by P_l = prod_{j<l} s_j,
    # which every reader of it -- an RMSNorm -- cannot see (up to eps), so the two outputs added
    # to it are divided by P_l too, and the scalar disappears. `div_running` names the scalar
    # tensor; the packer divides by the product of its value over the layers BEFORE this one.
    run = {"div_running": LAYER_SCALE}
    consts = [
        {"op": "put", "tensor": pre + "input_layernorm.weight", "dst": L.CD_LNW, "cap": L.ELN},
        {"op": "put", "tensor": pre + "post_attention_layernorm.weight", "dst": L.CD_POSTLN, "cap": L.ELN, **run},
        {"op": "put", "tensor": pre + "self_attn.q_norm.weight", "dst": L.CD_META, "cap": G.HD * 2},
        {"op": "put", "tensor": pre + "self_attn.k_norm.weight", "dst": L.CD_META + G.HD * 2, "cap": G.HD * 2},
        {"op": "put", "tensor": pre + "pre_feedforward_layernorm.weight", "dst": L.CD_PREFFN, "cap": L.ELN},
        {"op": "put", "tensor": pre + "post_feedforward_layernorm.weight", "dst": L.CD_POSTFFN, "cap": L.ELN, **run},
    ]
    return {"pool": pool, "consts": consts}


def pack_plan(spec: ModelSpec) -> dict:
    s = sub_spec(spec, "local")
    L = D.layout(s)
    hid = spec.hidden
    f32 = spec.quant == "q4_1_f32"
    return {
        "pool_bytes": pool_bytes(spec), "chunk_bytes": D.chunk_bytes(D.pool_quant(spec)),
        "layer_types": {KINDS[k]: _layer_plan(spec, k) for k in KINDS},
        "lm_head": {"pool_bytes": L.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "std_perm_gguf" if f32 else "std_perm", "tensor": "lm_head.weight", "dst": 0,
                             "nch": q4_chunks(D.lm_rows(spec), hid, D.pool_quant(spec)), "in_dim": hid}]},
        # GGUF stores the raw table; HF multiplies by sqrt(hidden) at lookup (as Gemma 3)
        "embed": {"tensor": "model.embed_tokens.weight", "dim": hid, **({"scale": hid ** 0.5} if f32 else {})},
        "norm": {"tensor": "model.norm.weight", "bytes": hid * 2},
    }


def _kernel_sets(spec: ModelSpec) -> dict[str, str]:
    f32 = spec.quant == "q4_1_f32"
    return {"local": "dx_f32" if f32 else "dx", "global": "dxg_f32" if f32 else "dxg"}


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    """dx_local on the local design's context, dx on the global one's. Each attnpos kernel
    names its own KV and position-record row (the engine's per-kernel geometry); each layer
    type its own consts / act / state sizes and position table."""
    f32 = spec.quant == "q4_1_f32"
    sets = _kernel_sets(spec)
    lm_set = "lm_head_q4_f32" if f32 else "lm_head_q4"
    Ll = layout(spec, max_ctx, "local")
    out = {
        "contexts": {"dx": f"{sets['local']}/final.xclbin", "dxg": f"{sets['global']}/final.xclbin",
                     "ln": "ln/final.xclbin", "lm": f"{lm_set}/final.xclbin"},
        "kernels": {"ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
                    "lm": {"context": "lm", "insts": f"{lm_set}/insts.bin", "build": lm_set}},
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": D.lm_rows(spec) * 4,
                    "lmpool": Ll.LMHEAD_POOL_BYTES},
    }
    for kind, lt in KINDS.items():
        L = layout(spec, max_ctx, kind)
        local = kind == "local"
        kn, tab, ctx = ("dx_local", "ptab_local", "dx") if local else ("dx", "ptab", "dxg")
        window = spec.sliding_window if local else 0
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        out["kernels"][kn] = {"context": ctx, "insts": f"{sets[kind]}/insts.bin", "patch": "attnpos",
                              "build": sets[kind], "window": window,
                              "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW}
        out["globals"][tab] = {"per_row": L.PTAB_ROW, "inv_freq": spec.rope_inv_freq(local=local), "window": window,
                               "rotary_dim": sub_spec(spec, kind).rotary_dim}
        out["layer_types"][lt] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES, "state": {"kind": "kv", "row": L.KV_ROW}},
            "program": [{"op": "run", "kernel": kn, "args": args}],
        }
    if spec.quant == "q4_1":
        g = programs(replace(spec, quant="q4_1_f32"), max_ctx)
        out["gguf"] = {k: g[k] for k in ("contexts", "kernels", "layer_types", "tail", "globals")}
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    n = D.cores_for(sub_spec(spec, "local"))
    rows = D.lm_rows(spec)
    b = {"ln": {"design": "ln/ln.py", "build_dir": f"ln/build_{spec.hidden}_{spec.norm_eps:g}",
                "env": {"LN_N": str(spec.hidden), "LN_EPS": f"{spec.norm_eps:g}"}},
         "lm_head_q4": {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{rows}",
                        "env": {"LMHEAD_N": str(rows), "LMHEAD_K": str(spec.hidden), "LMHEAD_CORES": str(n)}},
         "lm_head_q4_f32": {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{rows}_f32",
                            "env": {"LMHEAD_N": str(rows), "LMHEAD_K": str(spec.hidden),
                                    "LMHEAD_CORES": str(n), "LMHEAD_SCALES_F32": "1"}}}
    for kind, name in (("local", "dx"), ("global", "dxg")):
        bd = f"dense/build_gemma4_{kind}_h{spec.hidden}"
        b[name] = {"design": "dense/dx.py", "build_dir": bd, "env": {KIND_ENV: kind}}
        b[name + "_f32"] = {"design": "dense/dx.py", "build_dir": bd + "_f32",
                            "env": {KIND_ENV: kind, "GEMV_SCALES_F32": "1"}}
    return b


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    """The local layers' rows are the model-wide defaults; the global kernels override theirs."""
    out = D.manifest_layout(sub_spec(spec, "local"), max_ctx)
    out["pool_bytes"] = pool_bytes(spec)
    if spec.final_softcap:
        out["final_logit_softcapping"] = spec.final_softcap
    return out


def hf_config_check(spec: ModelSpec) -> dict:
    """The text tower's fields the kernels bake in (config.json nests them under text_config)."""
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "intermediate_size": spec.intermediate, "head_dim": spec.head_dim,
            "global_head_dim": spec.global_head_dim or spec.head_dim,
            "num_global_key_value_heads": spec.global_kv_heads, "attention_k_eq_v": spec.attn_k_eq_v,
            "sliding_window": spec.sliding_window, "final_logit_softcapping": spec.final_softcap or None}


GEN_KERNELS = D.GEN_KERNELS
KERNEL_SOURCES = list(D.KERNEL_SOURCES)
