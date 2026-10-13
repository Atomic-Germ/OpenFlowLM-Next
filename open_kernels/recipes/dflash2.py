"""The DFlash 2 drafter family (OPEN-DRAFT-DFLASH2): every matrix of the drafter, and the target's
borrowed lm head, as one multi-row ternary GEMV (designs/gemv_t2n) -- one xclbin, one hardware
context, one instruction stream per (N, K). Everything between the GEMVs runs on the host
(src/open_qwen36/dflash2.cpp). Any drafter of this geometry links to the same set.

The manifest's `layout.draft` says which stream serves which op and how its work is split (the
plan the stream was built from, which the engine needs to build the tables and add up y), and
`pack.weights` how each weight buffer is packed from the container (t2_perm, OPEN-QUANT-T2).
"""
from __future__ import annotations

import sys
from pathlib import Path

from .catalogue import require
from .spec import ModelSpec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "designs" / "gemv_t2n"))
from t2n_plan import plan  # noqa: E402

T2_CHUNK = 2176
N_CORES = 8
KT_MAX = 8
HEAD_OP = "head"


def _ops(spec: ModelSpec) -> dict[str, tuple[int, int]]:
    """op -> (N, K). kv is every layer's k|v at once (the KV injection of a committed row)."""
    h, ff = spec.hidden, spec.intermediate
    q, kv = spec.attn_q_width, spec.attn_kv_width
    taps = len(spec.extra["dflash2"]["target_layer_ids"])
    nconv = 2 * spec.extra["dflash2"]["conv_kernel_size"] * h // spec.extra["dflash2"]["conv_group_size"]
    return {
        "fc": (h, taps * h),
        "kv": (spec.num_layers * 2 * kv, h),
        "conv": (nconv, h),
        "qkv": (q + 2 * kv, h),
        "o": (h, q),
        "gu": (2 * ff, h),
        "down": (h, ff),
        HEAD_OP: (spec.vocab, h),
    }


def kernel_name(n: int, k: int) -> str:
    return f"t2n_n{n}_k{k}"


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    if spec.family != "dflash2":
        raise ValueError(f"dflash2 recipe given a {spec.family} spec")
    if spec.quant != "t2" or not spec.hadamard:
        raise ValueError("the dflash2 family runs only the ternary drafter (t2 in the Hadamard basis)")
    ops = _ops(spec)
    for name, (n, k) in ops.items():
        require("gemv_t2n", N=n, K=k)
        if k % 1024:
            raise ValueError(f"dflash2 {name}: K {k} is not whole 1024-blocks of the Hadamard transform")
    return {"ops": ops}


def layout(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    return recipe(spec, max_ctx)


def _t2(tensor: str, rows: int, cols: int, dst: int) -> dict:
    return {"op": "t2_perm", "tensor": tensor, "dst": dst, "nch": rows * cols // 8192, "in_dim": cols,
            "chunk_bytes": T2_CHUNK}


def pack_plan(spec: ModelSpec) -> dict:
    """Weight buffers: `fc` and `kv` once, the rest per layer ("{l}"). Each is t2_perm runs back to back."""
    h, ff = spec.hidden, spec.intermediate
    q, kv = spec.attn_q_width, spec.attn_kv_width
    ops = _ops(spec)

    def buf(parts):
        out, dst = [], 0
        for tensor, rows, cols in parts:
            out.append(_t2(tensor, rows, cols, dst))
            dst += rows * cols // 8192 * T2_CHUNK
        return {"bytes": dst, "pack": out}

    w = {"fc": buf([("fc.weight", h, ops["fc"][1])]),
         "kv": buf([(f"blk.{i}.attn_{x}.weight", kv, h) for i in range(spec.num_layers) for x in ("k", "v")])}
    per_layer = {
        "conv_attn": buf([("blk.{l}.attn_conv_proj.weight", ops["conv"][0], h)]),
        "qkv": buf([("blk.{l}.attn_q.weight", q, h), ("blk.{l}.attn_k.weight", kv, h),
                    ("blk.{l}.attn_v.weight", kv, h)]),
        "o": buf([("blk.{l}.attn_output.weight", h, q)]),
        "conv_ffn": buf([("blk.{l}.ffn_conv_proj.weight", ops["conv"][0], h)]),
        "gu": buf([("blk.{l}.ffn_gate.weight", ff, h), ("blk.{l}.ffn_up.weight", ff, h)]),
        "down": buf([("blk.{l}.ffn_down.weight", h, ff)]),
    }
    return {"layer_types": {}, "weights": w, "layer_weights": per_layer}


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    shapes = sorted(set(_ops(spec).values()))
    first = kernel_name(*shapes[0])
    kernels = {kernel_name(n, k): {"context": "draft", "insts": f"{kernel_name(n, k)}/insts.bin",
                                   "build": kernel_name(n, k)} for n, k in shapes}
    return {"contexts": {"draft": f"{first}/final.xclbin"}, "kernels": kernels, "layer_types": {},
            "tail": [], "globals": {}}


def builds(spec: ModelSpec) -> dict[str, dict]:
    return {kernel_name(n, k): {"design": "gemv_t2n/gemv_t2n.py", "build_dir": f"gemv_t2n/build_n{n}_k{k}",
                                "env": {"T2N_N": str(n), "T2N_K": str(k), "T2N_CORES": str(N_CORES),
                                        "T2N_KT_MAX": str(KT_MAX)}}
            for n, k in sorted(set(_ops(spec).values()))}


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    gemv = {}
    for op, (n, k) in _ops(spec).items():
        gemv[op] = {"kernel": kernel_name(n, k), **plan(n, k, N_CORES, KT_MAX)}
    d = spec.extra["dflash2"]
    return {"hidden": spec.hidden, "vocab": spec.vocab, "real_vocab": spec.real_vocab,
            "chunk_bytes": T2_CHUNK, "rope_theta": spec.rope_theta, "rotary_dim": spec.rotary_dim,
            "draft": {"gemv": gemv, "head_weight": "target:lmpool", **d}}


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "intermediate_size": spec.intermediate, "head_dim": spec.head_dim,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads}


GEN_KERNELS = None          # no generated kernel TUs: one design, its shape is the instruction stream
KERNEL_SOURCES = ["designs/gemv_t2n/*.py", "designs/gemv_t2n/*.cc", "build_design.py"]
Q8_ROLES = frozenset()
