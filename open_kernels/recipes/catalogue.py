"""The op catalogue: each kernel template, the parameters it takes, and the
points it has been VALIDATED at (a design's make_test.py / compare.py, or a
whole-layer run against the fp64 replica). A recipe asking for a point outside
a template's validated set fails here, at generation time, with the constraint
named -- never at run time with garbage.

A template that is parametric in the source but only tested at one point lists
that one point: `attn` takes HD / NH / KVH as compile-time macros, and its set
grows when a new point has been run and compared, not when the macro exists.

Also here: the physical limits a generated design must respect (checked, not
derived -- see the plan's "design geometry" item).

Traces: OPEN-OP-RANGE (specs/open-engine/spec.md).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


_WARNED: set = set()


class OpRangeError(ValueError):
    """A recipe asked a template for a parameter outside its validated set."""


@dataclass(frozen=True)
class Param:
    """One parameter of a template: an explicit set of validated values, or a
    predicate with a description of the rule."""
    values: frozenset | None = None
    rule: Callable[[Any], bool] | None = None
    describe: str = ""

    def ok(self, v: Any) -> bool:
        if self.values is not None:
            return v in self.values
        return bool(self.rule(v)) if self.rule else True

    def expected(self) -> str:
        if self.values is not None:
            return "{" + ", ".join(str(x) for x in sorted(self.values)) + "}"
        return self.describe


def values(*vs) -> Param:
    return Param(values=frozenset(vs))


def multiple_of(m: int) -> Param:
    return Param(rule=lambda v: isinstance(v, int) and v > 0 and v % m == 0, describe=f"a positive multiple of {m}")


def any_positive() -> Param:
    return Param(rule=lambda v: v > 0, describe="any positive value (host-side)")


@dataclass(frozen=True)
class Combination:
    """A group of parameters validated only TOGETHER.

    Checking a parameter at a time lets a never-run COMBINATION of individually
    validated values through: (head_dim 128, num_heads 16, num_kv_heads 8) passed
    silently because 128 came from Qwen3-4B, 16 from the 27B and 8 from Qwen3-4B,
    and no model had ever asked for that GQA grouping. A combination lists the
    configurations a family procedure has actually compared, one tuple each.

    `defaults` fills in a key a call site does not pass (older `require` sites
    predate the knob, and its absence means the un-knobbed behaviour).
    """
    keys: tuple[str, ...]
    points: frozenset
    defaults: dict = field(default_factory=dict)

    def value(self, kw: dict) -> tuple:
        return tuple(kw[k] if k in kw else self.defaults[k] for k in self.keys)

    def expected(self) -> str:
        return "{" + ", ".join(str(p) for p in sorted(self.points, key=repr)) + "}"


def combination(*points, defaults: dict | None = None, keys: tuple[str, ...]) -> Combination:
    return Combination(keys=keys, points=frozenset(points), defaults=defaults or {})


@dataclass(frozen=True)
class Template:
    name: str
    source: str                          # where the kernel lives (open_kernels/designs/...)
    params: dict[str, Param] = field(default_factory=dict)
    note: str = ""
    combos: tuple[Combination, ...] = ()  # groups checked as a whole, not one key at a time

    def _refuse(self, msg: str) -> None:
        msg = msg + (f" ({self.note})" if self.note else "")
        if os.environ.get("OPEN_KERNELS_UNVALIDATED"):
            # the way a new point gets validated: build it, run its fixture, then add it here
            if msg not in _WARNED:
                _WARNED.add(msg)
                print(f"catalogue: UNVALIDATED point allowed by OPEN_KERNELS_UNVALIDATED: {msg}", file=sys.stderr)
            return
        raise OpRangeError(msg)

    def require(self, **kw) -> None:
        grouped = {k for c in self.combos for k in c.keys}
        for k, v in kw.items():
            if k in grouped:
                continue
            if k not in self.params:
                raise OpRangeError(f"{self.name}: no parameter {k!r} "
                                   f"(has {sorted(set(self.params) | grouped)})")
            p = self.params[k]
            if not p.ok(v):
                self._refuse(f"{self.name}: {k}={v!r} is outside the validated set {p.expected()}")
        for c in self.combos:
            missing = [k for k in c.keys if k not in kw and k not in c.defaults]
            if missing:
                raise OpRangeError(f"{self.name}: require() is missing {missing} -- "
                                   f"{list(c.keys)} are validated as one combination")
            v = c.value(kw)
            if v not in c.points:
                self._refuse(f"{self.name}: {tuple(c.keys)} = {v} is outside the validated "
                             f"combinations {c.expected()}")


CATALOGUE: dict[str, Template] = {t.name: t for t in [
    Template("gemv_q4", "designs/gemv_q4/gemv_q4.h",
             {"K": values(1024, 2048, 3072, 3584, 3840, 4096, 2560, 6144, 8192, 9216, 9728, 10240,
                          10752, 12288, 14336, 15360),
                                                 # 2048 / 4096: the 27B layers; 2560 / 9728: Qwen3-4B;
                                                 # 14336: Llama 3.1 8B; 10240: Gemma 3 4B; 12288: Qwen3-8B;
                                                 # 6144: Qwen3-1.7B; 1024 / 3072: Qwen3-0.6B;
                                                 # 3072 / 8192: Llama 3.2 3B; 8192: Llama 3.2 1B; 8192: Granite 4.2 3B;
                                                 # 9216: Qwen3.5 4B; 6144: Qwen3.5 2B; 3584: Qwen3.5 0.8B;
                                                 # 10752: Nanbeige4.1-3B (OPEN-FAMILY-LLAMA3, 2026-09-10);
                                                 # 3840 / 15360: Gemma 3 12B
              "rs": values(2, 4),                # band row split: standard layout / expert stripes
              "rows_per_core": multiple_of(64),  # one y element per 64-row band
              "per_call": values(2, 1)},         # chunks per w element (10 KB; 5 KB when the table is wide)
             note="a new K needs gemv_q4/make_test.py + compare.py at that K first"),
    Template("gemv_q8", "designs/gemv_q4/gemv_q8.h",
             {"K": values(2048, 4096),           # 2048: the 35B's qkv / z / q / k / v / gate;
                                                 # 4096: its out and o projections. Both entered with
                                                 # OPEN-QUANT-Q8's hardware pass (Ornith-1.0-35B-A3B).
              "rs": values(4),                   # a q8 band is four 16-row half-tiles per k-tile
              "rows_per_core": multiple_of(64),  # one y element per 64-row band, as the q4 GEMV
              "per_call": values(2, 1)},         # half-tiles per w element (same 5120 B element)
             note="a q8 projection streams 16-row half-tiles of the container's chunks; a new K needs "
                  "the OPEN-QUANT-Q8 procedure (the 8-layer slice against the container's own q8 values) "
                  "at that K first"),
    Template("gemv_q4_prep_f32", "designs/gemv_q4/gemv_tab.h",
             {"K": values(512)},                 # the expert hidden h (moe_experts' law)
             note="validated through the whole-layer MoE block only"),
    Template("attn", "designs/attn/attn.h",
             {"rope_theta": any_positive()},
             combos=(combination(
                 # (head_dim, num_heads, num_kv_heads, rotary_dim, qk_norm, attn_gate, qk_norm_post_rope)
                 (256, 16, 2, 64, True, True, False),      # the 27B / 35B MoE (OPEN-FAMILY-QWEN36MOE)
                 (128, 32, 8, 128, True, False, False),    # Qwen3-4B and Qwen3-8B (OPEN-FAMILY-QWEN3)
                 (128, 32, 8, 128, False, False, False),   # Llama 3.1 8B (OPEN-FAMILY-LLAMA3)
                 (256, 8, 4, 256, True, False, False),     # Gemma 3 4B, a 1024-row window (OPEN-FAMILY-GEMMA3)
                 (256, 16, 8, 256, True, False, False),    # Gemma 3 12B: 16 heads over 8 kv, RB 1 (OPEN-FAMILY-GEMMA3)
                 (128, 32, 8, 128, True, False, True),     # Hy-MT2-7B, qk-norm AFTER RoPE (OPEN-FAMILY-HUNYUAN)
                 (128, 16, 8, 128, True, False, False),    # Qwen3-1.7B / 0.6B: a GQA group of 2 (OPEN-FAMILY-QWEN3)
                 (128, 24, 8, 128, False, False, False),   # Llama 3.2 3B: GQA group 3, OG_AOUT_ELEMS 3 (OPEN-FAMILY-LLAMA3)
                 (64, 32, 8, 64, False, False, False),     # Llama 3.2 1B: head dim 64 (OPEN-FAMILY-LLAMA3)
                 (256, 16, 4, 64, True, True, False),      # Qwen3.5 4B: gated, partial RoPE 64 (OPEN-FAMILY-QWEN35)
                 (256, 8, 2, 64, True, True, False),       # Qwen3.5 2B / 0.8B: the same, 8 heads over 2 kv (OPEN-FAMILY-QWEN35)
                 (64, 40, 8, 64, False, False, False),    # Granite 4.2 3B (OPEN-FAMILY-GRANITE)
                 (128, 20, 4, 128, False, False, False),  # Nanbeige4.1-3B: GQA group 5, two q heads per element (OPEN-FAMILY-LLAMA3)
                 (128, 24, 8, 96, False, False, False),   # Phi4-mini: 96 of 128 dims rotated, the 16-lane tail (OPEN-FAMILY-PHI3)
                 keys=("head_dim", "num_heads", "num_kv_heads", "rotary_dim",
                       "qk_norm", "attn_gate", "qk_norm_post_rope"),
                 # the MoE and Qwen3.5 recipes predate the post-RoPE knob and never set it
                 defaults={"qk_norm_post_rope": False}),),
             note="ATTN_* are compile-time macros, and the geometry is validated as a WHOLE "
                  "tuple: checking head_dim / num_heads / num_kv_heads one at a time let "
                  "(128, 16, 8) -- a GQA group of 2 at head dim 128 -- through silently, "
                  "because each value came from a different model. A new tuple needs a family "
                  "procedure run at exactly that configuration"),
    Template("deltanet", "designs/layer_x/dnx.h",
             {"heads": values(16, 32), "dim": values(128), "key_heads": values(16), "conv_kernel": values(4)},
             note="Qwen3-Next / 3.5 / 3.6 families only; heads/key_heads is the value heads per "
                  "key head (dn_glue.h kGrp) and 16 heads pack the alpha/beta projection padded "
                  "to 32 lanes; 16 entered with OPEN-FAMILY-QWEN35's 2B / 0.8B pass"),
    Template("ln", "designs/ln/ln.cc",
             {"width": values(1024, 2048, 2560, 3072, 3840, 4096)}),
                                # LN_N; 1024 / 2048 take the fused single-core path (N <= 2048),
                                # 2560 / 3072 / 4096 the split-output one
    Template("router", "designs/router/router.h",
             {"experts": values(256), "topk": values(8)}),
    Template("moe", "designs/layer_x/moe_*.cc",
             {"ff": values(512), "experts": values(256), "topk": values(8), "shared_expert": values(True),
              "hidden": values(2048), "n_cores": values(8)}),
    Template("lm_head_q8", "designs/lm_head_q8/lm_head_q8.h",
             {"K": values(1024, 2048, 2560, 4096), "vocab": multiple_of(128)},
             note="K is LMHEAD_K, a compile-time knob of the kernel and of the pool's supertile "
                  "order (pack.py lmhead_q8 reads it as in_dim); 2560 entered with OPEN-FAMILY-QWEN35, "
                  "then 4096 (the 9B) and 1024 (the 0.8B) with the rest of that family"),
    Template("lm_head_q4", "designs/lm_head_q4/lm_head_q4.py",
             {"K": values(1024, 2048, 2560, 3072, 3840, 4096), "vocab": multiple_of(64)},
             note="the q4 head is the gemv_q4 kernel with lm_head_q8's uneven band split; validated per K"),
]}


# A container that MIXES weight formats puts both GEMV bodies on the same main core
# (OPEN-QUANT-Q8's fold), and that core has LIMITS["program_bytes"] of program memory.
# Whether the pair fits is not derivable -- only aiecc can measure it, and it does not
# print the shortfall -- so this is a validated set like any other: the (family, hidden)
# widths whose mixed core HAS been built. Qwen3.5's 4B (hidden 2560) is the one size that
# overflowed, and it is the only width here whose glue side channel carries two unequal
# halves, so lx.py emits two projection walks instead of one. Both flag levers (-Oz on the
# GEMV translation units, then on the whole core) were spent without recovering it.
# See .claude/plans/q8m-hw-results.md section 2.
MIXED_CORE_FITS = frozenset({
    ("qwen35", 4096),    # Qwen3.8-Distilled-9B-NPU2
    ("qwen35", 2048),    # Qwen3.8-Distilled-2B-NPU2
    ("qwen35", 1024),    # Qwen3.5-0.8B-NPU2
})


def mixed_core_fits(family: str, hidden: int) -> bool:
    """Has a main core carrying BOTH weight formats' GEMV bodies been built at this width?
    A recipe that cannot answer yes narrows the container's q8 role back to q4_1 rather
    than handing the user an export that dies 60 s into aiecc (recipes/load.py)."""
    return (family, hidden) in MIXED_CORE_FITS


# Physical limits of one whole-layer design on the XDNA2 array (npu2, 8 columns):
# a generated design must CHECK these; they are not tunables.
LIMITS = {
    "max_buffer_args": 8,        # ERT state 6 above that (opcode-3 runs)
    "n_cols": 8,                 # main cores: one per column
    "shim_fills": 13,            # per design (phase 2 whole-layer plan)
    "shim_drains": 11,
    "program_bytes": 16384,      # per core; only the build can measure it
}


def require(op: str, **kw) -> Template:
    t = CATALOGUE.get(op)
    if t is None:
        raise OpRangeError(f"no template {op!r} in the catalogue (has {sorted(CATALOGUE)})")
    t.require(**kw)
    return t


def check_buffer_args(design: str, args: list[str]) -> None:
    if len(args) > LIMITS["max_buffer_args"]:
        raise OpRangeError(f"{design}: {len(args)} buffer arguments {args}, the runtime allows "
                           f"{LIMITS['max_buffer_args']}")
