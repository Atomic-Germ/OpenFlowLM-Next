"""The fast attention path's knobs -- softmax exponentials batched on the vector unit
(ATTN_VEXP), heads split over cores (ATTN_NHL / ACORES), cached rows blocked per call
(ATTN_RB); OPEN-ATTN-CONTEXT -- shared by the dense recipe (designs/dense/dx.py) and the
MoE / Qwen3.5 one (designs/layer_x/ax.py), which compile the same attn.h.

A family joins by measurement (model/sweep_positions.ps1 and the fp64 replica), not by
declaration: every family NOT in FAST_ATTENTION compiles the attention it compiled
before, byte for byte. ATTN_FAST=1 builds an unlisted family on the path for exactly
that measurement; it is a probe variable, so the build key sees it (recipes/cache.py)
and the probe build cannot be shipped as a real one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .catalogue import OpRangeError
from .spec import ModelSpec

# ATTN_NULL, ATTN_ABL, ATTN_RB and ATTN_FAST change the COMPILED kernel, and cache.build_key
# hashes sources + spec + quant, which cannot see an environment variable. Two exports
# that differ only in a probe would therefore share a key, and export_qwen36_kernels.py
# skips a build whose key the destination already has -- so a probe build could be
# shipped as a real one, silently. `probe_env()` is what cache.py folds in to stop that;
# it returns {} when nothing is set, so an ordinary build's key is unchanged.
PROBE_VARS = ("ATTN_NULL", "ATTN_ABL", "ATTN_RB", "ATTN_FAST")

RB_SUPPORTED = (1, 2, 4)      # attn_stepb.cc has bodies for 2 and 4; 1 is the unblocked path

# Measured: granite (2026-09-07, hd 64); qwen3 (2026-09-08, the first hd 128 point --
# 5050 -> 258 ms at position 2048, 300/300 greedy tokens identical to the shipped kernel);
# llama3 (2026-09-08, same shape without the qk norm -- 4024 -> 427 ms at 2048 on a loaded
# box, 54 identical greedy tokens then a 0.008-logit near-tie flip); hunyuan (2026-09-08,
# hd 128 with the norm after RoPE -- 3395 -> 165 ms at 2048, 43 identical tokens then a
# 0.05-logit near-tie flip); gemma3 (2026-09-08, hd 256 at RB 2 -- 512 -> 96 ms at 2048,
# 41 identical tokens then a 0.06-logit near-tie flip); qwen35 (2026-09-08, the ax design:
# 4 cores x 2 heads with the gate at RB 1 -- 176 -> 70 ms at 2048 on the 0.8B, 200/200 greedy
# tokens identical); qwen36moe (2026-09-08, 4 x 4 with the gate at RB 1 on a 16-layer prefix
# of the 35B -- 217 -> 43 ms part0 at 2048, 85 identical greedy tokens; the 8-layer prefix
# 60/60 -- the residual corr spread is the routed experts flipping on near-ties, not the
# attention).
# phi3 (2026-09-10, Phi4-mini: hd 128 with a 96-dim rotation, measured on the fast path
# against its slow-path pass -- see specs OPEN-FAMILY-PHI3).
FAST_ATTENTION = ("granite", "qwen3", "llama3", "hunyuan", "gemma3", "qwen35", "qwen36moe", "phi3")
# Both designs put attention core c at Tile(2 + c, 3) and its og drain at Tile(3 + c, 0):
# six columns for the split, whatever the head count.
MAX_ATTN_CORES = 6


@dataclass(frozen=True)
class AttnKnobs:
    VEXP: int; MLS: int                  # batched softmax exponentials; the ml stride
    ACORES: int; NHL: int; RB: int       # attention cores; heads each owns; cached rows per call


def probe_env() -> dict[str, str]:
    """The probe variables that are actually set, for the build key."""
    return {k: os.environ[k] for k in PROBE_VARS if os.environ.get(k)}


def fast_attention(spec: ModelSpec) -> bool:
    return spec.family in FAST_ATTENTION or os.environ.get("ATTN_FAST") == "1"


def attn_cores(nh: int) -> int:
    """Cores for the split: the largest divisor of the HEAD COUNT that fits the columns.

    It used to be the largest divisor of `og_elems = NH // HPO`, because a core had to
    own whole og elements. That made NHL exactly HPO in every family, and it cost
    Gemma 3 two thirds of the fabric: 8 heads over 4 kv is og_elems 2, so two cores, on
    a split that has room for six. attn.h's kOGH lets a core own fewer heads than an og
    element holds, so the only remaining requirement is that the heads divide evenly.
    """
    return max(d for d in range(1, min(nh, MAX_ATTN_CORES) + 1) if nh % d == 0)


def _probe_rb(default: int, nhl: int) -> int:
    """ATTN_RB, validated. Rejected rather than passed on to fail at compile time."""
    raw = os.environ.get("ATTN_RB")
    if raw is None:
        return default
    try:
        rb = int(raw)
    except ValueError:
        raise OpRangeError(f"ATTN_RB={raw!r} is not an integer (expected one of {RB_SUPPORTED})")
    if rb not in RB_SUPPORTED:
        raise OpRangeError(
            f"ATTN_RB={rb} is not supported: attn_stepb.cc has a body for 2 and 4, and 1 is the "
            f"unblocked path. Anything else compiles to `#error` after a full design build.")
    if rb > 1 and (rb * block_lanes(nhl)) not in (8, 16, 32):
        raise OpRangeError(
            f"ATTN_RB={rb} with {nhl} heads per core gives a {rb * block_lanes(nhl)}-lane score block, and the "
            f"block exponential needs 8, 16 or 32. Legal here: "
            f"{[r for r in RB_SUPPORTED if r == 1 or (r * block_lanes(nhl)) in (8, 16, 32)]}.")
    return rb


def block_lanes(nhl: int) -> int:
    """attn.h's kNL: the block kernel pads a core's head vector to 8 lanes, the narrowest
    fp32 vector, so a core owning 2 or 4 heads still exponentiates 8 lanes per row."""
    return max(nhl, 8)


def knobs(spec: ModelSpec, nh: int, hpo: int) -> AttnKnobs:
    """The knobs for a family with `nh` heads and `hpo` heads per og element.

    attn.h's online softmax spends two sexp() per head per position, and sexp is software
    float on the scalar unit. ATTN_VEXP batches them through the vector unit instead --
    same arithmetic, ~1e-7 either way. Attention then runs on ONE core while ~22 of the
    array's 32 sit idle, and it is still the whole decode step; heads are independent,
    so the work splits cleanly over attn_cores(). Rows per kernel call: one row per call
    reloads q for every head and reloads, rescales and stores the whole output
    accumulator at every position; a block pays those once for RB rows and
    exponentiates the whole block in one vector (RB * NHL lanes), which must be a whole
    vector of 8, 16 or 32."""
    fast = fast_attention(spec)
    vexp = 1 if fast else 0
    acores = attn_cores(nh) if fast else 1
    nhl = nh // acores
    mls = ((nhl + 31) // 32) * 32 if vexp else nhl
    rb = 1
    if vexp:
        # Four rows at head dim 64 / 128; two at 256, where the block kernel's unrolled
        # score and V loops are twice as long per row and four rows overflow the core's
        # 16 KB of program memory (Gemma3-4B, 2026-09-08: "Overflow of program memory");
        # none at 256 with the attention gate (the 35B, Qwen3.5), whose sigmoid puts the
        # exponential and the reciprocal on the same core -- the block kernel does not fit
        # beside them (Qwen3.5-0.8B, 2026-09-08). The vector softmax and the core split are
        # the large part of the gain; the block is the rest.
        # ... and none at 256 with EIGHT heads on a core (Gemma 3 12B: 16 heads over 8 kv
        # gives og elements of 8, so ACORES is 2 and NHL is 8). There the block kernel's
        # working set does not fit L1 at all -- `'aie.tile' op allocated buffers exceeded
        # available memory`, before any program-memory limit is reached. RB 1 builds and
        # costs little: RB 1 -> 2 measured 1.22x of the attention ARITHMETIC on granite,
        # and that arithmetic is ~18% of a decode step.
        cap = 4 if spec.head_dim < 256 else (1 if (spec.attn_gate or nhl > 2) else 2)
        rb = max((r for r in (4, 2, 1) if r <= cap and (r * block_lanes(nhl)) in (8, 16, 32)), default=1)
        rb = _probe_rb(rb, nhl)
    return AttnKnobs(VEXP=vexp, MLS=mls, ACORES=acores, NHL=nhl, RB=rb)
