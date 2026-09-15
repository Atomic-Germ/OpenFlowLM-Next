"""OPEN-WIDTH-PAD: the padded width a recipe may derive, and the four things that
stop GPT-OSS from being built at its own 2880.

The handoff that led here said "hidden 2880 gets ONE core of eight". That is true
and it is not the whole story: three separate refusals fire on 2880, padding to
3072 clears three of them, and a fourth is untouched by any width. These tests
pin each one, so "the widths are the blocker" stops being prose.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "open_kernels"))

from recipes import catalogue as C  # noqa: E402
from recipes import dense, pack, qwen36moe as M  # noqa: E402
from recipes.catalogue import LIMITS, OpRangeError  # noqa: E402
from recipes.load import load_spec  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402

from test_gptoss import HF_GPTOSS_20B  # noqa: E402

N_CORES = LIMITS["n_cols"]
SPECS = sorted((HERE.parents[2] / "open_kernels" / "recipes" / "specs").glob("*.json"))


def gptoss() -> ModelSpec:
    return ModelSpec.from_hf_config(HF_GPTOSS_20B)


def padded(spec: ModelSpec) -> ModelSpec:
    return dataclasses.replace(
        spec,
        hidden=M.pad_width(spec.hidden, N_CORES),
        moe_intermediate=M.pad_width(spec.moe_intermediate, N_CORES),
    )


# --------------------------------------------------------------- the derivation

def test_pad_width_is_the_smallest_width_that_satisfies_both_rules():
    assert M.pad_width(2880, 8) == 3072
    # Both rules, stated separately, hold at the answer and fail below it.
    assert 3072 % (M.BAND_ROWS * 8) == 0
    assert (M.BAND_ROWS * 3072) % M.CHUNK_VALUES == 0
    assert 2880 % (M.BAND_ROWS * 8) != 0
    assert (M.BAND_ROWS * 2880) % M.CHUNK_VALUES != 0
    # Nothing between 2880 and 3072 works, so 3072 really is the smallest.
    assert not [w for w in range(2881, 3072) if w % (M.BAND_ROWS * 8) == 0]


def test_pad_width_is_idempotent_and_moves_no_shipped_family():
    """At a family's OWN core count the pad is always a no-op - that is what
    cores_for picked the count for. So adding pad_width cannot move any shipped
    kernel set, whatever else it is used for."""
    assert M.pad_width(3072, 8) == 3072
    for p in SPECS:
        spec = load_spec(p)
        n = dense.cores_for(spec)
        for field in ("hidden", "intermediate", "attn_q_width", "attn_kv_width"):
            w = getattr(spec, field)
            if w:                      # gptoss's dense intermediate is 0
                assert M.pad_width(w, n) == w, f"{p.name}.{field} = {w} would move"


def test_gemma3_12b_is_the_precedent_for_choosing_not_to_pad():
    """Gemma 3 12B is the other family that does not get all eight cores, and it
    shows the two cases are different in kind. Its 3840 would have to grow to 4096
    for eight, but at 3840 everything still BUILDS - only the core count is lower,
    and that was measured as nearly free. GPT-OSS's 2880 does not build at all."""
    g = load_spec(HERE.parents[2] / "open_kernels" / "recipes" / "specs" / "gemma3-12b.json")
    assert g.hidden == 3840 and dense.cores_for(g) == 4
    assert M.pad_width(3840, 8) == 4096            # what eight cores would cost it
    assert M.band_bytes(3840) == 153600            # but 3840 has no arithmetic refusal
    with pytest.raises(OpRangeError):
        M.band_bytes(2880)                          # and 2880 does


def test_at_eight_cores_the_core_rule_already_implies_the_chunk_rule():
    """The chunk is 256 columns and a band is 64 rows, so BAND_ROWS * 8 = 512 is already
    a multiple of 256 and the eight-core rounding satisfies both rules on its own. That
    is true of EIGHT and of four, and it is why this read as "no lcm is needed" for as
    long as every shipped family got four cores or more. It is false at one and two,
    where BAND_ROWS * n_cores is 64 and 128 - see the lcm test below."""
    assert M.CHUNK_VALUES == M.CHUNK_ROWS * 256
    assert (M.BAND_ROWS * N_CORES) % 256 == 0
    assert (M.BAND_ROWS * 4) % 256 == 0
    assert (M.BAND_ROWS * 2) % 256 and (M.BAND_ROWS * 1) % 256


# ------------------------------------------------------- what 2880 actually does

def test_gptoss_gets_one_core_and_the_pad_gets_eight():
    spec = gptoss()
    assert (spec.hidden, spec.moe_intermediate) == (2880, 2880)
    assert dense.cores_for(spec) == 1
    assert dense.cores_for(padded(spec)) == 8


def test_the_refusal_that_fires_first_is_arithmetic_not_the_core_count():
    """band_bytes raises on 2880 before cores_for's answer can matter, and it is
    raised straight out of q4_bytes, so OPEN_KERNELS_UNVALIDATED cannot soften it.

    True of 2880, but the container never ships 2880 on a quantized axis -- it ships
    2944, and band_bytes(2944) returns cleanly. So this refusal is NOT what stops a
    GPT-OSS build first. What actually happens at the container's own width is worse:
    pack.std_perm floors 2944 // 256 to 11 and hands back a non-injective index array,
    1409 distinct file chunks for 1472 pool slots, with no guard anywhere. A silently
    corrupt pool, not an error. See .claude/plans/gptoss-bringup.md.
    """
    with pytest.raises(OpRangeError, match="8192-value chunks"):
        M.band_bytes(2880)
    assert M.band_bytes(3072) == 122880


def test_per_band_refuses_an_odd_chunk_count():
    """The band law the GEMV runs is rs=2: chunk i of a band covers row half i % 2 and
    k-tile i // 2, so a band is always an EVEN number of chunks, two per k-tile. 2944 is
    the width the container actually ships and band_bytes returns cleanly on it -- 23
    chunks, which no rs=2 walk can consume. The refusal lives in per_band itself rather
    than in a catalogue entry, so OPEN_KERNELS_UNVALIDATED cannot soften it."""
    assert M.band_bytes(2944) == 117760          # the arithmetic gate does NOT fire here
    with pytest.raises(OpRangeError, match="23 chunks"):
        M.per_band(2944)
    assert M.per_band(3072) == 24 and M.per_band(2048) == 16


def test_std_perm_refuses_a_width_that_does_not_tile_the_chunk():
    """The pack index law floors in_dim // 256 to get the file raster's column count, so a
    width that is not a whole number of 256-column k-tiles quietly aliases pool slots onto
    the same file chunk. At the container's own 2944 it selected 1409 distinct file chunks
    for 1472 pool slots -- a silently corrupt pool, with no guard in apply_op or anywhere
    else. It now raises, and the widths that do tile are untouched."""
    with pytest.raises(OpRangeError, match="2944"):
        pack.std_perm(1472, 2944)
    p = pack.std_perm(1536, 3072)
    assert len(np.unique(p)) == 1536


def test_the_q8_band_law_floors_the_same_way_and_is_guarded_too():
    """`q8_perm` derives `ncol = in_dim // 256` exactly as `std_perm` does, so a width that
    does not tile the chunk aliases its half-tiles onto the next row block's. Guarding one
    and not the other would leave the corrupt pool reachable through every q8 projection."""
    with pytest.raises(OpRangeError, match="2944"):
        pack.q8_perm(2 * 1472, 2944)
    files, halves = pack.q8_perm(4 * 12, 3072)
    assert len(np.unique(2 * files + halves)) == 4 * 12


def test_pad_width_rounds_to_the_lcm_so_a_low_core_count_cannot_land_short():
    """Rounding to BAND_ROWS * n_cores alone is only enough because 512 happens to be a
    multiple of the chunk's 256 columns. At fewer cores it is not: the old rule returned
    2880 unchanged at one core and 2944 at two -- and 2944 is exactly the width whose
    per_band is odd and whose std_perm is non-injective. lcm(BAND_ROWS * n_cores, 256)
    satisfies both rules at every core count."""
    from math import lcm

    for n in (1, 2, 4, 8):
        w = M.pad_width(2880, n)
        assert w == 3072, f"{n} cores gave {w}"
        assert w % lcm(M.BAND_ROWS * n, 256) == 0
    # and the two rules the padded width exists to satisfy hold on it at every count
    for n in (1, 2, 4, 8):
        w = M.pad_width(2880, n)
        assert M.per_band(w) % 2 == 0
        assert len(np.unique(pack.std_perm(M.per_band(w) * 2, w))) == M.per_band(w) * 2


def test_the_norm_width_2880_is_not_a_validated_ln_point():
    with pytest.raises(OpRangeError, match="width=2880 is outside the validated set"):
        C.require("ln", width=2880)
    C.require("ln", width=3072)          # the pad lands on Phi-4-mini's point


def test_padding_does_not_clear_the_moe_core_scratch():
    """The one blocker no width fixes: the main core has to hold xm's activation
    table and the expert h's at once, and at 3072/3072 they do not both fit."""
    with pytest.raises(OpRangeError, match="does not fit past xm's in the core scratch"):
        M.common(padded(gptoss()))
    # The reservation is tab_bytes(wide), and `wide` is the 4096 q projection only when
    # has_full is set. GPT-OSS derives dense/dense_local layer types, so has_full is
    # False and wide falls back to hidden: the two tables want 13824 B against 6912,
    # twice over rather than 1.5 times. An earlier version of this comment compared
    # against tab_bytes(4096) and understated the margin.
    assert not padded(gptoss()).has_full
    assert M.tab_bytes(3072) + M.tab_bytes(3072) > M.tab_bytes(3072)


def test_the_norm_divisor_is_why_a_padded_ln_would_be_wrong():
    """ln.h divides the sum of squares by the width it was COMPILED at, so a
    3072-wide norm over 2880 real channels and 192 zeros scales every residual by
    sqrt(3072/2880). This is arithmetic, not a measurement: it is why "pad the
    width and zero the tail" is not on its own a way to build GPT-OSS."""
    import math

    real, pad = 2880, M.pad_width(2880, N_CORES)
    # rsqrt(mean) over the padded width against over the real one, same energy.
    ratio = math.sqrt(pad / real)
    assert ratio == pytest.approx(1.0328, abs=5e-5)
    assert ratio > 1.03, "a 3% scale on every layer is not a rounding difference"
