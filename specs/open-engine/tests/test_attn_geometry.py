# Traces: OPEN-ATTN-CONTEXT (canonical spec: specs/open-engine/spec.md)
"""The attention geometry each dense family gets on the fast path -- softmax
exponentials on the vector unit, heads split over cores, rows blocked per call --
and that a family not yet measured on it keeps the single-core kernel it shipped.
The per-position cost itself is measured on the NPU (the spec's procedure); the
recipe's choice of geometry is what is checked here."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from recipes import dense as DR
from recipes.load import load_spec

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"

# spec file -> (ACORES, NHL, RB) on the fast path. Every row is a whole number of og
# elements per core and a block of 8 / 16 / 32 score lanes -- attn.h's two constraints.
FAST_GEOMETRY = {
    "qwen3-4b.json": (4, 8, 4),       # 32 heads / 8 kv at hd 128: 4 og elements
    "llama31-8b.json": (4, 8, 4),     # same shape, no qk norm
    "hy-mt2-7b.json": (4, 8, 4),      # same shape, norm after RoPE
    "gemma3-4b.json": (2, 4, 2),      # 8 / 4 at hd 256: 2 og elements; RB capped at 2 for hd 256
    "granite42-3b.json": (5, 8, 4),   # 40 / 8 at hd 64: 5 og elements (the measured family)
}


@pytest.fixture
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


@pytest.mark.parametrize("name", sorted(FAST_GEOMETRY))
def test_fast_path_geometry(name, unvalidated, monkeypatch):
    monkeypatch.setenv("ATTN_FAST", "1")
    G = DR.geometry(load_spec(SPECS / name))
    acores, nhl, rb = FAST_GEOMETRY[name]
    assert (G.VEXP, G.ACORES, G.NHL, G.RB) == (1, acores, nhl, rb), name
    assert G.NHL * G.ACORES == G.NH and G.NHL % G.HPO == 0
    assert G.RB * max(G.NHL, 8) in (8, 16, 32)
    assert G.MLS % 32 == 0 and G.MLS >= G.NHL


@pytest.mark.parametrize("name", ["qwen3-4b.json", "gemma3-4b.json"])
def test_unmeasured_family_keeps_the_shipped_kernel(name, unvalidated, monkeypatch):
    """Off the path a family compiles what it compiled before: one core, no VEXP, one row
    per call, ml packed. Flipping a family into FAST_ATTENTION is a deliberate edit backed
    by a measurement, never a side effect of a recipe change. Every dense family has been
    measured by now, so the list is emptied for the check."""
    from recipes import attnknobs
    monkeypatch.delenv("ATTN_FAST", raising=False)
    monkeypatch.setattr(attnknobs, "FAST_ATTENTION", ())
    G = DR.geometry(load_spec(SPECS / name))
    assert (G.VEXP, G.ACORES, G.NHL, G.RB, G.MLS) == (0, 1, G.NH, 1, G.NH), name


@pytest.mark.parametrize("name", ["granite42-3b.json", "qwen3-4b.json", "llama31-8b.json", "hy-mt2-7b.json", "gemma3-4b.json"])
def test_measured_family_is_on_the_path_without_the_probe(name, unvalidated, monkeypatch):
    """Granite (2026-09-07) and Qwen3 (2026-09-08, the first HD-128 point: 5050 -> 258 ms at
    position 2048, 300/300 greedy tokens identical to the shipped kernel)."""
    monkeypatch.delenv("ATTN_FAST", raising=False)
    G = DR.geometry(load_spec(SPECS / name))
    assert (G.VEXP, G.ACORES, G.NHL, G.RB) == (1,) + FAST_GEOMETRY[name]


def test_attn_fast_is_a_probe_variable(monkeypatch):
    """It changes the compiled kernel, so the build key has to see it (OPEN-BUILD-CACHE)."""
    assert "ATTN_FAST" in DR.PROBE_VARS
    monkeypatch.setenv("ATTN_FAST", "1")
    assert DR.probe_env() == {"ATTN_FAST": "1"}


@pytest.mark.parametrize("og_elems,cores", [(1, 1), (2, 2), (4, 4), (5, 5), (8, 4), (12, 6), (7, 1), (16, 4)])
def test_attn_cores_divides_and_fits(og_elems, cores):
    assert DR.attn_cores(og_elems) == cores
    assert og_elems % DR.attn_cores(og_elems) == 0 and DR.attn_cores(og_elems) <= DR.MAX_ATTN_CORES


# ---- the MoE / Qwen3.5 recipe compiles the same attn.h through designs/layer_x/ax.py
from recipes import qwen36moe as Q36  # noqa: E402
from recipes import qwen35 as Q35  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# spec -> (ACORES, NHL, RB): 16 / 2 / 256 (the 35B and the 27B) is 8 og elements -> 4 cores
# of 4 heads; Qwen3.5-9B's 16 / 4 / 256 is 4 og elements -> 4 cores of 4; the 0.8B's 8 / 2 /
# 256 is 4 og elements -> 4 cores of 2, padded to 8 lanes; hd 256 with the gate: RB 1.
MOE_FAST = {"qwen36-35b-a3b.json": (4, 4, 1), "qwen35-9b.json": (4, 4, 1)}   # hd 256 + gate: no row block


@pytest.mark.parametrize("name", sorted(MOE_FAST))
def test_moe_attn_geometry_on_the_fast_path(name, unvalidated, monkeypatch):
    monkeypatch.setenv("ATTN_FAST", "1")
    A = Q36.attn(load_spec(SPECS / name))
    acores, nhl, rb = MOE_FAST[name]
    assert (A.VEXP, A.ACORES, A.NHL, A.RB) == (1, acores, nhl, rb), name
    assert A.NHL * A.ACORES == A.NH and A.NHL % A.HPO == 0 and (A.RB == 1 or A.RB * max(A.NHL, 8) in (8, 16, 32))
    assert A.MLS % 32 == 0 and A.MLS >= A.NHL


@pytest.mark.parametrize("name", sorted(MOE_FAST))
def test_moe_attn_geometry_off_the_path_is_the_shipped_kernel(name, unvalidated, monkeypatch):
    """The 27B / 35B kernels (OPEN-LAYOUT-FREEZE) compile what they compiled before until
    measured: one core, no VEXP, one row per call. Qwen3.5 is measured (the 0.8B), so the
    list is emptied for the check."""
    from recipes import attnknobs
    monkeypatch.delenv("ATTN_FAST", raising=False)
    monkeypatch.setattr(attnknobs, "FAST_ATTENTION", ())
    A = Q36.attn(load_spec(SPECS / name))
    assert (A.VEXP, A.ACORES, A.NHL, A.RB, A.MLS) == (0, 1, A.NH, 1, A.NH), name


def test_qwen35_0p8b_fast_geometry_is_an_eight_lane_block(unvalidated, monkeypatch):
    """qwen35 is in FAST_ATTENTION (measured on the 0.8B, 2026-09-08): no probe needed."""
    import json
    from recipes.spec import ModelSpec
    monkeypatch.delenv("ATTN_FAST", raising=False)
    cfg = json.loads((FIXTURES / "config_qwen35_0p8b.json").read_text())
    spec = ModelSpec.from_hf_config(cfg)
    A = Q35.recipe(spec).attn
    assert (A.NH, A.KVH, A.HD, A.HPO) == (8, 2, 256, 2)
    assert (A.VEXP, A.ACORES, A.NHL, A.RB) == (1, 4, 2, 1)


def test_every_family_module_exposes_the_probe_hook():
    """cache.py folds `probe_env()` off the family module into the build key; a module
    without it would let an ATTN_FAST probe build share a key with the real one."""
    from recipes.families import FAMILIES, family_module
    for fam in FAMILIES:
        assert callable(getattr(family_module(fam), "probe_env", None)), fam
