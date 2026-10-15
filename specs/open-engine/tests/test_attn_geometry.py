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

# spec file -> (ACORES, NHL, RB) on the fast path. ACORES is the largest divisor of the
# HEAD COUNT that fits the columns (attnknobs.attn_cores); a core owns kOGH = min(NHL, HPO)
# heads of an og element, and a block is 8 / 16 / 32 score lanes -- attn.h's two constraints.
FAST_GEOMETRY = {
    "qwen3-4b.json": (4, 8, 4),       # 32 heads / 8 kv at hd 128: NHL == HPO, unchanged by the og split
    "llama31-8b.json": (4, 8, 4),     # same shape, no qk norm
    "hy-mt2-7b.json": (4, 8, 4),      # same shape, norm after RoPE
    "gemma3-4b.json": (4, 2, 2),      # 8 / 4 at hd 256: 2 heads a core once og is NHL wide (was 2 cores
                                      # of 4, the whole point of the split); RB capped at 2 for hd 256
    "granite42-3b.json": (5, 8, 8),   # 40 / 8 at hd 64: NHL == HPO, 8-row blocks (OPEN-ATTN-BLOCK8; ATTN_RB=4 is the old kernel)
    "phi4-mini-4b.json": (6, 4, 4),   # 24 / 8 at hd 128, a 96-dim rotation: 6 cores of 4 (was 3 of 8;
                                      # measured 2026-09-12, 123 -> 117 ms at 2048, 250/250 identical)
    "qwen25-3b.json": (4, 4, 4),      # 16 / 2 at hd 128: 4 cores of 4 heads. HPO is 2, so a core's
                                      # og element is 2 heads wide and it emits two of them -- the
                                      # kOGH split, on the narrowest attention element of any dense
                                      # family (512 B). Measured 2026-09-12.
    "gemma3-12b.json": (4, 4, 1),     # 16 / 8 at hd 256: 4 cores of 4; RB 1, the block kernel does not
                                      # fit L1 beside more than two heads at hd 256
    "lfm2-1.2b.json": (4, 8, 4),      # 32 / 8 at hd 64, q/k normed: the Llama 3.2 1B shape with the
                                      # norms. Only six of sixteen layers reach this kernel at all --
                                      # the rest are short_conv. Measured 2026-09-13.
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
    # attn.h's static_assert, exactly: the heads divide NH, and tile the og element
    # they are written through (kOGH = min(kNHL, kHPO)).
    assert G.NHL * G.ACORES == G.NH and G.NHL % min(G.NHL, G.HPO) == 0
    assert G.RB * max(G.NHL, 8) in (8, 16, 32, 64)
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


@pytest.mark.parametrize("name", ["granite42-3b.json", "qwen3-4b.json", "llama31-8b.json", "hy-mt2-7b.json",
                                  "gemma3-4b.json", "phi4-mini-4b.json", "gemma3-12b.json",
                                  "qwen25-3b.json", "lfm2-1.2b.json"])
def test_measured_family_is_on_the_path_without_the_probe(name, unvalidated, monkeypatch):
    """Granite (2026-09-07), Qwen3 (2026-09-08, the first HD-128 point: 5050 -> 258 ms at
    position 2048, 300/300 greedy tokens identical to the shipped kernel), and Phi-3
    (2026-09-10, the first partial-rotation point: 190 -> 84 ms/token on Phi4-mini)."""
    monkeypatch.delenv("ATTN_FAST", raising=False)
    G = DR.geometry(load_spec(SPECS / name))
    assert (G.VEXP, G.ACORES, G.NHL, G.RB) == (1,) + FAST_GEOMETRY[name]


@pytest.mark.parametrize("name,on", [("granite42-3b.json", True), ("qwen3-4b.json", False), ("llama31-8b.json", False),
                                     ("phi4-mini-4b.json", False)])
def test_score_tree_is_granite_only(name, on, unvalidated, monkeypatch):
    """The 8-heads-a-core score tree (attn.h ATTN_TREE8) is on for the measured family and byte-for-byte
    absent elsewhere; ATTN_TREE8 forces it either way for an A/B build."""
    from recipes import attnknobs
    monkeypatch.delenv("ATTN_TREE8", raising=False)
    spec = load_spec(SPECS / name)
    G = DR.geometry(spec)
    assert attnknobs.tree8(spec, G.NHL, G.RB, G.HD) is on
    monkeypatch.setenv("ATTN_TREE8", "0")
    assert attnknobs.tree8(spec, G.NHL, G.RB, G.HD) is False
    assert DR.probe_env() == {"ATTN_TREE8": "0"}


def test_kv_slices_cover_every_core_head():
    """Traces: OPEN-ATTN-KVSLICE. Granite's 5 cores x 8 heads touch 2 or 3 kv heads each: a 3-head window per core, the start
    clamped inside the row; at one kv head a core, slices of 2 start aligned so the two cores of a slice read one stream."""
    from recipes import attnknobs
    split = [k * 5 + o for k in range(8) for o in (0, 3)]      # Granite at two cores a kv head: its 5 heads as 3 + 2
    assert attnknobs.kv_slices(40, 8, 5) == (3, [0, 1, 3, 4, 5])
    assert attnknobs.kv_slices(40, 8, 8, 2) == (2, [0, 0, 2, 2, 4, 4, 6, 6])
    assert attnknobs.kv_slices(40, 8, 16, 2, split) == (2, [0] * 4 + [2] * 4 + [4] * 4 + [6] * 4)
    for (nh, kvh, ac, share, h0s) in ((40, 8, 5, 1, None), (40, 8, 8, 2, None), (40, 8, 16, 2, split), (32, 8, 4, 1, None),
                                      (24, 8, 6, 1, None), (16, 2, 4, 1, None)):
        kvs, kvb = attnknobs.kv_slices(nh, kvh, ac, share, h0s)
        grp, h0s = nh // kvh, h0s or [c * (nh // ac) for c in range(ac)]
        for c in range(ac):
            heads = range(h0s[c], (h0s + [nh])[c + 1])
            assert 0 <= kvb[c] and kvb[c] + kvs <= kvh
            assert all(kvb[c] <= h // grp < kvb[c] + kvs for h in heads), (nh, kvh, ac, c)


@pytest.mark.parametrize("name,rows", [("granite42-3b.json", 8), ("qwen3-4b.json", 0), ("qwen25-3b.json", 0), ("gemma3-4b.json", 0)])
def test_kvslice_is_measured_not_declared(name, rows, unvalidated, monkeypatch):
    """Traces: OPEN-ATTN-KVSLICE. On for Granite (the manifest's `kvslice` is its block rows), off elsewhere; ATTN_KVSLICE forces it
    off for an A/B build, and a q / k / v bias (Qwen2.5) or a sliding window (Gemma 3) keeps the broadcast even when forced."""
    from recipes import attnknobs
    monkeypatch.delenv("ATTN_KVSLICE", raising=False)
    spec = load_spec(SPECS / name)
    G = DR.geometry(spec)
    assert attnknobs.kvslice(spec, G.ACORES, G.RB, G.QKVB, G.WINDOW) == rows
    assert DR.programs(spec, 4096)["kernels"]["dx"].get("kvslice", 0) == rows
    monkeypatch.setenv("ATTN_KVSLICE", "0")
    assert attnknobs.kvslice(spec, G.ACORES, G.RB, G.QKVB, G.WINDOW) == 0
    assert DR.probe_env() == {"ATTN_KVSLICE": "0"}
    monkeypatch.setenv("ATTN_KVSLICE", "1")
    if G.QKVB or G.WINDOW:
        assert attnknobs.kvslice(spec, G.ACORES, G.RB, G.QKVB, G.WINDOW) == 0


@pytest.mark.parametrize("name,cores", [("granite42-3b.json", 2), ("qwen3-4b.json", 0), ("llama31-8b.json", 0)])
def test_kv_cores_is_measured_not_declared(name, cores, unvalidated, monkeypatch):
    """Traces: OPEN-ATTN-KVSLICE. Attention cores by kv head (attn.h ATTN_G5) for Granite only: 40 heads over 8 kv heads at
    hd 64, on the KVSLICE window; ATTN_KVCORES forces 0, 1 or 2 for an A/B build and is refused off that geometry."""
    from recipes import attnknobs
    for v in ("ATTN_KVCORES", "ATTN_KVSLICE"):
        monkeypatch.delenv(v, raising=False)
    spec = load_spec(SPECS / name)
    G = DR.geometry(spec)
    ks = attnknobs.kvslice(spec, G.ACORES, G.RB, G.QKVB, G.WINDOW)
    assert attnknobs.kv_cores(spec, G.NH, G.KVH, G.HD, G.RB, ks) == cores
    monkeypatch.setenv("ATTN_KVCORES", "1")
    assert attnknobs.kv_cores(spec, G.NH, G.KVH, G.HD, G.RB, ks) == (1 if cores else 0)
    assert attnknobs.kv_cores(spec, G.NH, G.KVH, G.HD, G.RB, 0) == 0          # no per-core window, no kv-head cores
    assert DR.probe_env() == {"ATTN_KVCORES": "1"}


def test_attn_fast_is_a_probe_variable(monkeypatch):
    """It changes the compiled kernel, so the build key has to see it (OPEN-BUILD-CACHE)."""
    assert "ATTN_FAST" in DR.PROBE_VARS
    monkeypatch.setenv("ATTN_FAST", "1")
    assert DR.probe_env() == {"ATTN_FAST": "1"}


@pytest.mark.parametrize("nh,cores", [(1, 1), (2, 2), (4, 4), (5, 5), (8, 4), (12, 6), (7, 1), (16, 4)])
def test_attn_cores_divides_and_fits(nh, cores):
    """It divides the HEAD COUNT now, not the og element count -- a core may own fewer
    heads than an og element holds (attn.h kOGH)."""
    assert DR.attn_cores(nh) == cores
    assert nh % DR.attn_cores(nh) == 0 and DR.attn_cores(nh) <= DR.MAX_ATTN_CORES


# ---- the MoE / Qwen3.5 recipe compiles the same attn.h through designs/layer_x/ax.py
from recipes import qwen36moe as Q36  # noqa: E402
from recipes import qwen35 as Q35  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# spec -> (ACORES, NHL, RB, BLOCK): 16 / 2 / 256 (the 35B and the 27B) is 8 og elements ->
# 4 cores of 4 heads; Qwen3.5-9B's 16 / 4 / 256 is 4 og elements -> 4 cores of 4; the 0.8B's
# 8 / 2 / 256 is 4 og elements -> 4 cores of 2, padded to 8 lanes.
#
# Head dim 256 WITH the output gate had no row block at all until 2026-09-22: the gate's
# exponential and reciprocal live on the attention core and the block kernel would not fit
# beside them AND the single-row kernel. The 35B now runs RB 2 with the single-row kernel
# RETIRED (BLOCK = attn.h ATTN_BLOCK_ONLY) -- every row, including the new position's, goes
# through attn_stepb and the block masks the padding rows. Qwen3.5 is the same head dim and
# gate but has not been compared on it, so it keeps RB 1: this list is measurement, not
# declaration, exactly like FAST_ATTENTION.
MOE_FAST = {"qwen36-35b-a3b.json": (4, 4, 4, 1), "qwen35-9b.json": (4, 4, 1, 0)}


@pytest.mark.parametrize("name", sorted(MOE_FAST))
def test_moe_attn_geometry_on_the_fast_path(name, unvalidated, monkeypatch):
    monkeypatch.setenv("ATTN_FAST", "1")
    A = Q36.attn(load_spec(SPECS / name))
    acores, nhl, rb, block = MOE_FAST[name]
    assert (A.VEXP, A.ACORES, A.NHL, A.RB, A.BLOCK) == (1, acores, nhl, rb, block), name
    assert A.NHL * A.ACORES == A.NH and A.NHL % A.HPO == 0 and (A.RB == 1 or A.RB * max(A.NHL, 8) in (8, 16, 32, 64))
    assert A.MLS % 32 == 0 and A.MLS >= A.NHL
    # The retirement is only ever the block kernel's own cost model: there is no such
    # thing as a block-only build with nothing to block.
    assert not (A.BLOCK and A.RB == 1), name


def test_block_only_is_measured_not_declared(unvalidated, monkeypatch):
    """Retiring the single-row kernel changes the softmax's rounding (one block rescale
    instead of one per row), so a family joins by a greedy-continuation compare. The 35B
    passed on 2026-09-22; Qwen3.5 shares the head dim and the gate and has not been run."""
    from recipes import attnknobs
    monkeypatch.delenv("ATTN_FAST", raising=False)
    assert attnknobs.BLOCK_ONLY_MEASURED == ("qwen36moe",)
    monkeypatch.setattr(attnknobs, "BLOCK_ONLY_MEASURED", ())
    A = Q36.attn(load_spec(SPECS / "qwen36-35b-a3b.json"))
    assert (A.RB, A.BLOCK) == (1, 0)


def test_block_only_needs_the_manifest_to_carry_rb(unvalidated, monkeypatch):
    """The driver pads the streamed row count to whole blocks and the kernel derives its
    block count from the same position; a kernel built one way against a driver patching
    the other way deadlocks on the fifo rather than answering wrongly, so `rb` travels in
    the manifest beside the kernel and is absent when nothing blocks."""
    monkeypatch.delenv("ATTN_FAST", raising=False)
    p = Q36.programs(load_spec(SPECS / "qwen36-35b-a3b.json"))
    assert p["kernels"]["ax0"]["rb"] == 4
    assert "rb" not in p["kernels"]["ax1"] and "rb" not in p["kernels"]["lx0"]


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


def test_block8_is_granite_only_and_forced_back_by_the_probe(unvalidated, monkeypatch):
    # Traces: OPEN-ATTN-BLOCK8
    monkeypatch.delenv("ATTN_RB", raising=False)
    assert DR.geometry(load_spec(SPECS / "granite42-3b.json")).RB == 8
    assert DR.geometry(load_spec(SPECS / "qwen3-4b.json")).RB == 4
    monkeypatch.setenv("ATTN_RB", "4")
    assert DR.geometry(load_spec(SPECS / "granite42-3b.json")).RB == 4
    from recipes import attnknobs
    flags = attnknobs.tree8_flags(4)
    assert "-DATTN_EXPA=2" in flags
    monkeypatch.setenv("ATTN_EXPA", "0")
    assert "-DATTN_EXPA=2" not in attnknobs.tree8_flags(4)
