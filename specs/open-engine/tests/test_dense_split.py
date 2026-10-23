"""The dense recipe's split down GEMV and split norm helper.

Two widths force the dense layer off the shape its kernels were first built for:

  * an FF whose activation table does not fit a main core's L1 (Qwen3-14B: FF 17408
    is 39 168 B of table against a 60 KB budget) -- the down GEMV reduces over FF,
    so it runs in two K pieces instead, each with its own table;
  * a hidden whose norm helper cannot hold its five input elements (HID 5120: five
    10 KB elements plus the output and the stack do not fit 64 KB) -- the helper
    streams instead.

Both are the MoE / Qwen3.5 tail's mechanisms (recipes/qwen36moe.py down_split and
norm_split), shared with it rather than copied: the qwen35 family's 27B has been
running them since 2026-09-30, and the dense recipe now runs the same ones.

What this checks is the SHAPE of the split -- the pieces, the budget, the pool's
band arithmetic. The numbers are the whole-layer check's business (the model
against the fp64 replica), and the attention geometry is the gate's
(designs/attn/probe_dense.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recipes import dense as DR
from recipes.load import load_spec
from recipes.qwen36moe import band_bytes

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"


def spec(name: str):
    return load_spec(SPECS / name)


def test_the_14b_splits_the_down_gemm_and_the_norm_helper():
    s = spec("qwen3-14b.json")
    G = DR.geometry(s)
    # the pieces cut at a whole f32 element of h, the first at or below FF / 2
    assert G.DOWN_SPLIT == (8192, 9216)
    assert sum(G.DOWN_SPLIT) == s.intermediate
    assert all(k % (DR.ELEM // 4) == 0 for k in G.DOWN_SPLIT)
    # the norm helper streams at HID 5120: six 10 KB elements plus the stack
    assert G.NORM_SPLIT is True
    # the widest table is the widest PIECE, not the FF
    assert G.KWIDE == max(G.DOWN_SPLIT)
    assert G.PER_CALL == 2


def test_the_split_repacks_nothing():
    """The pool is written the same way: inside a band, chunk c covers k-tile c / 2,
    so the pieces are strided taps over the bands the packer already writes."""
    s = spec("qwen3-14b.json")
    pieces = DR.geometry(s).DOWN_SPLIT
    assert sum(band_bytes(k) for k in pieces) == band_bytes(s.intermediate)


def test_the_shipped_families_keep_their_whole_ff():
    """A spec that fits keeps the kernel it always built: no split, no streaming
    norm, and the same widest table as before."""
    for name in ("qwen3-4b.json", "qwen3-8b.json", "qwen3-0.6b.json", "llama31-8b.json",
                 "gemma3-4b.json", "granite42-3b.json", "phi4-mini-4b.json"):
        s = spec(name)
        G = DR.geometry(s)
        assert G.DOWN_SPLIT == (), name
        assert G.NORM_SPLIT is False, name
        assert G.KWIDE == max(s.hidden, s.attn_q_width, s.intermediate), name


def test_the_pieces_are_validated_gemv_widths():
    """Each piece is a GEMV the core actually runs, so its K must be one the
    catalogue has seen."""
    from recipes.catalogue import CATALOGUE
    s = spec("qwen3-14b.json")
    for k in DR.geometry(s).DOWN_SPLIT:
        assert k in CATALOGUE["gemv_q4"].params["K"].values


def test_a_width_that_needs_three_pieces_is_refused():
    """The split is two pieces; a third is not implemented, and the recipe says so
    rather than building something it has never run."""
    from recipes.catalogue import OpRangeError
    s = spec("qwen3-14b.json")
    import dataclasses
    wider = dataclasses.replace(s, intermediate=40000, layer_types=s.layer_types)
    with pytest.raises(OpRangeError):
        DR.geometry(wider)
