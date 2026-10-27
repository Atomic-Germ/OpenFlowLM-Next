# Traces: OPEN-GGUF-DIRECT (the manifest's `gguf` section).
"""The manifest section a GGUF model loads, checked for the two things that are easy to
get WRONG when the section's chunk law changes.

A GGUF runs on the q4_1 chunk law (`q4_1_gguf`), the same one every container uses, so
"the sections agree" is true for almost everything -- except the PACK OPS, which are the
whole difference: the base section's read a `.q4nx`, the section's read a GGUF. And two
decisions were keyed on the OLD twin's name (`q4_1_f32`) because it used to be the only
GGUF law: the lm head's pack op, and Gemma3's embedding scale (a container pre-scales the
table at conversion; a GGUF's token_embd is raw).

Both failed only at load, on hardware, with "lm_head.weight is not in a q4nx container" or
a Gemma3 that answers nonsense -- so they are pinned here.
"""
from __future__ import annotations

import pytest

from recipes.load import load_spec
from recipes.manifest import manifest
from recipes.spec import ModelSpec


def _spec(stem):
    return load_spec(__import__("pathlib").Path(__file__).resolve().parents[3]
                     / "open_kernels" / "recipes" / "specs" / f"{stem}.json")


def test_the_gguf_section_packs_the_head_from_the_gguf():
    """The lm head is packed from `lm_head.weight` like any projection, so its op has to be
    the GGUF-reading one -- `std_perm` on a GGUF dies before a single token is generated."""
    m = manifest(_spec("qwen3-4b"), 4096)
    g = m["gguf"]
    assert g["pack"]["lm_head"]["ops"][0]["op"] == "std_perm_gguf"
    # the projections, for the same reason
    assert g["layer_types"]["dense"]["pack"]["pool"][0]["op"] == "std_perm_gguf"
    # and the base section stays the container's
    assert m["pack"]["lm_head"]["ops"][0]["op"] == "std_perm"


def test_a_gemma3_gguf_scales_its_embedding_table():
    """Gemma3's container pre-scales token_embd by sqrt(hidden) at conversion; a GGUF's is
    raw. Without the scale the model still runs -- and answers nonsense."""
    m = manifest(_spec("gemma3-4b"), 4096)
    g = m["gguf"]
    assert g["pack"]["embed"]["scale"] == pytest.approx(g["spec"]["hidden"] ** 0.5, rel=1e-12)
    assert "scale" not in m["pack"]["embed"]


def test_reads_gguf_names_both_gguf_quant_laws():
    """`q4_1_f32` (the dropped f32 twin) and `q4_1_gguf` (the q4_1 law with GGUF ops) both
    mean "these weights are a GGUF file"; every decision that asks that must use the
    predicate, not one of the names."""
    from dataclasses import replace
    s = _spec("qwen3-4b")
    for q, want in (("q4_1", False), ("q4_1_f32", True), ("q4_1_gguf", True), ("q8", False)):
        assert replace(s, quant=q).reads_gguf is want, q
