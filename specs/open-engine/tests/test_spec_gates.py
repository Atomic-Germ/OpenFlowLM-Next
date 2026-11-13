# Traces: OPEN-SPEC-DERIVE (canonical spec: specs/open-engine/spec.md)
"""The catalogue's enumeration mode, which the spec workflow is built on.

`utilities/op-gate.py` and `utilities/build-new-spec.py` need to know EVERY
point a new size is missing before they can run the fixtures that validate
them. The recipe validates its ops in a long sequence and raises at the first
miss, so without `catalogue.collecting()` a caller learns one point at a time --
which is how "add a size" turned into a hand-driven loop.

These tests pin the three properties that make the automation safe:

  * `collecting()` reports every missing point in ONE pass, and the points it
    reports are exactly the ones the recipe would have raised on;
  * it never changes what a normal build sees (outside the block, a miss still
    raises);
  * every shipped spec passes it with nothing missing -- so a new spec that is
    added without its gates fails here rather than at build time.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"


def _spec(name: str):
    from recipes.load import load_spec
    return load_spec(SPECS / name)


def test_every_shipped_spec_is_fully_validated():
    """No shipped spec is missing a catalogue point.

    A spec whose geometry was never run would be here -- and its build would
    fail for anyone who tries. The workflow gates the points before the spec is
    committed, so this is the check that it actually happened.
    """
    from recipes import catalogue as C
    from recipes.families import for_spec

    bad = {}
    for p in sorted(SPECS.glob("*.json")):
        spec = _spec(p.name)
        with C.collecting() as missing:
            for_spec(spec).recipe(spec)
        if missing:
            bad[p.name] = missing
    assert not bad, f"shipped specs with unvalidated catalogue points: {bad}"


def test_collecting_reports_every_point_in_one_pass():
    """A spec that misses several points reports all of them, not just the first.

    Qwen3-8B at an FF width nothing has run: the down and up projections are
    both new K, so both call sites must appear.
    """
    from recipes import catalogue as C
    from recipes.families import for_spec

    spec = dataclasses.replace(_spec("qwen3-8b.json"), intermediate=12544)
    with C.collecting() as missing:
        for_spec(spec).recipe(spec)
    ops = {op for op, _ in missing}
    assert "gemv_q4" in ops
    ks = {kw["K"] for op, kw in missing if op == "gemv_q4"}
    assert 12544 in ks, f"the new FF width is not reported: {missing}"


def test_collecting_does_not_hide_a_miss_outside_its_block():
    """The block is opt-in: a plain recipe() still raises on the first miss."""
    from recipes.catalogue import OpRangeError
    from recipes.families import for_spec

    spec = dataclasses.replace(_spec("qwen3-8b.json"), intermediate=12544)
    with pytest.raises(OpRangeError):
        for_spec(spec).recipe(spec)


def test_every_gateable_op_has_a_gate_or_a_reason():
    """The ops a spec can miss are all classified in utilities/op-gate.py.

    `attn` is delegated to build-new-spec's own probe; the rest either have a
    runnable fixture (GATES) or a named reason (NOT_AUTOMATED). An op that is
    in neither would fail the workflow with "no automated gate" and no hint.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "utilities" / "op-gate.py"
    spec = importlib.util.spec_from_file_location("_op_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from recipes.catalogue import CATALOGUE
    classified = set(mod.GATES) | set(mod.NOT_AUTOMATED)
    assert set(CATALOGUE) <= classified, (
        "catalogue ops with no gate and no stated reason: "
        f"{sorted(set(CATALOGUE) - classified)}")
