# Traces: OPEN-KERNEL-PIPELINE
"""An op with no per-geometry fixture is DEFERRED, not a build wall."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "utilities" / "op-gate.py"

_spec = importlib.util.spec_from_file_location("op_gate_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
og = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = og
_spec.loader.exec_module(og)


def test_blocked_points_are_returned_not_raised(monkeypatch):
    """The pasted Coder-Next run died here; it must instead continue the build."""
    monkeypatch.setattr(og, "missing_points", lambda spec: [
        ("router", {"experts": 512, "topk": 10}),
        ("moe", {"ff": 512, "experts": 512}),
    ])
    spec = SimpleNamespace()
    gated, blocked = og.gate_points(spec, Path("x.json"), "py", {}, None)
    assert gated == []
    assert [op for op, _, _ in blocked] == ["router", "moe"]
    assert all("whole-layer" in why for _, _, why in blocked)


def test_a_fixture_that_fails_is_still_an_error(monkeypatch):
    """A real shortfall is not the same as a missing fixture."""
    monkeypatch.setattr(og, "missing_points", lambda spec: [("ln", {"width": 999})])
    monkeypatch.setattr(og, "GATE_FN", {"ln": lambda *a: (_ for _ in ()).throw(
        og.GateError("fixture failed"))})
    import pytest
    with pytest.raises(og.GateError, match="fixture failed"):
        og.gate_points(SimpleNamespace(), Path("x.json"), "py", {}, None)
