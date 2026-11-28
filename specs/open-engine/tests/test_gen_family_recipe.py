# Traces: OPEN-FAMILY-ADDITION
"""A generated recipe module fills in derived constants and names real walls."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "utilities" / "gen-family-recipe.py"

_spec = importlib.util.spec_from_file_location("gen_family_recipe_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
gfr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gfr
_spec.loader.exec_module(gfr)


def test_render_fills_derived_constants_and_names_the_wall(tmp_path):
    d = {
        "spec": {"family": "flint", "shared_expert_intermediate": 1024,
                 "moe_intermediate": 512},
        "hardware": {"N_CORES": 8},
        "quant": {"CH": 5120, "PER_CALL": 2},
        "template": {"template": "moe-router", "closest": "qwen36moe/flint"},
        "capabilities": {"TOPK_MAX": 6, "SLOTS": 8, "EXPERTS": 256,
                         "SHARED_SLOTS": 2, "ATTN_CORES": 4},
        "inputs": {"design": "designs/flint ..."},
    }
    text = gfr.render("flint", "qwen36moe", "designs/flint", d)
    assert "TOPK = 6" in text and "SLOTS = 8" in text and "SHARED_SLOTS = 2" in text
    assert "shared_expert_intermediate" in text and "1024" in text
    assert "delegat" in text.lower() and "OpRangeError" in text
    p = tmp_path / "flint.py"
    p.write_text(text)
    compile(text, str(p), "exec")


def test_dispatch_registration_is_idempotent_and_dry_runnable(monkeypatch, tmp_path):
    """families.py gets a dispatch branch + tuple entry, scripted like specs."""
    fake = tmp_path / "families.py"
    fake.write_text(
        'from types import ModuleType\n\n\n'
        'def family_module(name):\n'
        '    if name == "qwen36moe":\n'
        '        from . import qwen36moe\n'
        '        return qwen36moe\n'
        '    raise ValueError(name)\n\n\n'
        'FAMILIES = ("qwen36moe", "qwen3")\n', encoding="utf-8")
    monkeypatch.setattr(gfr, "FAMILIES_PY", fake)
    assert gfr.register_dispatch("flint", dry=True) is True
    assert 'name == "flint"' not in fake.read_text()      # dry run writes nothing
    assert gfr.register_dispatch("flint", dry=False) is True
    text = fake.read_text()
    assert 'name == "flint"' in text and '"flint"' in text
    assert gfr.register_dispatch("flint", dry=False) is False  # idempotent


def test_a_family_cannot_delegate_to_itself(tmp_path):
    """The user's `qwen36moe -> qwen36moe` run destroyed the real recipe."""
    import pytest
    d = {"spec": {}, "hardware": {}, "quant": {}, "template": {}, "capabilities": {},
         "inputs": {}}
    with pytest.raises(SystemExit, match="cannot delegate to itself"):
        gfr.render("qwen36moe", "qwen36moe", "designs/x", d)


def test_render_with_no_difference_has_no_wall(tmp_path):
    d = {
        "spec": {"family": "x", "shared_expert_intermediate": 0, "moe_intermediate": 512},
        "hardware": {"N_CORES": 8}, "quant": {"CH": 5120, "PER_CALL": 2},
        "template": {"template": "dense", "closest": "qwen3/dense"},
        "capabilities": {}, "inputs": {"design": "designs/x"},
    }
    text = gfr.render("x", "qwen3", "designs/x", d)
    assert "DIFFERENCES = {}" in text
