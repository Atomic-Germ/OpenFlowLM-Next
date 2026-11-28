# Traces: OPEN-FAMILY-ADDITION
"""Registering new names for an existing recipe is a mechanical, reversible edit."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "utilities" / "register-family-alias.py"

_spec = importlib.util.spec_from_file_location("register_family_alias_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
rfa = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rfa
_spec.loader.exec_module(rfa)


def test_a_dry_run_writes_nothing():
    result = rfa.register("qwen35", model_types=["zzdry_v1"], arches=["zzdryarch"],
                          prefixes=["zzdry"], dry=True)
    assert result["family"] == "qwen35"
    assert result["builder"] == "_qwen35_hf"
    assert result["model_types"] == ["zzdry_v1"]
    assert "zzdry_v1" not in rfa._read(rfa.ADD)
    assert "zzdry_v1" not in rfa._read(rfa.SPEC)


def test_register_resolves_then_remove_is_clean(tmp_path, monkeypatch):
    """The real files are edited, verified, and restored exactly."""
    add_before = rfa._read(rfa.ADD)
    spec_before = rfa._read(rfa.SPEC)
    try:
        result = rfa.register("qwen35", model_types=["zzcycle_v1"], arches=["zzcyclearch"],
                              prefixes=["zzcycle"])
        assert result["builder"]
        assert '"zzcycle_v1": "qwen35"' in rfa._read(rfa.ADD)
        assert '"zzcyclearch": "_qwen35_gguf"' in rfa._read(rfa.SPEC)
        rfa.register("qwen35", model_types=["zzcycle_v1"], arches=["zzcyclearch"],
                     prefixes=["zzcycle"], remove=True)
        assert rfa._read(rfa.ADD) == add_before
        assert rfa._read(rfa.SPEC) == spec_before
    finally:
        rfa.ADD.write_text(add_before, encoding="utf-8")
        rfa.SPEC.write_text(spec_before, encoding="utf-8")


def test_it_refuses_a_family_without_a_recipe():
    import pytest
    with pytest.raises(SystemExit, match="not a known family"):
        rfa.register("no_such_family", arches=["x"])
