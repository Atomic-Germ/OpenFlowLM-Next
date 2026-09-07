# Traces: OPEN-BUILD-CACHE (canonical spec: specs/open-engine/spec.md)
"""The build key covers the recipe sources, the kernel sources the recipe's
designs include, the spec and the quant format -- and nothing informational."""
from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

from recipes.cache import ROOT, build_key, source_files
from recipes.load import default_spec


def copy_tree(dst: Path) -> Path:
    for f in source_files(default_spec()):
        rel = f.relative_to(ROOT)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst / rel)
    return dst


def test_key_is_stable_and_covers_the_sources(tmp_path):
    spec = default_spec()
    root = copy_tree(tmp_path / "ok")
    k1 = build_key(spec, root)
    assert k1 == build_key(spec, root) == build_key(spec)
    assert k1.startswith("sha256:") and len(k1) == 7 + 64
    files = [f.relative_to(root).as_posix() for f in source_files(spec, root)]
    for must in ("recipes/qwen36moe.py", "designs/layer_x/lx.py", "designs/layer_x/xcommon.py",
                 "designs/attn/attn.h", "designs/gemv_q4/gemv_q4.h", "designs/layer_x/gen_kernels.py",
                 "designs/lm_head_q8/lm_head_q8.py", "include/vecmath.h"):
        assert must in files, must


def test_a_kernel_source_edit_changes_the_key(tmp_path):
    spec = default_spec()
    root = copy_tree(tmp_path / "edit")
    before = build_key(spec, root)
    p = root / "designs" / "attn" / "attn.h"
    p.write_text(p.read_text(encoding="utf-8") + "\n// touched\n", encoding="utf-8")
    assert build_key(spec, root) != before


def test_a_recipe_edit_changes_the_key(tmp_path):
    spec = default_spec()
    root = copy_tree(tmp_path / "recipe")
    before = build_key(spec, root)
    p = root / "recipes" / "qwen36moe.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")
    assert build_key(spec, root) != before


def test_spec_and_quant_change_the_key_but_extra_does_not():
    spec = default_spec()
    k = build_key(spec)
    assert build_key(dataclasses.replace(spec, rope_theta=1e6)) != k
    assert build_key(dataclasses.replace(spec, quant="q4_k")) != k
    assert build_key(dataclasses.replace(spec, extra={"model": "another-name"})) == k

# ---- the probe environment, which the key could not see until 2026-09-07

def _granite_spec():
    """A DENSE spec: probe_env() lives on the dense recipe, so the MoE default
    spec would exercise the getattr fallback instead of the thing under test."""
    from recipes.load import load_spec
    return load_spec(ROOT / "recipes" / "specs" / "granite42-3b.json")


def test_a_probe_build_does_not_share_a_key_with_a_real_one(monkeypatch):
    """ATTN_NULL / ATTN_ABL / ATTN_RB change the compiled kernel, and build_key
    hashes sources + spec + quant -- which cannot see an environment variable.
    Without them in the key, exporting a probe and then a real build reuses the
    probe's artifacts, because export_qwen36_kernels.py skips a build whose key
    the destination already carries. That ships a kernel set computing nothing,
    silently."""
    for var in ("ATTN_NULL", "ATTN_ABL", "ATTN_RB"):
        monkeypatch.delenv(var, raising=False)
    spec = _granite_spec()
    clean = build_key(spec)

    keys = {clean}
    for var, value in (("ATTN_NULL", "1"), ("ATTN_ABL", "1"), ("ATTN_RB", "2")):
        monkeypatch.setenv(var, value)
        keys.add(build_key(spec))
        monkeypatch.delenv(var)
    assert len(keys) == 4, "each probe must give the key a value of its own"

    # ...and with nothing set the key must be exactly where it was, so adding
    # this did not invalidate every already-built set in the tree.
    assert build_key(spec) == clean


def test_attn_rb_is_refused_rather_than_left_to_the_compiler(monkeypatch):
    """An unsupported ATTN_RB used to reach attn_stepb.cc's `#error` after a full
    design build, or to produce a score block the block exponential has no width
    for. Both are minutes away from the mistake."""
    import pytest

    from recipes.catalogue import OpRangeError
    import recipes.dense as DR

    spec = _granite_spec()
    for bad, expect in (("banan", "not an integer"), ("3", "not supported"), ("0", "not supported")):
        monkeypatch.setenv("ATTN_RB", bad)
        with pytest.raises(OpRangeError, match=expect):
            DR.geometry(spec)
    monkeypatch.setenv("ATTN_RB", "4")
    assert DR.geometry(spec).RB == 4
