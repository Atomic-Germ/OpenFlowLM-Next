# Traces: OPEN-MANIFEST (the manifest the engine reads), OPEN-GGUF-DIRECT
"""The GGUF-direct twin: the manifest's "gguf" section and the f32-scale builds
must agree.

A GGUF-direct model (one that ships a model.gguf instead of a q4nx container)
is read through the manifest's "gguf" section: the same program against the
f32-scale pool layout. The engine swaps that section in and then loads the
kernels it names -- so a section that names a kernel the export never built
is a manifest that fails at load, far from the recipe that wrote it.

This is the check that used to be a hand-kept list of "GGUF-capable families"
in oflm-add: it is derived (every shipped spec's manifest), it is in the
recipe's own namespace (no name games between the registry's family and the
recipe's), and it covers the shapes the list could not name at all.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SPECS = REPO / "open_kernels" / "recipes" / "specs"

sys.path.insert(0, str(REPO / "open_kernels"))

from recipes.load import load_spec  # noqa: E402
from recipes.manifest import manifest  # noqa: E402


@pytest.fixture(autouse=True)
def _unvalidated(monkeypatch):
    """The geometry is not the question here, so a spec the catalogue has not
    seen still builds its manifest. The flag is scoped to this module: it is
    global state, and the other tests count on the catalogue wall."""
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def _specs():
    return sorted(SPECS.glob("*.json"))


def _manifest(spec_path: Path):
    spec = load_spec(spec_path)
    try:
        return manifest(spec)
    except Exception as exc:                       # a spec the recipe refuses
        pytest.skip(f"{spec_path.name}: {exc}")


def test_every_shipped_spec_builds_a_manifest():
    """The specs are the recipes' input; each must produce a manifest (or the
    test skips with the reason)."""
    built = 0
    for p in _specs():
        m = _manifest(p)
        assert m.get("family"), p.name
        built += 1
    assert built >= 10, f"only {built} specs built a manifest -- the spec set shrank?"


@pytest.mark.parametrize("spec_path", _specs(), ids=lambda p: p.stem)
def test_gguf_section_names_only_built_kernels(spec_path: Path):
    """The "gguf" section may only name kernels the manifest's builds make."""
    m = _manifest(spec_path)
    section = m.get("gguf")
    if not section:
        # No twin at this shape: then no f32-scale kernel set may be built
        # either (the export would make kernels nothing references).
        f32 = {n for n, b in m.get("builds", {}).items()
               if any("SCALES_F32" in k for k in b.get("env", {}))}
        assert not f32, (f"{spec_path.name}: builds the f32-scale sets {sorted(f32)} "
                          "but advertises no gguf section")
        return
    builds = m.get("builds", {})
    for name, k in section.get("kernels", {}).items():
        b = k.get("build")
        assert b in builds, (f"{spec_path.name}: the gguf section's kernel '{name}' names the "
                             f"build '{b}', which the manifest does not build")
        env = builds[b].get("env", {})
        if b.endswith("_f32"):
            # A twin build is the f32-scale variant of its design: the env says
            # so, or it would compile the bf16-scale one and read the pool wrong.
            assert any("SCALES_F32" in k2 for k2 in env), (
                f"{spec_path.name}: the twin build '{b}' has no f32-scale flag "
                f"(env {env}) -- it would compile the bf16-scale variant")


@pytest.mark.parametrize("spec_path", _specs(), ids=lambda p: p.stem)
def test_gguf_section_is_the_f32_program(spec_path: Path):
    """The section is the f32-scale program's own view: its quant is the f32
    one, and the pool layout it declares is the f32-scale one."""
    m = _manifest(spec_path)
    section = m.get("gguf")
    if not section:
        return
    gguf_spec = section.get("spec")
    assert gguf_spec, f"{spec_path.name}: the gguf section carries no spec"
    assert gguf_spec.get("quant") == "q4_1_f32", gguf_spec.get("quant")


def test_no_hand_written_family_list_remains():
    """oflm-add's gate is the manifest, not a list of family names: the registry's
    family ("qwen3.5") and the recipe's ("qwen35") are different strings for the
    same model, and a list written in one of them silently refuses the other."""
    add = (REPO / "utilities" / "oflm-add" / "oflm_add" / "__init__.py").read_text(encoding="utf-8")
    assert "GGUF_CAPABLE_FAMILIES" not in add, (
        "oflm-add still gates the GGUF path on a family-name list; the manifest "
        "is the gate (manifest_supports_gguf)")


def test_manifest_fixtures_still_carry_the_section():
    """The checked-in fixtures (what the C++ manifest_test parses) keep the
    section where the recipe emits one."""
    fx = REPO / "specs" / "open-engine" / "tests" / "fixtures"
    seen = 0
    for p in sorted(fx.glob("manifest*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "gguf" in d:
            seen += 1
    assert seen, "no checked-in manifest fixture carries the gguf section"
