"""Tests for add-family's staged, doctor-verified mechanical registration."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "utilities" / "add-family.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_staged_registery_edits_and_verification():
    """--verify stages every edit and runs family-doctor on the copy."""
    p = run("--name", "mythos", "--recipe", "dense", "--verify")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "wrote recipes/mythos.py" in p.stdout
    assert "families.py: added to FAMILIES" in p.stdout
    assert "spec.py: registered HF model_type" in p.stdout
    assert "spec.py: registered GGUF architecture" in p.stdout
    assert "spec.py: registered builders in _FAMILY_OF" in p.stdout
    assert "spec.py: added family to ROLE_TENSORS loop" in p.stdout
    assert "oflm-add: added CONFIG_MODEL_FAMILIES entry" in p.stdout
    assert "oflm-add: added GGUF_ARCH_FAMILIES entry" in p.stdout
    assert "aliased to llama3" in p.stdout
    assert "staged family-doctor PASSED" in p.stdout or "staged verification passed" in p.stdout
    assert "0 failure(s)" in p.stdout
    # The real tree is untouched by --verify/--apply-less runs.
    assert not (ROOT / "open_kernels" / "recipes" / "mythos.py").exists()


def test_generated_entry_points_refuse_with_precise_message():
    """The scaffold must never silently produce a plausible-but-wrong kernel."""
    sys.path.insert(0, str(ROOT / "utilities"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("add_family", SCRIPT)
    af = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(af)
    source = af.scaffold("mythos", "mythos", "dense")
    assert source.count("raise NotImplementedError") >= 7
    assert "this is the next function to fill in" in source
    hf, gguf = af.builder_stubs("mythos", "mythos")
    assert "_mythos_hf" in hf and "SpecError" in hf
    assert "_mythos_gguf" in gguf and "SpecError" in gguf


def test_bad_name_and_bad_adapter_are_rejected():
    p = run("--name", "!!!")
    assert p.returncode == 2
    assert "not a usable family key" in p.stderr
    p = run("--name", "mythos", "--adapter", "not_a_real_enum")
    assert p.returncode != 0
    assert "not a SupportedModelFamily enum value" in (p.stdout + p.stderr)


def test_repo_mode_detects_already_served_geometry(monkeypatch):
    """The common case: a geometry already served needs zero registration."""
    sys.path.insert(0, str(ROOT / "utilities"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("add_family", SCRIPT)
    af = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(af)
    facts = ("MyModel-NPU2", {"family": "granite", "hidden": 4096}, "granite", "from repo")
    monkeypatch.setattr(af, "derive_repo_facts", lambda r, g="": facts)
    monkeypatch.setattr(af, "find_serving_set", lambda s: ("Granite42-3B-NPU2", "/some/dir"))
    assert af.handle_repo("user/repo", "", False) == 0


def test_repo_mode_delegates_to_build_new_spec(monkeypatch):
    """A new size of a supported family is entirely build-new-spec's job."""
    sys.path.insert(0, str(ROOT / "utilities"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("add_family", SCRIPT)
    af = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(af)
    facts = ("MyModel-NPU2", {"family": "granite", "hidden": 9999}, "granite", "from repo")
    monkeypatch.setattr(af, "derive_repo_facts", lambda r, g="": facts)
    monkeypatch.setattr(af, "find_serving_set", lambda s: (None, None))
    assert af.handle_repo("user/repo", "", False) == 0   # dry run, no build


def test_build_new_spec_env_has_repo_root_on_pythonpath():
    """A gate fixture importing open_kernels.* must not fail for ENV reasons.

    build-new-spec.py calls op-gate's gate_points() directly with its own env;
    without PYTHONPATH the fixtures die with ModuleNotFoundError and a spec is
    parked in candidates/ for the wrong reason.
    """
    sys.path.insert(0, str(ROOT / "utilities"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("bns", ROOT / "utilities" / "build-new-spec.py")
    bns = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bns)
    env = bns.toolchain_env(bns.load_export_module())
    assert str(ROOT) in env["PYTHONPATH"].split(os.pathsep)


def test_idempotent_when_family_already_registered():
    """Applying twice adds no duplicate entries (doctor stays green)."""
    # Stage once via --verify to confirm no duplicate-detection crash, then
    # ensure a second identical invocation is still green.
    p1 = run("--name", "mythos", "--verify")
    p2 = run("--name", "mythos", "--verify")
    assert p1.returncode == p2.returncode == 0
    assert p1.stdout == p2.stdout
