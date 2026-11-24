"""Tests for family-doctor's machine-verifiable invariants.

Each test pins one of the skill assertions that used to live only in prose
(open-family-addition / open-kernel-families), so a regression is caught by a
command rather than by an agent remembering to read a SKILL.md.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "utilities" / "family-doctor.py"


def run():
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                          capture_output=True, text=True)


def test_current_tree_is_clean():
    p = run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "0 failure(s)" in p.stdout
    assert "0 warning(s)" in p.stdout


def test_bad_patch_key_and_fields_are_caught(tmp_path, monkeypatch):
    """A hand-edited patch with an unnormalized key/unknown field must be red."""
    # family-doctor reads PATCHES_PATH from the module; import it and point it
    # at a temp file so the shipped data is untouched.
    sys.path.insert(0, str(ROOT / "utilities"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("family_doctor", SCRIPT)
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    bad = tmp_path / "patches.json"
    bad.write_text(json.dumps({
        "Llama-3.2-1B-Instruct-GGUF": {
            "rope_scaling": "not-an-object", "bogus_field": 1,
        }}))
    monkeypatch.setattr(fd, "PATCHES_PATH", bad)
    r = fd.Report()
    fd.check_patches(r)
    msgs = " ".join(r.failures)
    assert "not normalized" in msgs
    assert "unknown field" in msgs
    assert "rope_scaling must be an object" in msgs


def test_dense_reexports_are_checked():
    """The dx.py AttributeError trap is a real check, not a no-op."""
    sys.path.insert(0, str(ROOT / "utilities"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("family_doctor", SCRIPT)
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    r = fd.Report()
    fd.check_reexports(r)
    assert r.failures == [], r.failures
    assert r.checks >= len(fd.DENSE_FAMILIES)
