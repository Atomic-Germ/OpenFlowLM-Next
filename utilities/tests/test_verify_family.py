"""Tests for verify-family's NPU-free reporting and bar-status logic."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "utilities" / "verify-family.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_bar_needs_measurement_without_evidence():
    # A missing GGUF makes the helper FAIL, but the bar must still be honest.
    p = run("--gguf", "/does/not/exist.gguf", "--json")
    assert p.returncode == 1
    report = json.loads(p.stdout)
    assert report["ok"] is False
    assert len(report["production_bar"]) == 10
    assert all(b["status"] == "needs-measurement" for b in report["production_bar"])


def test_evidence_marks_only_claimed_items_pass(tmp_path):
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"quantized": {"status": "pass", "note": "Q4_K_M Unsloth"}}))
    p = run("--gguf", "/does/not/exist.gguf", "--evidence", str(ev), "--json")
    report = json.loads(p.stdout)
    statuses = {b["id"]: b["status"] for b in report["production_bar"]}
    assert statuses["quantized"] == "pass"
    assert statuses["benchmarks"] == "needs-measurement"
    assert sum(v == "pass" for v in statuses.values()) == 1


def test_requires_a_target():
    p = run()
    assert p.returncode != 0
    assert "nothing to check" in (p.stdout + p.stderr)


def test_install_check_flags_missing_files(tmp_path):
    empty = tmp_path / "Empty-NPU2"
    empty.mkdir()
    p = run("--model-dir", str(empty), "--json")
    report = json.loads(p.stdout)
    install = next(c for c in report["checks"] if c["name"] == "install")
    assert install["status"] == "FAIL"
    assert "missing config.json" in install["detail"]
