"""CPU execution of the real production fold, not the synthetic probe.

OFLM_OPEN_FOLD_DEDUP_INPUT_SPLIT=1 opts the shared Core (native CLI and
engine) into reusing high.low for low-weight products. Unset, empty, zero
and all other values retain the control. It only affects four-product mode.
The input must remain valid and unchanged through dispatch; mapped outputs
are borrowed and may be overwritten on every dispatch. No hardware or
performance acceptance is established by these CPU tests.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    path = tmp_path_factory.mktemp("fold-dedup")
    exe = path / "fold-dedup"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT / "src"),
                    str(Path(__file__).with_suffix(".cpp")), "-o", str(exe)], check=True)
    return exe


@pytest.mark.parametrize("case", ["numeric", "validation", "timing"])
def test_production_candidate_against_control(native, case):
    subprocess.run([str(native), case], check=True)
