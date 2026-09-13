# Traces: OPEN-BUILD-CACHE (canonical spec: specs/open-engine/spec.md)
"""A generated kernel TU must not also be tracked in git.

.gitignore says why: these files are written per-spec during the export, so "a tracked
copy drifts to match whichever spec was exported last". Ignoring a path does nothing once
it is tracked, and `designs/lm_head_q4/gemv_q4_gy.cc` was tracked for two releases -- every
export rewrote it, every `git add -A` swept the rewrite into the commit, and twice a
session was spent restoring it by hand.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def tracked(pattern: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", pattern], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


@pytest.mark.parametrize("pattern", [
    "open_kernels/designs/dense/*.cc",
    "open_kernels/designs/layer_x/*.cc",
    "open_kernels/designs/lm_head_q4/*.cc",
    "open_kernels/designs/gemv_q4/gemv_q4_prep_k*.cc",
])
def test_no_generated_tu_is_tracked(pattern):
    files = tracked(pattern)
    assert files == [], (
        f"{pattern} is ignored but still tracked: {files}. Ignoring does not apply to a "
        f"tracked file, so every export rewrites it. `git rm --cached <path>`.")
