# Traces: OPEN-MANIFEST (canonical spec: specs/open-engine/spec.md)
"""The checked-in manifest fixture (what src/open_qwen36/manifest_test.cpp
parses) is the recipe's current output; regenerate it with
`python specs/open-engine/tests/make_fixtures.py` when the recipe changes."""
from __future__ import annotations

import json
from pathlib import Path

from make_fixtures import (FIXTURE, FIXTURE_G3, FIXTURE_HY, FIXTURE_Q3, fixture_manifest,
                           fixture_manifest_g3, fixture_manifest_hy, fixture_manifest_q3)


def test_fixture_is_current():
    assert FIXTURE.is_file(), "run make_fixtures.py"
    on_disk = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert on_disk == fixture_manifest(), "fixtures/manifest_qwen36.json is stale: run make_fixtures.py"


def test_dense_fixture_is_current():
    assert FIXTURE_Q3.is_file(), "run make_fixtures.py"
    assert json.loads(FIXTURE_Q3.read_text(encoding="utf-8")) == fixture_manifest_q3(), "fixtures/manifest_qwen3_4b.json is stale"


def test_gemma_fixture_is_current(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")      # until OPEN-FAMILY-GEMMA3's points are in the catalogue
    assert FIXTURE_G3.is_file(), "run make_fixtures.py"
    assert json.loads(FIXTURE_G3.read_text(encoding="utf-8")) == fixture_manifest_g3(), "fixtures/manifest_gemma3_4b.json is stale"


def test_hunyuan_fixture_is_current():
    assert FIXTURE_HY.is_file(), "run make_fixtures.py"
    assert json.loads(FIXTURE_HY.read_text(encoding="utf-8")) == fixture_manifest_hy(), "fixtures/manifest_hy_mt2_7b.json is stale"
