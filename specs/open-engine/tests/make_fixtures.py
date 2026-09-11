"""Write the manifest fixtures with a fixed build key (the real one hashes the
sources, which would churn the fixtures on every edit):

    fixtures/manifest_qwen36.json     the 27B (Qwen3.6-35B-A3B, the qwen36moe recipe)
    fixtures/manifest_qwen3_4b.json   Qwen3-4B (the qwen3 dense recipe)
    fixtures/manifest_gemma3_4b.json  Gemma3-4B (two layer types, a sliding window)
    fixtures/manifest_hy_mt2_7b.json  Hy-MT2-7B (post-RoPE q/k norm, a padded head)
    fixtures/manifest_qwen35_9b.json  Qwen3.8-Distilled-9B (the qwen35 composition)
    fixtures/manifest_phi4_mini_4b.json  Phi4-mini (a 96-dim rotation, longrope's two tables, hf_config_defaults)

    python specs/open-engine/tests/make_fixtures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "open_kernels"))
from recipes.load import default_spec, load_spec  # noqa: E402
from recipes.manifest import manifest  # noqa: E402

FIXTURE = HERE / "fixtures" / "manifest_qwen36.json"
FIXTURE_Q3 = HERE / "fixtures" / "manifest_qwen3_4b.json"
SPEC_Q3 = HERE.parents[2] / "open_kernels" / "recipes" / "specs" / "qwen3-4b.json"
FIXTURE_G3 = HERE / "fixtures" / "manifest_gemma3_4b.json"
SPEC_G3 = HERE.parents[2] / "open_kernels" / "recipes" / "specs" / "gemma3-4b.json"
FIXTURE_HY = HERE / "fixtures" / "manifest_hy_mt2_7b.json"
SPEC_HY = HERE.parents[2] / "open_kernels" / "recipes" / "specs" / "hy-mt2-7b.json"
FIXTURE_Q35 = HERE / "fixtures" / "manifest_qwen35_9b.json"
SPEC_Q35 = HERE.parents[2] / "open_kernels" / "recipes" / "specs" / "qwen35-9b.json"
FIXTURE_PH = HERE / "fixtures" / "manifest_phi4_mini_4b.json"
SPEC_PH = HERE.parents[2] / "open_kernels" / "recipes" / "specs" / "phi4-mini-4b.json"


def fixture_manifest() -> dict:
    return manifest(default_spec(), key="sha256:fixture")


def fixture_manifest_q3() -> dict:
    return manifest(load_spec(SPEC_Q3), key="sha256:fixture")


def fixture_manifest_g3() -> dict:
    return manifest(load_spec(SPEC_G3), key="sha256:fixture")


def fixture_manifest_hy() -> dict:
    return manifest(load_spec(SPEC_HY), key="sha256:fixture")


def fixture_manifest_q35() -> dict:
    """OPEN_KERNELS_UNVALIDATED: the qwen35 points (K 12288 GEMVs, lm_head_q8 at K 4096)
    enter the catalogue only after OPEN-FAMILY-QWEN35's hardware pass."""
    os.environ["OPEN_KERNELS_UNVALIDATED"] = "1"
    return manifest(load_spec(SPEC_Q35), key="sha256:fixture")


def fixture_manifest_ph() -> dict:
    return manifest(load_spec(SPEC_PH), key="sha256:fixture")


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    for f, m in ((FIXTURE, fixture_manifest()), (FIXTURE_Q3, fixture_manifest_q3()),
                 (FIXTURE_G3, fixture_manifest_g3()), (FIXTURE_HY, fixture_manifest_hy()),
                 (FIXTURE_Q35, fixture_manifest_q35()), (FIXTURE_PH, fixture_manifest_ph())):
        f.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {f}")
