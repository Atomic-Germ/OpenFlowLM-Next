# Traces: OPEN-SPEC-DERIVE (canonical spec: specs/open-engine/spec.md)
"""Every shipped spec names the model it builds kernels for, and no two claim
the same one.

`extra.model` is the export destination: `export_qwen36_kernels.py` writes
`src/xclbins/<extra.model>/open_kernels`, and `oflm add` links a converted
model's kernels by that same name. Two specs sharing a name means one silently
overwrites the other -- gemma3-12b shipped as "Gemma3-4B-NPU2" and the 4B sets
were overwritten on every full export (#54, fixed here). A name that does not
match the spec's own family and size is the same mistake waiting to happen, so
both are refused.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"

# The name a spec's file implies: <family>-<size>, e.g. gemma3-12b -> Gemma3-12B-NPU2.
# A few specs are named after something other than their size (a distilled
# fine-tune, a MoE's parameter count); they are listed with what they must say.
EXPECTED = {
    "qwen35-9b": "Qwen3.5-9B-NPU2",
    "qwen35-27b": "Qwen3.8-27B-NPU2",
    "k2-horizon-3.7b": "K2-Horizon-3.7B-NPU2",
    "qwen36-35b-a3b": "Qwen3.6-35B-A3B-NPU2",
    "hy-mt2-7b": "Hy-MT2-7B-NPU2",
    "lfm2-1.2b": "LFM2-1.2B-NPU2",
    "minicpm5-2b": "MiniCPM5-2B-NPU2",
    "phi4-mini-4b": "Phi4-mini-Instruct-NPU2",
    "llama31-8b": "Llama-3.1-8B-NPU2",
    "granite42-3b": "Granite-4.2-3B-NPU2",
    "qwen25-3b": "Qwen2.5-3B-Instruct-NPU2",
    "qwen3-4b": "Qwen3-4B-NPU2",
}


def expected_model(stem: str) -> str:
    if stem in EXPECTED:
        return EXPECTED[stem]
    family, _, size = stem.rpartition("-")
    assert family and size, f"{stem}: expected <family>-<size>"
    # gemma3-4b -> Gemma3-4B-NPU2: the family is a brand, the size is shouted.
    return f"{family}-{size.upper()}-NPU2".replace("gemma3", "Gemma3")


def specs() -> list[Path]:
    return sorted(SPECS.glob("*.json"))


def test_specs_are_present():
    # A glob that matches nothing would make every test below pass vacuously.
    assert len(specs()) >= 12


@pytest.mark.parametrize("path", specs(), ids=lambda p: p.stem)
def test_model_name_matches_the_spec(path: Path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    model = spec.get("extra", {}).get("model")
    assert model, f"{path.name}: extra.model is what the export is named after"
    want = expected_model(path.stem)
    assert model == want, f"{path.name} ({spec['family']}) exports as {model}; it should be {want}"


def test_no_two_specs_claim_one_model():
    seen: dict[str, str] = {}
    for path in specs():
        model = json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get("model")
        if model is None:
            continue
        assert model not in seen, (
            f"{path.name} and {seen[model]} both export as {model}; the second one "
            "overwrites the first in src/xclbins"
        )
        seen[model] = path.name


@pytest.mark.parametrize("path", specs(), ids=lambda p: p.stem)
def test_model_name_is_a_plain_directory_component(path: Path):
    model = json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get("model", "")
    assert model and "/" not in model and not model.startswith(".")
    # It becomes a path under src/xclbins and a store directory name.
    assert re.fullmatch(r"[A-Za-z0-9._-]+", model), f"{path.name}: {model!r} is not a safe directory name"
