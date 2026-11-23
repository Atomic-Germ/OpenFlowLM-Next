# Traces: OPEN-SPEC-DERIVE (canonical spec: specs/open-engine/spec.md)
"""Every shipped spec has a safe, unique, geometry-free export label.

`extra.model` is the export destination: ``export_qwen36_kernels.py`` writes
``src/xclbins/<extra.model>/open_kernels``. It is ONLY a directory label.
``oflm add`` links a model to kernels by geometry -- ``ModelSpec.kernel_hash()``
and ``serves()`` -- not by this name, and not by depth or parameter count: a
27B set runs a 40B merged model at a different depth, and a 9B Qwen3.5 set
differs structurally from 27B and is matched on the geometry parts that matter.
The name therefore must not encode a size either.

The invariants worth enforcing are:

  * no two specs claim the same export directory (the #54 regression: gemma3-12b
    shipped as "Gemma3-4B-NPU2" and overwrote the 4B sets on every full export);
  * the label is a plain, safe path component;
  * it is not named after a GGUF upload suffix -- quant format is a property of
    the file, not of the kernels.

There is deliberately NO frozen list of expected names and NO name-to-size
rule. Either would turn every new base/size into a manual edit and would reject
the legitimate cases above. The CLI derives a generic label from the base model
the repo quantizes; these tests check only what keeps exports collision-free.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"

# Upload suffixes that are properties of a GGUF FILE, not of the model whose
# geometry the kernels implement. A spec named after one of these is really a
# repo slug, which would fork the kernels of a second repo of the same base.
UPLOAD_SUFFIXES = ("gguf", "i1", "q4nx", "q4_k_m", "q8_0", "f16", "bf16")


def specs() -> list[Path]:
    return sorted(SPECS.glob("*.json"))


def test_specs_are_present():
    # A glob that matches nothing would make every test below pass vacuously.
    assert len(specs()) >= 12


@pytest.mark.parametrize("path", specs(), ids=lambda p: p.stem)
def test_model_name_is_a_real_label(path: Path):
    model = json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get("model")
    assert model, f"{path.name}: extra.model is what the export is named after"
    assert len(model) >= 3, f"{path.name}: {model!r} is too short to be an export label"


@pytest.mark.parametrize("path", specs(), ids=lambda p: p.stem)
def test_model_name_is_geometry_free(path: Path):
    model = json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get("model", "")
    tail = model.lower().rsplit("-", 1)[-1]
    assert tail not in UPLOAD_SUFFIXES, (
        f"{path.name}: {model!r} is named after a GGUF upload suffix, not the model; "
        "quant format is a property of the file, not the kernels"
    )


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
