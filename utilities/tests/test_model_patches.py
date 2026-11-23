"""Tests for the base-model patch file (rope_scaling / real_vocab).

The patch file is hand-curated but keyed by generic base name, so a finetune or
quant inherits it. These tests pin the contract that keeps the file useful:
normalization of the key, that existing values win over a patch, and that the
patch reaches both the spec dict (gen-kernel-spec) and a ModelSpec (oflm-add).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OK = ROOT / "open_kernels"
sys.path.insert(0, str(OK))
sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))

from recipes.model_patches import (  # noqa: E402
    PATCH_FIELDS,
    apply_to_dict,
    base_patch,
    load_patches,
    patch_key,
    patch_fields,
)

SHIPPED = OK / "recipes" / "patches.json"


def test_patch_key_normalizes_quant_and_slug_suffixes():
    assert patch_key("Llama-3.2-1B-Instruct-GGUF") == "Llama-3.2-1B-Instruct-NPU2"
    assert patch_key("meta-llama/Llama-3.2-1B-Instruct") == "Llama-3.2-1B-Instruct-NPU2"
    assert patch_key("Llama-3.2-1B-Instruct-GGUF") == "Llama-3.2-1B-Instruct-NPU2"
    assert patch_key("") == ""


def test_shipped_patches_load_and_are_keyed_by_generic_name():
    patches = load_patches(SHIPPED)
    assert "Llama-3.2-1B-Instruct-NPU2" in patches
    entry = patches["Llama-3.2-1B-Instruct-NPU2"]
    assert entry["real_vocab"] == 128256
    assert entry["rope_scaling"]["factor"] == 32.0
    # provenance never becomes a spec field
    assert "_source" not in PATCH_FIELDS


def test_base_patch_handles_finetune_slug(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"Llama-3.2-1B-Instruct-NPU2": {"real_vocab": 128256}}))
    assert base_patch("some-org/Llama-3.2-1B-Instruct-GGUF", p)["real_vocab"] == 128256
    assert base_patch("Qwen3-8B-NPU2", p) == {}


def test_patch_fields_only_returns_known_fields(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"X-NPU2": {"real_vocab": 7, "rope_scaling": {"factor": 1},
                                        "_source": "foo"}}))
    fields = patch_fields("X-NPU2", p)
    assert fields == {"rope_scaling": {"factor": 1}, "real_vocab": 7}


def test_apply_to_dict_fills_only_missing_fields(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"X-NPU2": {"real_vocab": 7, "rope_scaling": {"factor": 1}}}))
    spec = {"real_vocab": None, "rope_scaling": None}
    apply_to_dict(spec, "X-NPU2", p)
    assert spec == {"real_vocab": 7, "rope_scaling": {"factor": 1}}

    # a value the model itself states wins over the patch
    stated = {"real_vocab": 5, "rope_scaling": {"factor": 9}}
    apply_to_dict(stated, "X-NPU2", p)
    assert stated == {"real_vocab": 5, "rope_scaling": {"factor": 9}}


def test_patch_reaches_models_spec_via_oflm_add(tmp_path):
    import oflm_add
    from recipes.spec import ModelSpec

    spec = ModelSpec.from_dict(json.loads((OK / "recipes" / "specs" / "llama32-1b.json").read_text()))
    # blank the two patchable fields as a GGUF header would
    blanked = dataclasses_replace(spec, rope_scaling=None, real_vocab=None)
    patched = oflm_add.apply_base_patch_to_spec(blanked, "Llama-3.2-1B-Instruct", quiet=True)
    assert patched.real_vocab == 128256
    assert patched.rope_scaling["factor"] == 32.0
    # kernel_hash is restored to the shipped spec's
    assert patched.kernel_hash() == spec.kernel_hash()


def dataclasses_replace(spec, **kw):
    import dataclasses
    return dataclasses.replace(spec, **kw)
