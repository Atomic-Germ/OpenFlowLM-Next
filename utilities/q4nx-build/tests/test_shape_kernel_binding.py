"""The pack-time support audit must ask "does an installed set bind this SHAPE?",
not "is there something named after this model?".

q4nx-build used to check `src/xclbins/<model dir name>` and a `model_list.json`
entry keyed to the model. Both are per-name, and both report a perfectly
runnable fine-tune as unsupported -- which is the assumption that keeps pushing
people toward hard-coding. Kernel sets are per-family and matched on geometry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
OPEN_KERNELS = REPO / "open_kernels"


def _recipes_on_path():
    if str(OPEN_KERNELS) not in sys.path:
        sys.path.insert(0, str(OPEN_KERNELS))


def _load_spec(name: str):
    _recipes_on_path()
    from recipes.spec import ModelSpec
    return ModelSpec.from_json((OPEN_KERNELS / "recipes" / "specs" / f"{name}.json").read_text())


def _write_manifest(xclbins: Path, set_name: str, spec_dict: dict, *, with_key: bool) -> None:
    d = xclbins / set_name / "open_kernels"
    d.mkdir(parents=True)
    _recipes_on_path()
    from recipes.spec import ModelSpec
    s = ModelSpec.from_dict(spec_dict)
    doc = {"manifest_version": 1, "family": s.family, "spec": spec_dict,
           "spec_hash": s.spec_hash()}
    if with_key:
        doc["kernel_hash"] = s.kernel_hash()
    (d / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")


def test_a_finetune_binds_a_family_set_regardless_of_its_name(tmp_path):
    """The set is named for a DIFFERENT model entirely; the fine-tune still binds."""
    from q4nx.cli import _shape_kernel_set_installed
    base = _load_spec("qwen3-4b").to_dict()
    xclbins = tmp_path / "src" / "xclbins"
    _write_manifest(xclbins, "Some-Official-Name-NPU2", base, with_key=True)

    _recipes_on_path()
    from recipes.spec import ModelSpec
    fine_tune = ModelSpec.from_dict({**base, "real_vocab": base["real_vocab"] - 7})

    assert _shape_kernel_set_installed(tmp_path, fine_tune) is True


def test_an_unseen_model_name_does_not_need_its_own_directory(tmp_path):
    """Nothing named after this model exists, and that is correct and fine."""
    from q4nx.cli import _shape_kernel_set_installed
    base = _load_spec("qwen3-4b").to_dict()
    xclbins = tmp_path / "src" / "xclbins"
    _write_manifest(xclbins, "Qwen3-4B-NPU2", base, with_key=True)

    assert not (xclbins / "Their-Unheard-Of-FineTune-NPU2").exists()
    _recipes_on_path()
    from recipes.spec import ModelSpec
    theirs = ModelSpec.from_dict({**base, "real_vocab": 151000})
    assert _shape_kernel_set_installed(tmp_path, theirs) is True


def test_a_pre_kernel_hash_manifest_still_binds(tmp_path):
    """An already-installed set carries only spec_hash and the spec dict.

    The shape key is a pure function of that spec, so such a set must still serve
    a shape-identical fine-tune without anyone re-exporting it.
    """
    from q4nx.cli import _shape_kernel_set_installed
    base = _load_spec("qwen3-4b").to_dict()
    xclbins = tmp_path / "src" / "xclbins"
    _write_manifest(xclbins, "Qwen3-4B-NPU2", base, with_key=False)
    mf = xclbins / "Qwen3-4B-NPU2" / "open_kernels" / "manifest.json"
    assert "kernel_hash" not in json.loads(mf.read_text())

    _recipes_on_path()
    from recipes.spec import ModelSpec
    fine_tune = ModelSpec.from_dict({**base, "real_vocab": 151000})
    assert fine_tune.spec_hash() != json.loads(mf.read_text())["spec_hash"]
    assert _shape_kernel_set_installed(tmp_path, fine_tune) is True


def test_a_genuinely_different_shape_does_not_bind(tmp_path):
    from q4nx.cli import _shape_kernel_set_installed
    base = _load_spec("qwen3-4b").to_dict()
    xclbins = tmp_path / "src" / "xclbins"
    _write_manifest(xclbins, "Qwen3-4B-NPU2", base, with_key=True)

    _recipes_on_path()
    from recipes.spec import ModelSpec
    other = ModelSpec.from_dict({**base, "hidden": base["hidden"] * 2})
    assert _shape_kernel_set_installed(tmp_path, other) is False