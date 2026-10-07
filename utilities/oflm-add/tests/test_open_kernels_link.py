# Traces: OPEN-ADD-KERNEL-LINK (canonical spec: specs/open-engine/spec.md)
#
# oflm-add prefers an exact ModelSpec hash, but may also select a same-family
# candidate by geometry. Hashes describe provenance, not per-model eligibility.
# Synthetic model directories and manifests pin both routes down.
import json
import struct
import sys
from pathlib import Path

import pytest

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402

# A Qwen3-dense config small enough to be obviously synthetic; every key the
# recipes' HF deriver reads for model_type "qwen3".
CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 512,
    "vocab_size": 1024,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
}

# The container header the quant-map deriver reads: an 8-byte little-endian
# length then the safetensors JSON. 5120-byte chunks == q4_1 (spec.CHUNK_FORMAT).
Q4NX_HEADER = {
    "__metadata__": {"format": "q4nx"},
    "model.layers.0.self_attn.q_proj.weight": {"dtype": "I8", "shape": [4, 5120]},
    "model.layers.0.mlp.down_proj.weight": {"dtype": "I8", "shape": [4, 5120]},
}

OTHER_HASH = "sha256:" + "de" * 32


def write_model(root, name="Synth-4B-NPU2"):
    d = root / "models" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    blob = json.dumps(Q4NX_HEADER).encode()
    with open(d / "model.q4nx", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    return d


def write_kernel_set(xclbins, model_name, spec_hash, spec=None):
    d = xclbins / model_name / "open_kernels"
    d.mkdir(parents=True)
    manifest = {"manifest_version": 1, "family": "qwen3", "spec_hash": spec_hash}
    if spec is not None:
        manifest["spec"] = spec
    (d / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return d


@pytest.fixture
def model_dir(tmp_path):
    return write_model(tmp_path)


def test_spec_hash_derives_from_the_model_directory(model_dir):
    """The spec comes off config.json + the container header, like the recipes."""
    spec_hash, note = oflm_add.model_spec_hash(model_dir)
    assert spec_hash, note
    assert spec_hash.startswith("sha256:")
    # Stable: the same directory derives the same hash.
    assert oflm_add.model_spec_hash(model_dir)[0] == spec_hash


def test_matching_spec_hash_wins_over_a_same_family_set(tmp_path, model_dir):
    spec_hash, _ = oflm_add.model_spec_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "AAA-Wrong-Shape-NPU2", OTHER_HASH)
    right = write_kernel_set(xclbins, "ZZZ-Same-Shape-NPU2", spec_hash)

    found, source = oflm_add.find_open_kernels(spec_hash, [xclbins], model_dir.name)
    assert found == right
    assert source == "ZZZ-Same-Shape-NPU2"


def test_the_models_own_directory_wins_when_several_match(tmp_path, model_dir):
    spec_hash, _ = oflm_add.model_spec_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "AAA-Same-Shape-NPU2", spec_hash)
    mine = write_kernel_set(xclbins, model_dir.name, spec_hash)

    found, source = oflm_add.find_open_kernels(spec_hash, [xclbins], model_dir.name)
    assert found == mine
    assert source == model_dir.name


def test_no_match_selects_nothing(tmp_path, model_dir):
    spec_hash, _ = oflm_add.model_spec_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "Wrong-NPU2", OTHER_HASH)

    assert oflm_add.find_open_kernels(spec_hash, [xclbins], model_dir.name) == (None, None)


def test_hash_mismatch_does_not_hide_a_same_family_candidate(tmp_path):
    xclbins = tmp_path / "xclbins"
    candidate_spec = {
        "family": "qwen3", "hidden": 256, "num_layers": 2,
        "layer_types": ["dense", "dense"], "num_heads": 4,
        "num_kv_heads": 2, "head_dim": 64, "rotary_dim": 64,
        "rope_theta": 1_000_000.0, "intermediate": 512,
        "quant": "q4_1", "extra": {"model": "an-unlisted-finetune"},
    }
    donor_spec = {**candidate_spec, "extra": {"model": "known-qwen"}}
    donor = write_kernel_set(xclbins, "Qwen3-4B-NPU2", OTHER_HASH, donor_spec)

    found, source = oflm_add.find_open_kernels(
        "sha256:" + "ab" * 32, [xclbins], "Novel-Finetune-NPU2",
        family="qwen3", model_spec=candidate_spec,
    )

    assert found == donor
    assert source.startswith("Qwen3-4B-NPU2 (same family qwen3")


def test_setup_links_same_family_candidate_when_hash_differs(tmp_path, model_dir, capsys):
    spec, note = oflm_add.model_spec_for_dir(model_dir)
    assert spec is not None, note
    xclbins = tmp_path / "xclbins"
    donor = write_kernel_set(xclbins, "Qwen3-family-bundle", OTHER_HASH, spec.to_dict())

    linked = oflm_add.setup_open_kernels(model_dir, model_dir.name, [xclbins])
    if not linked:
        pytest.skip("this account cannot create a directory link")

    assert (model_dir / "open_kernels").resolve() == donor.resolve()
    assert "hash differs" in capsys.readouterr().err


def test_exact_hash_still_precedes_same_family_similarity(tmp_path, model_dir):
    spec_hash, _ = oflm_add.model_spec_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    exact = write_kernel_set(xclbins, "Exact", spec_hash)
    same_family = write_kernel_set(
        xclbins, "Qwen3-family", OTHER_HASH,
        {"family": "qwen3", "hidden": 256, "extra": {"model": "other"}},
    )

    found, source = oflm_add.find_open_kernels(
        spec_hash, [xclbins], model_dir.name,
        family="qwen3", model_spec={"family": "qwen3", "hidden": 256},
    )

    assert found == exact
    assert source == "Exact"


def test_setup_links_the_match_where_find_kernels_looks(tmp_path, model_dir, capsys):
    spec_hash, _ = oflm_add.model_spec_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "Wrong-NPU2", OTHER_HASH)
    right = write_kernel_set(xclbins, "Right-NPU2", spec_hash)

    linked = oflm_add.setup_open_kernels(model_dir, model_dir.name, [xclbins])
    if not linked:  # Windows without developer mode: no symlink and no junction
        pytest.skip("this account cannot create a directory link")
    # Engine::find_kernels checks <model dir>/open_kernels before the xclbins root.
    assert (model_dir / "open_kernels").resolve() == right.resolve()
    assert (model_dir / "open_kernels" / "manifest.json").is_file()
    assert "Right-NPU2" in capsys.readouterr().err


def test_no_match_prints_the_export_command(tmp_path, model_dir, capsys):
    xclbins = tmp_path / "xclbins"
    xclbins.mkdir()
    assert oflm_add.setup_open_kernels(model_dir, model_dir.name, [xclbins]) is False
    err = capsys.readouterr().err
    assert "export_qwen36_kernels.py" in err
    assert f'--model-dir "{model_dir}"' in err
    assert not (model_dir / "open_kernels").exists()


def test_override_takes_the_directory_as_given(tmp_path, model_dir):
    xclbins = tmp_path / "xclbins"
    other = write_kernel_set(xclbins, "Unrelated-NPU2", OTHER_HASH)

    linked = oflm_add.setup_open_kernels(
        model_dir, model_dir.name, [xclbins], override=other
    )
    if not linked:
        pytest.skip("this account cannot create a directory link")
    assert (model_dir / "open_kernels").resolve() == other.resolve()


def test_override_without_a_manifest_is_refused(tmp_path, model_dir):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        oflm_add.setup_open_kernels(model_dir, model_dir.name, [], override=empty)
