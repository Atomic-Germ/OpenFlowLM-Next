# Traces: OPEN-ADD-KERNEL-LINK (canonical spec: specs/open-engine/spec.md)
#
# oflm-add picks the open kernel set by ModelSpec, not by model name: the set
# whose manifest.json carries the same spec_hash the model derives. Two fake
# kernel dirs (one matching, one not) and a synthetic model directory
# (config.json + a model.q4nx safetensors header) pin that choice down.
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


def write_kernel_set(xclbins, model_name, spec_hash):
    d = xclbins / model_name / "open_kernels"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"manifest_version": 1, "family": "qwen3", "spec_hash": spec_hash}),
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


def test_issue_routing_distinguishes_our_bug_from_the_maintainers():
    """A GGUF-direct refusal has no single right tracker, and guessing sends
    every report to the wrong project.

    The discriminator is one the user can check in a minute: run the same GGUF
    on the CPU reference. Wrong there too and the file or its conversion is at
    fault, which is the model's own repo; right there and the kernels are wrong,
    which is ours. A shape nobody ever built is neither -- it is a build
    request, and telling someone to file a bug for it wastes their time.

    Quant and mmproj rejections get no routing at all: they are documented
    instructions, not failures, and a "where do I report this" line under them
    would invite noise.
    """
    from oflm_add import gguf_issue_routing

    never_built = gguf_issue_routing(
        "no installed open kernel set is built for this model's shape")
    assert "build request" in never_built
    assert "issues" not in never_built, "a build request is not a bug report"

    missing_twin = gguf_issue_routing(
        "Qwen3.5-9B-NPU2 matches this model's shape but has no GGUF-direct "
        "build (its open_kernels manifest has no 'gguf' section)")
    assert "already installed" in missing_twin or "installed" in missing_twin
    assert "Atomic-Germ" in missing_twin, "building a twin is our work"
    assert "HuggingFace" not in missing_twin

    wrong_output = gguf_issue_routing("some unforeseen reason")
    assert "--device cpu" in wrong_output, "must give the discriminator step"
    assert "HuggingFace" in wrong_output and "Atomic-Germ" in wrong_output

    assert gguf_issue_routing(
        "quant Q5_K_M not supported by the open kernels (have Q4_1/...)") is None
    assert gguf_issue_routing(
        "vision projector (mmproj), not the language model") is None
