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


def test_building_a_missing_size_writes_the_spec_and_runs_the_script(tmp_path, monkeypatch):
    """A geometry nothing serves is BUILT here, not handed to the user as a command.

    The spec comes from the model's own file (so a gated base repo cannot change the
    geometry), lands where build-new-spec.py looks for it, and that script does the
    build/gate/catalogue work.
    """
    from recipes.spec import ModelSpec

    checkout = tmp_path / "open_kernels"
    (checkout / "recipes" / "specs").mkdir(parents=True)
    (tmp_path / "utilities").mkdir()
    marker = tmp_path / "ran"
    (tmp_path / "utilities" / "build-new-spec.py").write_text(
        f"open({str(marker)!r}, 'w').write('ok')\n", encoding="utf-8")
    monkeypatch.setattr(oflm_add, "open_kernels_checkout", lambda: checkout)

    spec = ModelSpec(family="qwen3", hidden=4096, num_layers=36, layer_types=("dense",) * 36,
                     vocab=151936, real_vocab=151669, num_heads=32, num_kv_heads=8,
                     head_dim=128, rotary_dim=128, rope_theta=5e6, qk_norm=True,
                     attn_gate=False, intermediate=12288)
    oflm_add.build_geometry(spec, "qwen3vl", "8b")
    written = checkout / "recipes" / "specs" / "qwen3vl-8b.json"
    assert written.is_file(), "the spec goes where build-new-spec.py looks"
    assert ModelSpec.from_json(written.read_text()).kernel_hash() == spec.kernel_hash()
    assert marker.read_text() == "ok", "the build script runs (not a printed instruction)"


def test_a_failed_build_returns_none_so_the_refusal_can_say_so(tmp_path, monkeypatch):
    from recipes.spec import ModelSpec

    checkout = tmp_path / "open_kernels"
    (checkout / "recipes" / "specs").mkdir(parents=True)
    (tmp_path / "utilities").mkdir()
    (tmp_path / "utilities" / "build-new-spec.py").write_text("raise SystemExit(1)\n",
                                                              encoding="utf-8")
    monkeypatch.setattr(oflm_add, "open_kernels_checkout", lambda: checkout)
    spec = ModelSpec(family="qwen3", hidden=4096, num_layers=36, layer_types=("dense",) * 36,
                     vocab=151936, real_vocab=151669, num_heads=32, num_kv_heads=8,
                     head_dim=128, rotary_dim=128, rope_theta=5e6, qk_norm=True,
                     attn_gate=False, intermediate=12288)
    assert oflm_add.build_geometry(spec, "qwen3vl", "8b") is None


def test_no_checkout_means_no_build_and_no_script_advice(tmp_path, monkeypatch):
    """An installed package has no recipes and no toolchain: the size is ours to ship."""
    from recipes.spec import ModelSpec

    monkeypatch.setattr(oflm_add, "open_kernels_checkout", lambda: None)
    spec = ModelSpec(family="qwen3", hidden=4096, num_layers=36, layer_types=("dense",) * 36,
                     vocab=151936, real_vocab=151669, num_heads=32, num_kv_heads=8,
                     head_dim=128, rotary_dim=128, rope_theta=5e6, qk_norm=True,
                     attn_gate=False, intermediate=12288)
    assert oflm_add.build_geometry(spec, "qwen3vl", "8b") is None
    msg = oflm_add.kernel_set_refusal("repo", "qwen3vl", ["no installed set has this geometry"])
    assert "build-new-spec.py" not in msg, "no script the user cannot run"
    assert "not part of this release" in msg


def test_a_sibling_manifest_keeps_the_sets_head_and_takes_the_models_loop():
    """The base set's `vocab` (the head kernel's compiled row count) must survive.

    A sibling set is the base's kernels with this model's manifest: the depth, the
    layer sequence and the tokenizer's id count come from the model, everything else
    -- `vocab` above all -- stays the base's. Writing the model's own vocab would tell
    the engine a band count the compiled head does not have.
    """
    from recipes.spec import ModelSpec

    def spec(vocab, real_vocab, layers, family="qwen3"):
        return ModelSpec(family=family, hidden=2560, num_layers=layers,
                         layer_types=("dense",) * layers, vocab=vocab, real_vocab=real_vocab,
                         num_heads=8, num_kv_heads=4, head_dim=128, rotary_dim=128,
                         rope_theta=1e6, qk_norm=True, attn_gate=False, intermediate=10240)

    base = spec(262208, 262145, 34)          # a container padded to whole head bands
    model = spec(262144, 262144, 36)         # the same model's GGUF, its own token count
    assert base.serves(model) is None, "the set serves the narrower head"

    own = oflm_add.sibling_spec(base, model)
    assert own.vocab == 262208, "the head stays the one the kernel was compiled for"
    assert own.real_vocab == 262144, "the logits are masked at the model's own id count"
    assert own.num_layers == 36 and own.layer_types == ("dense",) * 36
    assert own.kernel_hash() == base.kernel_hash()
