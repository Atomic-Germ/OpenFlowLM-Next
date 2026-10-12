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


def write_kernel_set(xclbins, model_name, kernel_hash, spec_hash=None, spec=None):
    """A kernel set directory. `spec` is the spec dict a real manifest carries --
    the shape key is derivable from it, which is what makes a set written before
    kernel_hash still bind a fine-tune."""
    d = xclbins / model_name / "open_kernels"
    d.mkdir(parents=True)
    doc = {"manifest_version": 1, "family": "qwen3",
           "kernel_hash": kernel_hash, "spec_hash": spec_hash or kernel_hash}
    if spec is not None:
        doc["spec"] = spec
    (d / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    return d


@pytest.fixture
def model_dir(tmp_path):
    return write_model(tmp_path)


def test_kernel_hash_derives_from_the_model_directory(model_dir):
    """The spec comes off config.json + the container header, like the recipes."""
    kernel_hash, spec_hash, note = oflm_add.model_kernel_hash(model_dir)
    assert kernel_hash, note
    assert kernel_hash.startswith("sha256:")
    assert spec_hash.startswith("sha256:")
    # Stable: the same directory derives the same hashes.
    assert oflm_add.model_kernel_hash(model_dir)[0] == kernel_hash


def test_matching_spec_hash_wins_over_a_same_family_set(tmp_path, model_dir):
    kernel_hash, spec_hash, _ = oflm_add.model_kernel_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "AAA-Wrong-Shape-NPU2", OTHER_HASH)
    right = write_kernel_set(xclbins, "ZZZ-Same-Shape-NPU2", kernel_hash, spec_hash)

    found, source = oflm_add.find_open_kernels(kernel_hash, [xclbins], model_dir.name, spec_hash=spec_hash)
    assert found == right
    assert source == "ZZZ-Same-Shape-NPU2"


def test_the_models_own_directory_wins_when_several_match(tmp_path, model_dir):
    kernel_hash, spec_hash, _ = oflm_add.model_kernel_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "AAA-Same-Shape-NPU2", kernel_hash, spec_hash)
    mine = write_kernel_set(xclbins, model_dir.name, kernel_hash, spec_hash)

    found, source = oflm_add.find_open_kernels(kernel_hash, [xclbins], model_dir.name, spec_hash=spec_hash)
    assert found == mine
    assert source == model_dir.name


def test_no_match_selects_nothing(tmp_path, model_dir):
    kernel_hash, spec_hash, _ = oflm_add.model_kernel_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "Wrong-NPU2", OTHER_HASH)

    assert oflm_add.find_open_kernels(kernel_hash, [xclbins], model_dir.name, spec_hash=spec_hash) == (None, None)


def test_a_finetune_with_a_different_tokenizer_vocab_still_binds(tmp_path, model_dir):
    """The regression this key exists for.

    `real_vocab` sizes nothing -- `lm_rows` rounds `vocab` to a band -- so two
    models whose only difference is how many ids their tokenizer.json reports
    have byte-identical kernels. Matching on `spec_hash` (which covers
    `real_vocab`) refused the fine-tune and it silently fell back to closed
    kernels. `kernel_hash` is what makes the shape the binding key.
    """
    kernel_hash, spec_hash, _ = oflm_add.model_kernel_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    shipped = write_kernel_set(xclbins, "Shipped-NPU2", kernel_hash, spec_hash)

    # The installed manifest records the shipped model's REAL vocab. The fine-tune's
    # tokenizer.json reports a different id count, so its full spec hash moves.
    import recipes.spec as _rspec
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    spec_a = _rspec.ModelSpec.from_hf_config(cfg)                       # as shipped
    spec_b = _rspec.ModelSpec.from_hf_config(cfg, real_vocab=spec_a.real_vocab - 7)
    assert spec_a.spec_hash() != spec_b.spec_hash(), "the premise: spec_hash must move"
    assert spec_a.kernel_hash() == spec_b.kernel_hash(), "kernel_hash must NOT move"

    # The manifest a shipped set carries records the shipped model's own spec.
    manifest = json.loads((shipped / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["spec_hash"] == spec_a.spec_hash()

    # The bug: the fine-tune's own kernel_hash finds the set, even though its full
    # spec_hash differs from the manifest's. Before kernel_hash this returned None.
    assert spec_b.spec_hash() != manifest["spec_hash"]
    found, source = oflm_add.find_open_kernels(spec_b.kernel_hash(), [xclbins], model_dir.name)
    assert found == shipped
    assert source == "Shipped-NPU2"


def test_a_set_written_before_kernel_hash_still_binds_a_finetune(tmp_path, model_dir):
    """An ALREADY-INSTALLED set must not need a re-export to serve a fine-tune.

    `kernel_hash` is a pure function of a field set the manifest has always
    carried in its `spec` dict, so oflm-add derives it from there when the key is
    absent. Without this, every set built before the key could bind only a
    byte-identical spec, and a fine-tune was told to go build kernels that were
    already installed.
    """
    import recipes.spec as _rspec
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    spec_a = _rspec.ModelSpec.from_hf_config(cfg)
    spec_b = _rspec.ModelSpec.from_hf_config(cfg, real_vocab=spec_a.real_vocab - 7)
    kernel_hash, spec_hash, _ = oflm_add.model_kernel_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    d = write_kernel_set(xclbins, "Shipped-NPU2", kernel_hash, spec_hash, spec=spec_a.to_dict())
    # Strip the key, as a pre-kernel_hash manifest has it -- but KEEP spec.
    mj = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    del mj["kernel_hash"]
    (d / "manifest.json").write_text(json.dumps(mj), encoding="utf-8")
    assert spec_a.spec_hash() != spec_b.spec_hash()

    found, source = oflm_add.find_open_kernels(spec_b.kernel_hash(), [xclbins], "Finetune",
                                              spec_hash=spec_b.spec_hash())
    assert found == d
    assert source == "Shipped-NPU2"


def test_a_genuinely_different_shape_still_refuses(tmp_path, model_dir):
    """The shape key must not be so loose that a real difference binds."""
    kernel_hash, spec_hash, _ = oflm_add.model_kernel_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "Shipped-NPU2", kernel_hash, spec_hash)

    import recipes.spec as _rspec
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    other = _rspec.ModelSpec.from_hf_config({**cfg,
                                             "hidden_size": cfg["hidden_size"] * 2,
                                             "intermediate_size": cfg["intermediate_size"] * 2})
    assert other.kernel_hash() != kernel_hash
    assert oflm_add.find_open_kernels(other.kernel_hash(), [xclbins], "Other",
                                      spec_hash=other.spec_hash()) == (None, None)


def test_the_key_ignores_identity_but_respects_geometry(tmp_path, model_dir):
    """The boundary of the shape key, stated as a test.

    Identity must not matter -- a model nobody has seen the name of, with a
    different tokenizer id count, has to bind. Geometry must, because it is what
    the build actually compiles: `lm_rows` (dense.py) rounds `vocab` to a 64-row
    band and sizes the head from it, so a padded vocab IS a different head, and
    a padded/pruned model is genuinely a new shape rather than a mismatch to be
    papered over.
    """
    kernel_hash, spec_hash, _ = oflm_add.model_kernel_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    shipped = write_kernel_set(xclbins, "Shipped-NPU2", kernel_hash, spec_hash)

    import recipes.spec as _rspec
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    base = _rspec.ModelSpec.from_hf_config(cfg)

    def binds(spec):
        f, _s = oflm_add.find_open_kernels(spec.kernel_hash(), [xclbins], "Unheard-Of-Model",
                                          spec_hash=spec.spec_hash())
        return f == shipped

    # identity: not part of the key
    assert binds(base)
    assert binds(_rspec.ModelSpec.from_hf_config(cfg, real_vocab=base.real_vocab - 7))
    assert binds(_rspec.ModelSpec.from_dict({**base.to_dict(), "extra": {"model": "who-knows"}}))
    # geometry: part of the key
    assert not binds(_rspec.ModelSpec.from_dict({**base.to_dict(), "hidden": base.hidden * 2}))
    assert not binds(_rspec.ModelSpec.from_dict({**base.to_dict(), "intermediate": base.intermediate // 2}))
    assert not binds(_rspec.ModelSpec.from_dict({**base.to_dict(), "vocab": base.vocab + 128}))


def test_a_non_match_says_which_gap_it_is(tmp_path, model_dir, capsys):
    """Two different reasons a shape has no set; the user must be told which.

    A family with no recipe is a kernel gap. A shape the recipe builds fine but
    that was never built or shipped is a packaging gap. Reporting them the same
    way is what sends people off to add a name to a table.
    """
    xclbins = tmp_path / "xclbins"   # deliberately empty: nothing is installed
    oflm_add.setup_open_kernels(model_dir, model_dir.name, [xclbins], quiet=True)
    why = oflm_add._why_no_set(model_dir)
    assert why and "qwen3" in why
    # Exactly one of the three, and never a bare MISSING.
    assert any(p in why for p in ("packaging gap", "refuses this geometry",
                                  "no recipe")), why


def test_an_unknown_family_is_reported_as_a_kernel_gap(tmp_path):
    """qwen4_exp, for instance: the family itself has no recipe."""
    import recipes.spec as _rspec
    m = tmp_path / "q4exp"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"model_type": "qwen4_exp",
                                               "num_hidden_layers": 48}), encoding="utf-8")
    why = oflm_add._why_no_set(m)
    assert why and "no recipe" in why and "qwen4_exp" in why, why
    assert "prune" in why and "name" in why, "must say it is not a size/prune/name issue"


def test_setup_links_the_match_where_find_kernels_looks(tmp_path, model_dir, capsys):
    kernel_hash, spec_hash, _ = oflm_add.model_kernel_hash(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "Wrong-NPU2", OTHER_HASH)
    right = write_kernel_set(xclbins, "Right-NPU2", kernel_hash, spec_hash)

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
