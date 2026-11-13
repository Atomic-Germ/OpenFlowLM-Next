import json
import sys
from pathlib import Path

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402

REGISTRY = {
    "models": {
        "qwen3.5": {"9b": {"name": "Qwen3.5-9B-NPU2", "details": {"family": "qwen3.5"},
                              "size": 9000000000}},
        "qwen3": {"8b": {"name": "Qwen3-8B-NPU2", "details": {"family": "qwen3"},
                          "size": 8000000000}},
    }
}


def test_family_of_official_names_the_registry_family():
    assert oflm_add.family_of_official(REGISTRY, "Qwen3.5-9B-NPU2") == "qwen3.5"
    assert oflm_add.family_of_official(REGISTRY, "Nope-NPU2") is None


def test_a_finetune_inherits_the_family_of_the_model_it_is_built_on():
    """The README's base model is an official model: its family is the fine-tune's
    (the fine-tune runs on the base's kernels -- the same shape, same engine)."""
    fam, name = oflm_add.family_from_base_chain(REGISTRY, ["Qwen/Qwen3.5-9B"])
    assert fam == "qwen3.5"
    assert name == "Qwen3.5-9B-NPU2"
    # the container name and the repo slug name the same model
    assert oflm_add.family_from_base_chain(REGISTRY, ["Qwen3.5-9B-NPU2"])[0] == "qwen3.5"
    # an unrelated base says nothing
    assert oflm_add.family_from_base_chain(REGISTRY, ["someone/Some-Other-7B"])[0] is None


def test_the_geometry_route_finds_the_family_of_a_finetune(tmp_path):
    """A fine-tune whose name says nothing ("Ornith-1.5-9B") gets its family from
    the installed kernel set whose GEOMETRY it needs -- the depth and the
    tokenizer's id count are the manifest's business, not the kernels'."""
    spec = {"family": "qwen35", "hidden": 4096, "num_layers": 32,
            "layer_types": ["linear_attention"] * 32, "vocab": 248320, "real_vocab": 248070,
            "num_heads": 16, "num_kv_heads": 4, "head_dim": 256, "rotary_dim": 64,
            "rope_theta": 1000000.0, "rope_scaling": None, "rope_local_theta": 0.0,
            "sliding_window": 0, "qk_norm": True, "attn_gate": False,
            "lin_key_heads": 0, "lin_value_heads": 0, "lin_key_dim": 0, "lin_value_dim": 0,
            "conv_kernel": 0, "intermediate": 12288, "activation": "silu",
            "sandwich_norms": False, "num_experts": 0, "experts_per_tok": 0,
            "moe_intermediate": 0, "shared_expert_intermediate": 0, "norm_eps": 1e-06,
            "norm_groups": 1, "quant": "q4_1", "extra": {}}
    set_dir = tmp_path / "Qwen3.5-9B-NPU2" / "open_kernels"
    set_dir.mkdir(parents=True)
    (set_dir / "manifest.json").write_text(json.dumps({"family": "qwen35", "spec": spec}),
                                           encoding="utf-8")
    ModelSpec = oflm_add._model_spec_class()
    if ModelSpec is None:                     # no open_kernels checkout: the route is off
        return
    same = ModelSpec.from_dict(spec)
    deeper = ModelSpec.from_dict({**spec, "num_layers": 33, "layer_types": spec["layer_types"] + ["linear_attention"]})
    other = ModelSpec.from_dict({**spec, "hidden": 2048})
    assert same.kernel_hash() == deeper.kernel_hash(), "the depth must not move the kernel hash"
    assert same.kernel_hash() != other.kernel_hash()
    fam, name = oflm_add.family_from_kernels(REGISTRY, deeper, [tmp_path])
    assert (fam, name) == ("qwen3.5", "Qwen3.5-9B-NPU2")
    assert oflm_add.family_from_kernels(REGISTRY, other, [tmp_path]) == (None, None)


def test_the_name_is_the_last_resort():
    """An explicit --family wins; then the content; then the name."""
    assert oflm_add.derive_family(REGISTRY, "Qwen3-8B", explicit="phi4") == "phi4"
    assert oflm_add.derive_family(REGISTRY, "Qwen3-8B") == "qwen3"
    assert oflm_add.derive_family(REGISTRY, "Ornith-1.5-9B", content_families=["qwen3.5"]) == "qwen3.5"
    try:
        oflm_add.derive_family(REGISTRY, "Ornith-1.5-9B")
    except SystemExit as e:
        assert "--family" in str(e)
    else:
        raise AssertionError("an unnameable slug must ask for --family")


def test_the_gguf_gate_is_the_manifest_not_a_family_list(tmp_path):
    """The capability is the kernel set's manifest: a family-name list silently
    refuses the families whose runtime name differs from the recipe's
    ("qwen3.5" vs "qwen35")."""
    src = (OFLM_ADD / "oflm_add" / "__init__.py").read_text(encoding="utf-8")
    assert "GGUF_CAPABLE_FAMILIES" not in src
    good = tmp_path / "Qwen3.5-9B-NPU2" / "open_kernels"
    good.mkdir(parents=True)
    (good / "manifest.json").write_text(json.dumps({"family": "qwen35", "gguf": {}}), encoding="utf-8")
    stale = tmp_path / "stale" / "Qwen3.5-9B-NPU2" / "open_kernels"
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text(json.dumps({"family": "qwen35"}), encoding="utf-8")
    assert oflm_add.gguf_kernel_set("Qwen3.5-9B-NPU2", [tmp_path / "stale", tmp_path])
    assert oflm_add.gguf_kernel_set("Qwen3.5-9B-NPU2", [tmp_path / "stale"]) is None
