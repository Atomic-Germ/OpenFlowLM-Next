import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))

import oflm_support


def _write_model(root: Path, name: str, config: dict, tokenizer: dict | None = None) -> Path:
    model = root / name
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if tokenizer is not None:
        (model / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    header = {"__metadata__": {"format": "q4nx"},
              "model.layers.0.self_attn.q_proj.weight": {"dtype": "I8", "shape": [32, 5120]}}
    encoded = json.dumps(header).encode()
    (model / "model.q4nx").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    return model


def _qwen35_config() -> dict:
    spec_path = ROOT / "open_kernels" / "recipes" / "specs" / "qwen35-9b.json"
    d = json.loads(spec_path.read_text(encoding="utf-8"))
    s = d["spec"] if "spec" in d else d
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "hidden_size": s["hidden"],
        "num_hidden_layers": s["num_layers"],
        "num_attention_heads": s["num_heads"],
        "num_key_value_heads": s["num_kv_heads"],
        "head_dim": s["head_dim"],
        "intermediate_size": s["intermediate"],
        "vocab_size": s["vocab"],
        "rope_theta": s["rope_theta"],
        "partial_rotary_factor": s["rotary_dim"] / s["head_dim"],
        "rms_norm_eps": s["norm_eps"],
        "linear_num_key_heads": s["lin_key_heads"],
        "linear_num_value_heads": s["lin_value_heads"],
        "linear_key_head_dim": s["lin_key_dim"],
        "linear_value_head_dim": s["lin_value_dim"],
        "linear_conv_kernel_dim": s["conv_kernel"],
        "layer_types": s["layer_types"],
        "attn_output_gate": True,
    }


def _write_manifest(path: Path, model_dir: Path, recipes: Path) -> Path:
    # Use the recipe's own derivation and manifest serializer. This protects
    # tests from duplicating the family contract they intend to check.
    old = list(sys.path)
    sys.path.insert(0, str(recipes))
    try:
        from recipes.load import spec_from_model_dir
        from recipes.manifest import manifest
        m = manifest(spec_from_model_dir(model_dir))
    finally:
        sys.path[:] = old
    d = path / "Qwen3.5-9B-NPU2" / "open_kernels"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return d


def test_unknown_model_name_is_not_an_eligibility_gate(tmp_path):
    recipes = ROOT / "open_kernels"
    model = _write_model(tmp_path, "GRaPE-1.5-TEST-DONTDL", _qwen35_config(),
                         {"model": {"vocab": {"x": 0}}, "added_tokens": []})
    bundle = _write_manifest(tmp_path / "xclbins", model, recipes)

    report = oflm_support.inspect_model(model, [tmp_path / "xclbins"], recipes)

    assert report["identity"]["name"] == "GRaPE-1.5-TEST-DONTDL"
    assert report["requirements"]["kernel_family"] == "qwen35"
    assert report["selection"]["exact"]
    assert report["selection"]["exact"][0]["source"] == str(bundle)
    # Curated status is deliberately separate from the selection result.
    assert report["curated"]["model_list_entry"] is False


def test_different_repo_identity_does_not_change_contract(tmp_path):
    recipes = ROOT / "open_kernels"
    cfg = _qwen35_config()
    one_tokenizer = {"model": {"vocab": {"x": 0}}, "added_tokens": []}
    another_tokenizer = {"model": {"vocab": {"x": 1}}, "added_tokens": []}
    one = _write_model(tmp_path / "a", "Ornith-1.5-9B-NPU", cfg, one_tokenizer)
    two = _write_model(tmp_path / "b", "Qwen3.5-9B-Claude-NPU2", cfg, another_tokenizer)
    bundle = _write_manifest(tmp_path / "xclbins", one, recipes)

    a = oflm_support.inspect_model(one, [tmp_path / "xclbins"], recipes)
    b = oflm_support.inspect_model(two, [tmp_path / "xclbins"], recipes)

    assert a["requirements"]["spec_hash"] != b["requirements"]["spec_hash"]
    assert b["selection"]["family_shape"][0]["source"] == str(bundle)
    assert b["selection"]["family_shape"][0]["runtime_manifest_differences"] == ["real_vocab"]


def test_curated_family_donor_is_offered_without_candidate_registry_entry(tmp_path):
    recipes = ROOT / "open_kernels"
    model = _write_model(tmp_path / "models", "GRaPE-1.5-TEST-DONTDL", _qwen35_config(),
                         {"model": {"vocab": {"x": 0}}, "added_tokens": []})
    xclbins = tmp_path / "xclbins"
    donor = xclbins / "Qwen3.5-9B-NPU2"
    donor.mkdir(parents=True)
    model_list = tmp_path / "model_list.json"
    model_list.write_text(json.dumps({"models": {"qwen3.5": {"9b": {
        "name": "Qwen3.5-9B-NPU2", "details": {"family": "qwen3.5"}}}}}), encoding="utf-8")

    report = oflm_support.inspect_model(model, [xclbins], recipes, model_list)

    assert report["curated"]["model_list_entry"] is False
    assert report["requirements"]["runtime_family"] == "qwen3.5"
    assert report["curated"]["family_xclbin_donors"] == [{
        "name": "Qwen3.5-9B-NPU2", "path": str(donor),
        "family": "qwen3.5", "model_entry": False,
    }]
    assert "regardless" in report["next_action"]


def test_same_family_bundle_is_reported_even_when_shape_descriptor_differs(tmp_path):
    recipes = ROOT / "open_kernels"
    reference = _write_model(tmp_path / "ref", "Qwen3.5-9B-NPU2", _qwen35_config(),
                             {"model": {"vocab": {"x": 0}}, "added_tokens": []})
    candidate_cfg = _qwen35_config()
    candidate_cfg["intermediate_size"] += 512
    candidate = _write_model(tmp_path / "new", "Novel-Qwen-Finetune", candidate_cfg,
                              {"model": {"vocab": {"x": 0}}, "added_tokens": []})
    _write_manifest(tmp_path / "xclbins", reference, recipes)

    report = oflm_support.inspect_model(candidate, [tmp_path / "xclbins"], recipes)

    assert report["requirements"]["kernel_family"] == "qwen35"
    assert report["selection"]["exact"] == []
    assert len(report["selection"]["family_candidates"]) == 1
    assert not report["selection"]["family_candidates"][0]["contract_matches"]
    assert any("intermediate" in field
               for field in report["selection"]["family_candidates"][0]["contract_differences"])
    assert "not a refusal" in report["next_action"] or "candidates" in report["next_action"]


def test_curated_and_recipe_validity_are_reported_separately(tmp_path):
    model = _write_model(tmp_path, "Unofficial-Qwen-Finetune", _qwen35_config(),
                         {"model": {"vocab": {"x": 0}}, "added_tokens": []})
    report = oflm_support.inspect_model(model, [], ROOT / "open_kernels")

    assert report["recipe"]["status"] in ("validated", "catalogue-unvalidated")
    assert report["curated"]["checked_in_spec"] is None
    assert report["next_action"]


def test_structural_mismatch_is_named_not_hidden_by_model_name(tmp_path):
    cfg = _qwen35_config()
    cfg["hidden_act"] = "gelu"
    model = _write_model(tmp_path, "Qwen3.5-9B-NPU2", cfg,
                         {"model": {"vocab": {"x": 0}}, "added_tokens": []})
    report = oflm_support.inspect_model(model, [], ROOT / "open_kernels")

    assert report["recipe"]["status"] == "refused"
    assert "hidden_act" in report["recipe"]["error"]


def test_installed_mode_uses_manifest_without_recipe_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(oflm_support, "_recipe_root", lambda explicit=None: None)
    config = _qwen35_config()
    model = _write_model(tmp_path / "models", "Unlisted-Ornith-Finetune", config,
                         {"model": {"vocab": {"x": 0}}, "added_tokens": []})
    xclbins = tmp_path / "xclbins"
    manifest_dir = xclbins / "Qwen3.5-9B-NPU2" / "open_kernels"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps({
        "manifest_version": 1,
        "family": "qwen35",
        "spec_hash": "sha256:" + "ef" * 32,
        "hf_config_check": {"model_type": ["qwen3_5"], "hidden_size": config["hidden_size"]},
    }), encoding="utf-8")

    report = oflm_support.inspect_model(model, [xclbins], recipes=None)

    assert report["recipe"]["status"] == "unavailable"
    assert report["requirements"]["config_family"] is None
    assert len(report["selection"]["family_candidates"]) == 1
    assert report["selection"]["family_candidates"][0]["contract_matches"]
    assert "not a model-list refusal" in report["next_action"]


def test_unknown_architecture_prints_configurable_prefilled_issue(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(oflm_support, "_recipe_root", lambda explicit=None: None)
    monkeypatch.setenv("OFLM_SUPPORT_ISSUE_URL", "https://issues.example/new")
    model = _write_model(tmp_path, "Mystery-New-Family", {"model_type": "mystery"})

    assert oflm_support.main([str(model), "--xclbin-root", str(tmp_path / "empty"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["issue_url"].startswith("https://issues.example/new?")
    assert "model_type" in report["issue_url"]
