"""A q4nx model's content, not its repo name, selects the runtime family."""
import json
import struct
import sys
from pathlib import Path

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402


def _registry():
    return {"models": {
        "qwen3.5": {"9b": {
            "name": "Qwen3.5-9B-NPU2", "size": 9_000_000_000,
            "details": {"family": "qwen3.5"},
        }},
        "qwen2": {"3b": {
            "name": "Qwen2.5-3B-NPU2", "size": 3_000_000_000,
            "details": {"family": "qwen2"},
        }},
    }}


def test_content_family_beats_misleading_repo_name_and_official_hint():
    config = {"model_type": "some_wrapper", "text_config": {"model_type": "qwen3_5_text"}}
    wrong_official = {"details": {"family": "qwen2"}}

    assert oflm_add.family_from_config(config) == "qwen3.5"
    assert oflm_add.derive_family(
        _registry(), "GRaPE-1.5-TEST-DONTDL", base_entry=wrong_official, config=config
    ) == "qwen3.5"


def test_q4nx_build_family_frontmatter_is_a_content_fallback(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("---\nlicense: apache-2.0\noflm-family: qwen3.5\ntags:\n- q4nx\n---\n", encoding="utf-8")
    wrong_official = {"details": {"family": "qwen2"}}

    assert oflm_add.family_from_readme(readme) == "qwen3.5"
    assert oflm_add.derive_family(
        {"models": {}}, "GRaPE-1.5-TEST-DONTDL", base_entry=wrong_official,
        readme_family=oflm_add.family_from_readme(readme)
    ) == "qwen3.5"


def test_readme_prose_and_tags_are_not_used_as_family_signals(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "---\ntags:\n- qwen3_5\n---\nThis is a Qwen3.5 fine-tune.\n", encoding="utf-8"
    )
    assert oflm_add.family_from_readme(readme) is None


def test_unknown_repo_name_can_be_classified_without_a_registry_entry():
    config = {"model_type": "qwen3_5"}
    assert oflm_add.derive_family(
        {"models": {}}, "GRaPE-1.5-TEST-DONTDL", config=config
    ) == "qwen3.5"


def test_unseen_finetune_names_all_resolve_from_the_same_qwen_content():
    config = {"model_type": "qwen3_5"}
    names = [
        "Qwen3.8-Distilled-2B-NPU2",
        "Cyber-Ornith-1.5-9B-NPU2",
        "Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-NPU2",
        "Ornith-1.5-9B-NPU2",
        "Qwen3.5-9B-Claude-4.8-Opus-NPU2",
        "Qwen3.8-Distilled-Heretic-9B-NPU2",
        "Ornith1.5-Heretic-9B-NPU2",
        "Qwopus3.5-9B-Coder-NPU2",
    ]
    for name in names:
        assert oflm_add.derive_family({"models": {}}, name, config=config) == "qwen3.5"


def test_family_and_size_select_a_known_good_xclbin_donor_for_unlisted_model():
    registry = _registry()
    official, note = oflm_add.resolve_official(
        registry, "Ornith-1.5-9B-NPU2", "qwen3.5", 9_000_000_000
    )

    assert official is not None
    assert official[3]["name"] == "Qwen3.5-9B-NPU2"
    assert note is None


def test_family_donor_does_not_require_exact_tag_or_repo_identity():
    registry = _registry()
    official, note = oflm_add.resolve_official(
        registry, "Qwopus3.5-9B-Coder-NPU2", "qwen3.5", 8_700_000_000
    )

    assert official is not None
    assert official[3]["details"]["family"] == "qwen3.5"
    assert official[3]["name"] == "Qwen3.5-9B-NPU2"
    assert "differs from official" in note


def test_content_size_beats_a_repo_name_that_looks_like_another_size():
    registry = _registry()
    registry["models"]["qwen3.5"]["2b"] = {
        "name": "Qwen3.5-2B-NPU2", "size": 2_000_000_000,
        "details": {"family": "qwen3.5"},
    }

    official, note = oflm_add.resolve_official(
        registry, "Qwen3.5-9B-NPU2", "qwen3.5", 2_000_000_000
    )

    assert official[3]["name"] == "Qwen3.5-2B-NPU2"
    assert note is None


def test_content_does_not_override_an_explicit_family_override():
    config = {"model_type": "qwen3_5"}
    assert oflm_add.derive_family(
        _registry(), "GRaPE-1.5-TEST-DONTDL", explicit="qwen2", config=config
    ) == "qwen2"


def test_content_family_map_covers_qwen_multimodal_variants():
    assert oflm_add.family_from_config({"model_type": "qwen3_5_omni"}) == "qwen3.5-omni"
    assert oflm_add.family_from_config({"model_type": "qwen2_vl"}) == "qwen2vl"


def test_support_issue_link_is_opt_in_and_destination_configurable(monkeypatch):
    monkeypatch.delenv("OFLM_SUPPORT_ISSUE_URL", raising=False)
    assert oflm_add.support_issue_url("title", "body") is None
    monkeypatch.setenv("OFLM_SUPPORT_ISSUE_URL", "https://example.test/issues/new")
    link = oflm_add.support_issue_url("Need support", "model=Unknown")
    assert link.startswith("https://example.test/issues/new?")
    assert "title=Need+support" in link
    assert "body=model%3DUnknown" in link


def test_unknown_content_and_name_can_be_deferred_until_download():
    assert oflm_add.derive_family(
        {"models": {}}, "unknown-repo-name", allow_unknown=True
    ) is None


def test_name_without_parameter_marker_gets_tag_from_config_size():
    assert oflm_add.derive_tag("GRaPE-1.5-TEST-DONTDL", size_hint=4_500_000_000) == \
        "grape-1.5-test-dontdl:4.5b"


def test_config_geometry_can_correct_a_misleading_name_size():
    assert oflm_add.derive_tag(
        "Qwen3.5-9B-Claude-NPU2", size_hint=2_000_000_000,
        size_authoritative=True,
    ) == "qwen3.5-claude:2b"


def test_qwen35_family_size_comes_from_hidden_geometry_not_repo_name():
    config = {"model_type": "qwen3_5", "hidden_size": 4096}
    assert oflm_add.size_from_config(config) == 9_000_000_000


def test_install_unknown_hf_style_directory_uses_content_family_and_size(monkeypatch, tmp_path):
    source = tmp_path / "source" / "GRaPE-1.5-TEST-DONTDL"
    source.mkdir(parents=True)
    config = {
        "model_type": "qwen3_5", "hidden_size": 4096, "num_hidden_layers": 32,
        "intermediate_size": 12288, "vocab_size": 248320,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "README.md").write_text(
        "---\noflm-family: qwen3.5\n---\n", encoding="utf-8"
    )
    (source / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"x": 0}}}), encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    header = json.dumps({"__metadata__": {"format": "q4nx"}}).encode()
    (source / "model.q4nx").write_bytes(struct.pack("<Q", len(header)) + header)

    system_list = tmp_path / "system" / "model_list.json"
    system_list.parent.mkdir()
    system_list.write_text(json.dumps(_registry()), encoding="utf-8")
    models = tmp_path / "user" / "models"
    user_list = tmp_path / "user" / "model_list.json"
    xclbin_root = tmp_path / "system" / "xclbins"
    selected = []

    monkeypatch.setattr(oflm_add, "find_system_xclbin_root", lambda: xclbin_root)
    monkeypatch.setattr(oflm_add, "link_xclbins",
                        lambda root, user_root, model, donor, **kw: selected.append((model, donor)))
    monkeypatch.setattr(oflm_add, "setup_open_kernels", lambda *a, **kw: False)
    monkeypatch.setattr(oflm_add, "write_vision_sidecar", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", [
        "oflm-add", str(source), "--system-list", str(system_list),
        "--config", str(user_list), "--models-root", str(models),
        "--xclbin-dir", str(tmp_path / "user" / "xclbins"), "--quiet",
    ])

    oflm_add.main()

    installed = models / "GRaPE-1.5-TEST-DONTDL"
    registry = json.loads(user_list.read_text(encoding="utf-8"))
    entry = registry["models"]["grape-1.5-test-dontdl"]["9b"]
    assert entry["details"]["family"] == "qwen3.5"
    assert selected == [("GRaPE-1.5-TEST-DONTDL", "Qwen3.5-9B-NPU2")]
    assert (installed / "README.md").is_file()
