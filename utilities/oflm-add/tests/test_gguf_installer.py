import json
import sys
from pathlib import Path


OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402


def test_choose_gguf_prefers_q4_1_and_refuses_multipart():
    chosen, refused = oflm_add.choose_gguf_file([
        "model-Q8_0.gguf",
        "model-Q4_0.gguf",
        "model-Q4_1-00001-of-00002.gguf",
        "model-Q4_1.gguf",
    ])

    assert chosen == "model-Q4_1.gguf"
    assert "multi-part GGUF" in refused["model-Q4_1-00001-of-00002.gguf"]


def test_local_q4nx_can_take_precedence_over_gguf(tmp_path):
    (tmp_path / "model.q4nx").touch()
    tree = [{"path": "weights-Q4_1.gguf"}]

    assert oflm_add.repo_has_q4nx(tree, tmp_path)


def test_modelscope_tree_dict_and_remote_q4nx_precedence(tmp_path):
    tree = {
        "model.q4nx": {"Path": "model.q4nx"},
        "weights-Q4_1.gguf": {"Path": "weights-Q4_1.gguf"},
        "nested/other.gguf": {"Path": "nested/other.gguf"},
    }

    assert oflm_add.root_file_names(tree) == ["model.q4nx", "weights-Q4_1.gguf"]
    assert oflm_add.repo_has_q4nx(tree, tmp_path)


def test_modelscope_selected_gguf_downloads_as_canonical_name(monkeypatch, tmp_path):
    selected = "weights-Q4_1.gguf"
    tree = {
        selected: {"Path": selected, "Size": 123, "Sha256": "ABCD"},
    }
    calls = []

    monkeypatch.setattr(oflm_add, "ms_file_tree", lambda repo: ("modelscope.test", tree))
    monkeypatch.setattr(
        oflm_add,
        "download_file",
        lambda url, dest, **kwargs: calls.append((url, dest, kwargs)),
    )

    obtained = oflm_add.fetch_assets(
        "Org/Model", tmp_path, modelscope=True, gguf_name=selected, quiet=True
    )

    assert obtained == ["model.gguf"]
    url, dest, kwargs = calls[0]
    assert url.endswith(f"/resolve/master/{selected}")
    assert dest == tmp_path / "model.gguf"
    assert kwargs["expected_size"] == 123
    assert kwargs["expected_sha"] == "abcd"


def test_explicit_kernel_manifest_can_enable_gguf(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"gguf": {}}), encoding="utf-8")
    assert oflm_add.manifest_supports_gguf(manifest)

    manifest.write_text("{}", encoding="utf-8")
    assert not oflm_add.manifest_supports_gguf(manifest)
