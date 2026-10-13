# Traces: OPEN-REGISTRY-SPECULATIVE (canonical spec: specs/open-engine/spec.md)
#
# oflm-add writes the registry entry's `speculative` field from the kernel set it
# links: a full install of a synthetic model against a kernel set with or without
# a verify program, then the entry read back from the registry it wrote.
import json
import struct
import sys
from pathlib import Path

import pytest

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402

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
Q4NX_HEADER = {
    "__metadata__": {"format": "q4nx"},
    "model.layers.0.self_attn.q_proj.weight": {"dtype": "I8", "shape": [4, 5120]},
    "model.layers.0.mlp.down_proj.weight": {"dtype": "I8", "shape": [4, 5120]},
}
VERIFY = {"width": 2, "slot_bytes": 4096, "act_bytes": 4096, "programs": {}, "tail": [], "xres": "vxres",
          "logits": "vlogits"}


def write_repo(root):
    d = root / "repo" / "Synth-4B-NPU2"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    blob = json.dumps(Q4NX_HEADER).encode()
    with open(d / "model.q4nx", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    (d / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {}}}), encoding="utf-8")
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return d


@pytest.fixture
def install(tmp_path, monkeypatch):
    """Run `oflm-add <repo> [args]` against a kernel set with or without a verify block; the entry it wrote."""
    repo = write_repo(tmp_path)
    spec_hash, note = oflm_add.model_spec_hash(repo)
    assert spec_hash, note
    system_list = tmp_path / "system" / "model_list.json"
    system_list.parent.mkdir()
    system_list.write_text(json.dumps({"models": {}}), encoding="utf-8")
    registry = tmp_path / "user" / "model_list.json"
    registry.parent.mkdir()
    monkeypatch.setattr(oflm_add, "find_system_xclbin_root", lambda: None)

    def run(verify, *args):
        kernels = tmp_path / "xclbins" / "Synth-Family-NPU2" / "open_kernels"
        kernels.mkdir(parents=True)
        manifest = {"manifest_version": 1, "family": "qwen3", "spec_hash": spec_hash}
        if verify:
            manifest["verify"] = VERIFY
        (kernels / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "oflm-add", str(repo), "--tag", "synth:4b", "--family", "qwen3", "--quiet",
            "--system-list", str(system_list), "--config", str(registry),
            "--models-root", str(tmp_path / "models"), "--xclbin-dir", str(tmp_path / "xclbins"), *args,
        ])
        oflm_add.main()
        return json.loads(registry.read_text(encoding="utf-8"))["models"]["synth"]["4b"]

    run.registry = registry
    return run


def test_a_kernel_set_with_verify_declares_lookup(install):
    entry = install(True)
    assert entry["speculative"] == {"method": "lookup"}


def test_speculative_none_writes_no_field(install):
    entry = install(True, "--speculative", "none")
    assert "speculative" not in entry


def test_lookup_on_a_kernel_set_without_verify_is_refused(install):
    with pytest.raises(SystemExit) as e:
        install(False, "--speculative", "lookup")
    assert "has no verify program" in str(e.value)
    assert not install.registry.exists()


def test_a_kernel_set_without_verify_declares_nothing(install):
    entry = install(False)
    assert "speculative" not in entry


# ------------------------------------------------------------------ dflash2
#
# method dflash2 names a drafter container (`draft`): oflm-add installs it beside the model, links the
# drafter's own kernel set, and refuses when the pieces do not fit.

BONSAI_LIKE = dict(CONFIG, hidden_size=5120, vocab_size=248320, num_hidden_layers=64, intermediate_size=17408,
                   num_attention_heads=24, num_key_value_heads=4, head_dim=256)
DRAFTER_CONFIG = {
    "model_type": "dflash2", "architectures": ["DFlash2Drafter"], "hidden_size": 5120, "num_hidden_layers": 5,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 17408,
    "rms_norm_eps": 1e-6, "rope_theta": 10000000.0, "sliding_window": 2048, "target_layer_ids": [5, 19, 33, 47, 61],
    "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256, "selector_top_k": 16, "mask_token_id": 248070,
    "vocab_size": 248320, "hadamard": {"block_size": 1024, "basis": "unsigned"},
}


def write_container(d, config, header):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    blob = json.dumps(header).encode()
    with open(d / "model.q4nx", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    return d


@pytest.fixture
def install_draft(tmp_path, monkeypatch):
    """`oflm-add <Bonsai-like target> [args]` with a drafter container at tmp/Synth-Draft and kernel sets for both."""
    repo = tmp_path / "repo" / "Synth-27B-NPU2"
    write_container(repo, BONSAI_LIKE, Q4NX_HEADER)
    (repo / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {}}}), encoding="utf-8")
    (repo / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    target_hash, note = oflm_add.model_spec_hash(repo)
    assert target_hash, note
    system_list = tmp_path / "system" / "model_list.json"
    system_list.parent.mkdir()
    system_list.write_text(json.dumps({"models": {}}), encoding="utf-8")
    registry = tmp_path / "user" / "model_list.json"
    registry.parent.mkdir()
    monkeypatch.setattr(oflm_add, "find_system_xclbin_root", lambda: None)

    def run(*args, verify=True, drafter_config=DRAFTER_CONFIG, draft_kernels=True, draft_args=True):
        draft = write_container(tmp_path / "drafts" / "Synth-Draft", drafter_config, {"__metadata__": {"format": "q4nx"}})
        draft_hash, note = oflm_add.model_spec_hash(draft)
        assert draft_hash, note
        for name, family, h, with_verify in (("Synth-Family-NPU2", "qwen3", target_hash, verify),
                                             ("Synth-Draft-NPU2", "dflash2", draft_hash, False)):
            if name == "Synth-Draft-NPU2" and not draft_kernels:
                continue
            k = tmp_path / "xclbins" / name / "open_kernels"
            k.mkdir(parents=True)
            manifest = {"manifest_version": 1, "family": family, "spec_hash": h}
            if with_verify:
                manifest["verify"] = VERIFY
            (k / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "oflm-add", str(repo), "--tag", "synth:27b", "--family", "qwen3", "--quiet",
            "--system-list", str(system_list), "--config", str(registry),
            "--models-root", str(tmp_path / "models"), "--xclbin-dir", str(tmp_path / "xclbins"),
            *(["--draft-model", str(draft)] if draft_args else []), *args,
        ])
        oflm_add.main()
        return json.loads(registry.read_text(encoding="utf-8"))["models"]["synth"]["27b"]

    run.registry = registry
    run.models = tmp_path / "models"
    run.draft_kernels = tmp_path / "xclbins" / "Synth-Draft-NPU2" / "open_kernels"
    return run


def test_draft_model_declares_dflash2_and_installs_the_drafter(install_draft):
    entry = install_draft("--speculative", "dflash2")
    assert entry["speculative"] == {"method": "dflash2", "draft": "Synth-Draft"}
    drafter = install_draft.models / "Synth-Draft"
    assert json.loads((drafter / "config.json").read_text(encoding="utf-8"))["model_type"] == "dflash2"
    assert (drafter / "model.q4nx").is_file()
    assert (drafter / "open_kernels").resolve() == install_draft.draft_kernels.resolve()


def test_draft_model_alone_implies_dflash2(install_draft):
    entry = install_draft()
    assert entry["speculative"] == {"method": "dflash2", "draft": "Synth-Draft"}


def test_dflash2_on_a_kernel_set_without_verify_is_refused(install_draft):
    with pytest.raises(SystemExit) as e:
        install_draft("--speculative", "dflash2", verify=False)
    assert "has no verify program" in str(e.value)
    assert not install_draft.registry.exists()
    assert not (install_draft.models / "Synth-Draft").exists()


def test_a_drafter_for_another_target_is_refused(install_draft):
    with pytest.raises(SystemExit) as e:
        install_draft(drafter_config=dict(DRAFTER_CONFIG, hidden_size=4096))
    assert "trained for another target" in str(e.value)
    assert not install_draft.registry.exists()


def test_a_drafter_that_taps_a_missing_layer_is_refused(install_draft):
    with pytest.raises(SystemExit) as e:
        install_draft(drafter_config=dict(DRAFTER_CONFIG, target_layer_ids=[5, 19, 33, 47, 99]))
    assert "taps layer 99" in str(e.value)


def test_a_drafter_without_its_kernel_set_is_refused_with_the_build_command(install_draft):
    with pytest.raises(SystemExit) as e:
        install_draft(draft_kernels=False)
    assert "no installed open kernel set has its spec_hash" in str(e.value)
    assert "export_qwen36_kernels.py" in str(e.value)
    assert not install_draft.registry.exists()


def test_dflash2_without_a_draft_model_is_refused(install_draft):
    with pytest.raises(SystemExit) as e:
        install_draft("--speculative", "dflash2", draft_args=False)
    assert "needs --draft-model" in str(e.value)


def test_draft_model_with_lookup_is_refused(install_draft):
    with pytest.raises(SystemExit) as e:
        install_draft("--speculative", "lookup")
    assert "goes with --speculative dflash2" in str(e.value)


def test_a_draft_model_that_is_not_a_drafter_is_refused(install_draft, tmp_path):
    with pytest.raises(SystemExit) as e:
        install_draft(drafter_config=dict(DRAFTER_CONFIG, model_type="qwen3"))
    assert "model_type is not dflash2" in str(e.value)
