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
