# Traces: OPEN-SPEC-DERIVE / OPEN-KERNEL-PIPELINE
"""The build-new-family front door: extend the nearest recipe instead of stopping."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "utilities" / "build-new-family.py"

_spec = importlib.util.spec_from_file_location("build_new_family_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
bf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bf
_spec.loader.exec_module(bf)


def load_spec_data(name: str) -> dict:
    return json.loads((ROOT / "open_kernels" / "recipes" / "specs" / f"{name}.json").read_text())


def derived_from(name: str, **kw) -> bf.Derived:
    data = load_spec_data(name)
    return bf.Derived(status="ok", path=ROOT / f"{name}.json", data=data,
                      family=str(data["family"]), spec_hash="s", kernel_hash="k",
                      matched_spec=name, **kw)


class FakeAdd:
    def split_remote_repo(self, repo):
        return "huggingface", repo

    def hf_file_tree(self, repo):
        raise RuntimeError("offline")


def args_for(repo: str, **kw):
    defaults = dict(repo=repo, repo_opt="", gguf="", family="", tensor_repo="",
                    spec_name="", model_name="", dry_run=True, skip_build=False,
                    skip_install=False, skip_validate=False, force=False, verbose=False,
                    work="", json="", cli="", ref_model="", install_dir="", ids="1,2,3",
                    base_set="")
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_spec_name_uses_family_and_size_without_duplicating_the_family():
    data = {"extra": {"model": "Qwen3-0.6B-NPU2"}}
    assert bf.spec_name_for("qwen3", "Qwen/Qwen3-0.6B", data) == "qwen3-0.6b"
    # punctuation variants of the family prefix are still "family + size"
    data = {"extra": {"model": "Qwen3.8-27B-NPU2"}}
    assert bf.spec_name_for("qwen38", "Qwen/Qwen3.8-27B", data) == "qwen3.8-27b"
    assert bf.spec_name_for("llama3", "org/Ornith-1.0-35B", {}) == "llama3-ornith-1.0-35b"


def test_probe_classification_distinguishes_served_and_would_build():
    served, _ = bf.classify_probe(
        "tag: qwen3\n  gguf kernels : /opt/openflowlm/share/oflm/xclbins/Qwen3-0.6B-NPU2/open_kernels\n", 0)
    would, _ = bf.classify_probe(
        "tag: qwen3\n  gguf kernels : (would build from qwen3-0.6b)\n", 0)
    # A safetensors repo has no `gguf kernels` line; its set is on `xclbin source`.
    config, _ = bf.classify_probe(
        "tag : qwen3:0.6b\nxclbin source  : Qwen3-0.6B-NPU2\n", 0)
    assert served == "served"
    assert would == "would_build"
    assert config == "served"


def test_an_existing_kernel_geometry_is_reported_as_served():
    d = derived_from("qwen3-0.6b")
    plan = bf.build_plan(args_for("Qwen/Qwen3-0.6B"), "Qwen/Qwen3-0.6B", "", [],
                         d, {"status": "served", "detail": ""}, FakeAdd())
    assert plan["status"] == "served"
    assert "qwen3-0.6b" in plan["reason"]
    assert plan["derived"]["kernel_hash"] == "k"
    assert plan["recipe"]["supported"] is True
    assert all("build-new-spec" not in c for c in plan["next_commands"])


def test_gguf_url_honours_a_quant_request(monkeypatch):
    seen = {}

    def choose(names, want=None):
        seen["want"] = want
        return "UD-Q4_K_XL/model-00001-of-00004.gguf", {}

    class QuantAdd(FakeAdd):
        def hf_file_tree(self, repo):
            return [{"path": "UD-Q4_K_XL/model-00001-of-00004.gguf"}]

        def gguf_file_names(self, tree):
            return [e["path"] for e in tree if e["path"].endswith(".gguf")]

        choose_gguf_file = staticmethod(choose)

    url = bf.gguf_url_for("unsloth/Qwen3.8-Flash-Next-GGUF", QuantAdd(),
                          quant="UD-Q4_K_XL")
    assert seen["want"] == "UD-Q4_K_XL"
    assert url.endswith("UD-Q4_K_XL/model-00001-of-00004.gguf")


def test_a_geometry_match_with_an_unloadable_gguf_is_not_served():
    """Ternary PQ2_0 matched qwen35-4b but no quant installs: that is not 'served'."""
    d = derived_from("qwen35-4b")
    plan = bf.build_plan(args_for("prism-ml/Ternary-Bonsai-2-4B-gguf"),
                         "prism-ml/Ternary-Bonsai-2-4B-gguf", "", [], d,
                         {"status": "error", "detail": "no compatible GGUF installs",
                          "info": {"weight_errors": [
                              "Ternary-Bonsai-PQ2_0-MTP-Q8_0.gguf: unknown ggml tensor types [142]"]}},
                         FakeAdd())
    assert plan["status"] == "weights_blocked"
    joined = "\n".join(plan["next_commands"])
    assert "no GGUF in the repo is loadable" in joined
    assert "oflm add" not in joined
    assert "cmake" not in joined


def test_a_clean_probe_still_reports_served():
    d = derived_from("qwen35-4b")
    plan = bf.build_plan(args_for("org/Model"), "org/Model", "", [], d,
                         {"status": "served", "detail": ""}, FakeAdd())
    assert plan["status"] == "served"


def test_quant_tokens_come_from_directories_and_file_stems():
    assert bf.quant_token("UD-Q4_K_XL/Model-00001-of-00004.gguf") == "UD-Q4_K_XL"
    assert bf.quant_token("Model-Q6_K.gguf") == "Q6_K"
    assert bf.quant_token("Model-PQ2_0-MTP-Q8_0.gguf") == "PQ2_0"
    assert bf.quant_token("Model.gguf") == ""


def test_the_plan_uses_the_installer_chosen_gguf_not_a_repick():
    out = ("weights  : Q6_K/Qwen3-Coder-Next-Q6_K-00001-of-00003.gguf -> model.gguf (+2 shards) "
           "(GGUF-direct, q4_1 pools)\n")
    assert bf.installed_gguf_name(out) == "Q6_K/Qwen3-Coder-Next-Q6_K-00001-of-00003.gguf"
    d = derived_from("qwen3-0.6b")
    plan = bf.build_plan(args_for("unsloth/Qwen3-Coder-Next-GGUF"),
                         "unsloth/Qwen3-Coder-Next-GGUF", "", [], d,
                         {"status": "would_build", "detail": "",
                          "info": {"gguf": "Q6_K/Model-Q6_K-00001-of-00003.gguf"}},
                         FakeAdd())
    assert plan["gguf_url"].endswith("Q6_K/Model-Q6_K-00001-of-00003.gguf")


def test_an_unknown_architecture_extends_the_nearest_recipe_and_builds():
    d = bf.Derived(status="ok", path=ROOT / "candidate.json",
                   data={"family": "qwen3", "hidden": 1024},
                   family="qwen3", arch="inkling2_loop")
    plan = bf.build_plan(args_for("org/Inkling"), "org/Inkling", "", [], d,
                         {"status": "skipped", "detail": ""}, FakeAdd(),
                         facts=bf.GGUFFacts(arch="inkling2_loop", family="inkling2_loop"),
                         recipe="qwen3", recipe_reasons=["qwen3: 12 geometry fields match"],
                         extending=True)
    assert plan["status"] == "would_extend"
    assert "qwen3" in plan["reason"]
    assert plan["recipe"]["nearest"] == "qwen3"
    joined = "\n".join(plan["next_commands"])
    assert "build-new-spec --spec" in joined
    assert "oflm add org/Inkling" in joined
    assert "exit 2" not in joined
    assert "register-family" not in joined


def test_an_unreadable_architecture_says_so_without_a_wall():
    d = bf.Derived(status="error", output="ERROR: general.architecture 'qwen4exp' has no recipe",
                   arch="qwen4exp")
    plan = bf.build_plan(args_for("org/Qwen4Exp"), "org/Qwen4Exp", "", [], d,
                         {"status": "unreadable", "detail": ""}, FakeAdd())
    assert plan["status"] == "would_extend"
    assert "qwen4exp" in plan["reason"]
    assert "GGUF header" in plan["reason"]


def test_explicit_family_names_the_recipe_to_extend():
    d = bf.Derived(status="ok", data={"family": "llama3"}, family="llama3",
                   arch="inkling2_loop")
    plan = bf.build_plan(args_for("org/Inkling", family="llama3"), "org/Inkling", "", [], d,
                         {"status": "skipped", "detail": ""}, FakeAdd(),
                         facts=bf.GGUFFacts(arch="inkling2_loop", family="inkling2_loop"),
                         extending=True)
    assert plan["status"] == "would_extend"
    assert "llama3" in plan["reason"]


def test_architecture_signals_ignore_the_architecture_prefix():
    facts = bf.GGUFFacts(arch="inkling2_loop", kv={
        "inkling2_loop.loop.count": 3,
        "inkling2_loop.block_count": 24,
        "inkling2_loop.activation.situ_beta": 4.0,
        "general.name": "Inkling",
    })
    joined = "\n".join(bf.architecture_signals(facts))
    assert "loop.count" in joined
    assert "situ_beta" in joined
    # `inkling2_loop.` itself contains "loop": the prefix must not match every key.
    assert "block_count" not in joined


def test_tensor_suffixes_strip_the_block_prefix_and_known_roles():
    facts = bf.GGUFFacts(arch="x", tensors=[
        ("blk.0.attn_q.weight", "f32", (1,)),
        ("blk.0.attn_q_conv.weight", "f32", (1,)),
        ("token_embd.weight", "f32", (1,)),
    ])
    assert bf.tensor_suffixes(facts) == ["attn_q_conv.weight"]


def test_nearest_recipe_picks_by_geometry_not_by_name():
    facts = bf.GGUFFacts(arch="inkling2_loop", family="inkling2_loop")
    data = {"head_dim": 128, "num_heads": 24, "num_kv_heads": 8, "rotary_dim": 128,
            "qk_norm": False, "attn_gate": True, "num_experts": 0,
            "experts_per_tok": 0, "moe_intermediate": 0, "hidden": 2304,
            "intermediate": 9216, "sliding_window": 0, "conv_kernel": 0,
            "lin_key_heads": 0, "lin_value_heads": 0}
    family, reasons = bf.nearest_recipe(data, facts)
    assert family in ("qwen3", "llama3", "gemma3", "hunyuan", "granite", "qwen2",
                      "qwen35", "qwen36moe", "lfm2", "lfm2moe", "lfm3", "k2",
                      "gemma4", "dense", "phi3")
    assert reasons


def test_promote_spec_refuses_to_clobber_a_different_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "SPECS", tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"family": "qwen3", "hidden": 1024}))
    dest = tmp_path / "qwen3-0.6b.json"
    dest.write_text(json.dumps({"family": "qwen3", "hidden": 2048}))
    d = bf.Derived(path=candidate)
    with pytest.raises(bf.StageError, match="already exists with different"):
        bf.promote_spec(d, "qwen3-0.6b", force=False)
    assert bf.promote_spec(d, "qwen3-0.6b", force=True) == dest


def test_dry_run_never_promotes_a_spec(tmp_path, monkeypatch):
    d = derived_from("qwen3-0.6b")
    specs = tmp_path / "specs"
    specs.mkdir()
    monkeypatch.setattr(bf, "SPECS", specs)
    monkeypatch.setattr(bf, "prepare_environment", lambda dry: ("python3", {}))
    monkeypatch.setattr(bf, "base_models_for", lambda repo, add: [])
    monkeypatch.setattr(bf, "gguf_url_for", lambda repo, add, quant="": "")
    monkeypatch.setattr(bf, "derive_spec", lambda *a, **k: d)
    monkeypatch.setattr(bf, "probe_install", lambda *a, **k:
                        {"status": "served", "detail": "existing set"})
    out = tmp_path / "plan.json"
    monkeypatch.setattr(sys, "argv", ["build-new-family.py", "Qwen/Qwen3-0.6B",
                                      "--dry-run", "--json", str(out)])
    assert bf.main() == 0
    plan = json.loads(out.read_text())
    assert plan["status"] == "served"
    assert list(specs.iterdir()) == []
