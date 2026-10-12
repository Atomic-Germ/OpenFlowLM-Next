# Traces: OPEN-SPEC-DERIVE (canonical spec: specs/open-engine/spec.md)
"""Every shipped spec names the model it builds kernels for, and no two claim
the same one.

`extra.model` is the export destination: `export_qwen36_kernels.py` writes
`src/xclbins/<extra.model>/open_kernels`, and `oflm add` links a converted
model's kernels by that same name. Two specs sharing a name means one silently
overwrites the other -- gemma3-12b shipped as "Gemma3-4B-NPU2" and the 4B sets
were overwritten on every full export (#54, fixed here). A name that does not
match the spec's own family and size is the same mistake waiting to happen, so
both are refused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SPECS = REPO / "open_kernels" / "recipes" / "specs"
sys.path.insert(0, str(REPO / "open_kernels"))

from recipes.naming import expected_model, is_safe_dir_name  # noqa: E402

# The naming rule itself lives in recipes/naming.py, shared with the generator
# that writes these files. Two copies of it is the same bug with extra steps.


def specs() -> list[Path]:
    return sorted(SPECS.glob("*.json"))


def test_specs_are_present():
    # A glob that matches nothing would make every test below pass vacuously.
    assert len(specs()) >= 12


def test_every_spec_is_declared_in_the_matrix():
    """A spec file with no matrix row is an unexplained artifact.

    The matrix is what makes coverage structural rather than a function of who
    last needed a size: every family x size we intend to support is declared,
    and the generator either produces a spec for it or prints the catalogue
    refusal that has to be resolved first. A spec that nobody declared is how
    the old list grew without anybody noticing what was missing.
    """
    import json

    recipes = SPECS.parent
    matrix = json.loads((recipes / "matrix.json").read_text(encoding="utf-8"))
    declared = {f"{e['slug']}.json" for e in matrix["entries"]}
    orphans = {p.name for p in specs()} - declared
    assert not orphans, f"spec(s) with no matrix row: {sorted(orphans)}"


def test_matrix_slugs_are_unique_and_nameable():
    import json

    matrix = json.loads((SPECS.parent / "matrix.json").read_text(encoding="utf-8"))
    slugs = [e["slug"] for e in matrix["entries"]]
    assert len(slugs) == len(set(slugs)), "duplicate slug in matrix.json"
    for slug in slugs:
        model = expected_model(slug)
        assert is_safe_dir_name(model), f"{slug}: {model!r} is not a safe directory name"


def test_a_gguf_derives_the_shape_its_kernels_were_built_for():
    """The point of the normalisation: a GGUF and an installed manifest describe
    the same model and must hash the same.

    Three things stood between them. The GGUF carries the MTP/NextN block the
    packer never converts (33 layers against the kernels' 32). `norm_eps` comes
    back from the GGUF's f32 as 9.999999974752427e-07 where the container had
    1e-06, and `spec_hash` hashes the float. And `real_vocab` is the tokenizer's
    id count, which llama.cpp's padded `vocab_size` cannot express -- so it is
    excluded from `kernel_hash`, which is the hash that asks "will these kernels
    fit", as distinct from `spec_hash`, which asks "is this the same model".

    Without all three, no GGUF ever matches any kernel set and a custom
    fine-tune of a model we already ship cannot be installed.
    """
    import json

    from recipes.spec import ModelSpec
    sys.path.insert(0, str(REPO / "open_kernels"))
    md = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 33,                 # 32 trunk + 1 NextN/MTP block
        "qwen35.nextn_predict_layers": 1,
        "qwen35.embedding_length": 4096,
        "qwen35.feed_forward_length": 12288,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.vocab_size": 248320,
        "qwen35.full_attention_interval": 4,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.time_step_rank": 32,
        "qwen35.ssm.inner_size": 4096,
        "qwen35.rope.dimension_count": 64,
        "qwen35.rope.freq_base": 10000000.0,
        # as it comes back out of the GGUF's f32, not the container's 1e-06
        "qwen35.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
    }
    from_gguf = ModelSpec.from_gguf_metadata(md)

    assert from_gguf.num_layers == 32, "the MTP block is not in the trunk"
    assert from_gguf.norm_eps == 1e-06

    # The same model as a container describes it: 32 layers, exact epsilon, and
    # the true vocabulary, which only tokenizer.json knows.
    container = json.loads(
        (SPECS / "qwen35-9b.json").read_text(encoding="utf-8"))
    assert container["num_layers"] == 32
    assert from_gguf.kernel_hash() == ModelSpec.from_dict(container).kernel_hash()


@pytest.mark.parametrize("path", specs(), ids=lambda p: p.stem)
def test_model_name_matches_the_spec(path: Path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    model = spec.get("extra", {}).get("model")
    assert model, f"{path.name}: extra.model is what the export is named after"
    want = expected_model(path.stem)
    assert model == want, f"{path.name} ({spec['family']}) exports as {model}; it should be {want}"


def test_no_two_specs_claim_one_model():
    seen: dict[str, str] = {}
    for path in specs():
        model = json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get("model")
        if model is None:
            continue
        assert model not in seen, (
            f"{path.name} and {seen[model]} both export as {model}; the second one "
            "overwrites the first in src/xclbins"
        )
        seen[model] = path.name


@pytest.mark.parametrize("path", specs(), ids=lambda p: p.stem)
def test_model_name_is_a_plain_directory_component(path: Path):
    model = json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get("model", "")
    assert model and "/" not in model and not model.startswith(".")
    # It becomes a path under src/xclbins and a store directory name.
    assert is_safe_dir_name(model), f"{path.name}: {model!r} is not a safe directory name"
