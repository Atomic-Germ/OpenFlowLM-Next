"""The converter's default quant and the shipped kernel sets' quant must agree.

If a packer default drifts from what the kernels were built for, a freshly-packed
*base* model silently fails to bind the kernels that already exist for its shape —
and `oflm add` correctly, unhelpfully, reports it as a packaging gap.

The concrete case: every qwen3.5 config defaulted `linear_out_proj` (the
`ssm_out` projection) to `Q8_0`, so packs derived `quant={'linear_out':'q8'}` and
needed the `gemm_n8192_k4096` kernel, which was never built — the shipped 9B/27B
sets run `gemm_n4096_k4096` at q4_1. The host requantizes an out projection to
q4_1 for the kernel regardless (`pools.cpp`: "re-quantises a q8 source
transparently"), so the q8 storage bought nothing.

This test pins the config default AND asserts the derivation agrees with the
shipped spec's kernel_hash, so the two cannot drift apart again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "utilities" / "q4nx-build" / "configs"
OPEN_KERNELS = REPO / "open_kernels"
QWEN35 = sorted(CONFIGS.glob("qwen3.5_*.json"))


def _recipes_on_path():
    if str(OPEN_KERNELS) not in sys.path:
        sys.path.insert(0, str(OPEN_KERNELS))


def test_qwen35_configs_exist():
    assert QWEN35, "no qwen3.5 packer configs found"


@pytest.mark.parametrize("path", QWEN35, ids=lambda p: p.stem)
def test_linear_out_is_not_packed_at_q8(path: Path):
    """`linear_out` must stay at q4: the q8 variant demands an unbuilt kernel."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    param = cfg["name_map"]["linear_out_proj"]
    assert param["default_tensor_type"] != "Q8_0", (
        f"{path.name}: linear_out_proj packed at Q8_0 derives quant={{'linear_out':'q8'}}, "
        "which needs the gemm_n8192 kernel that was never built. The shipped qwen3.5 "
        "sets are q4_1.")


@pytest.mark.parametrize("path", QWEN35, ids=lambda p: p.stem)
def test_a_q4_packed_out_projection_derives_the_shipped_key(path: Path):
    """A q4-packed `ssm_out` must derive the SAME kernel_hash as the shipped spec.

    This is the end-to-end assertion the drift broke: the derived key has to equal
    the shipped `qwen35-*` spec's, or a freshly-packed base model won't bind the
    kernels that already exist for it.
    """
    _recipes_on_path()
    from recipes.spec import ModelSpec, quant_map_from_chunk_sizes

    stem = path.stem  # e.g. qwen3.5_9b
    size = stem.rsplit("_", 1)[-1]  # 9b
    spec_path = OPEN_KERNELS / "recipes" / "specs" / f"qwen35-{size}.json"
    if not spec_path.is_file():
        pytest.skip(f"no shipped qwen35-{size} spec yet")

    # A container whose out projection is q4 (5120-byte chunks), everything else q4.
    sizes = {
        "model.layers.0.linear_attn.ssm_out_proj.weight": 5120,
        "model.layers.0.linear_attn.qkv_proj.weight": 5120,
    }
    quant = quant_map_from_chunk_sizes("qwen35", sizes, {})
    assert quant.get("linear_out", "q4_1") == "q4_1", quant

    shipped = ModelSpec.from_json(spec_path.read_text(encoding="utf-8"))
    packed = ModelSpec.from_dict({**shipped.to_dict(), "quant": quant})
    assert packed.kernel_hash() == shipped.kernel_hash(), (
        "a q4-packed out projection must derive the shipped key, else freshly-packed "
        "qwen3.5 models cannot bind the shipped kernels")