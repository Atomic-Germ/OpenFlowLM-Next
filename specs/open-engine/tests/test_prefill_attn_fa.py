"""OPEN-PREFILL-ATTN-FA: the dense flash-attention route's recipe emission.

Traces: OPEN-PREFILL-ATTN-FA (canonical spec: specs/open-engine/spec.md). The kernel and the full-model checks are manual.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "open_kernels"))

from recipes.manifest import manifest  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402

TIERS = [str(L) for L in range(256, 4097, 256)]


def _spec(name: str) -> ModelSpec:
    return ModelSpec.from_dict(json.loads((ROOT / "open_kernels" / "recipes" / "specs" / f"{name}.json").read_text()))


def test_granite_carries_attn_fa_on_both_prefill_routes():
    m = manifest(_spec("granite42-3b"))
    lt = m["layer_types"]["dense"]
    for gb in (lt["gemm_block"], lt["gemm_block_variants"]["bfp16"]):
        fa = gb["attn_fa"]
        assert (fa["hd"], fa["l_max"], fa["q_head"], fa["qsplit"]) == (64, 4096, 4096 * 9, True)
        assert list(fa["kernels"]) == TIERS
        assert [fa["kernels"][L] for L in TIERS] == [f"af{L}" for L in TIERS]
        assert "attn_block" not in gb
    # one xclbin, a stream per 256 rows of window
    assert m["contexts"]["af"] == "af256/final.xclbin"
    assert {m["kernels"][f"af{L}"]["context"] for L in TIERS} == {"af"}
    assert m["builds"]["af2048"]["env"] == {"AF_L": "2048", "AF_HEADS": "40", "AF_KV_HEADS": "8", "AF_KV_LD": "1024",
                                            "AF_MAX_ROWS": "4096", "AF_QSPLIT": "1"}
    # the stream reads the layer's KV cache in place: a row is every kv head's K, then every V, bf16
    assert lt["buffers"]["state"]["row"] == 2 * 8 * 64 * 2


def test_granite_prep_stream_rides_the_attention_context():
    # OPEN-PREFILL-GLUE-PREP: one more stream of the af context, for the qkv GEMM's padded width, manifest version 6
    m = manifest(_spec("granite42-3b"))
    lt = m["layer_types"]["dense"]
    for gb in (lt["gemm_block"], lt["gemm_block_variants"]["bfp16"]):
        assert gb["attn_fa"]["prep"] == {"kernel": "afp", "n_pad": 4096}
    assert m["kernels"]["afp"]["context"] == "af" and m["manifest_version"] == 6
    env = m["builds"]["afp"]["env"]
    assert (env["AF_MODE"], env["AF_QKV_NPAD"], env["AF_HEADS"], env["AF_KV_HEADS"], env["AF_KV_LD"]) == ("prep", "4096", "40", "8", "1024")


@pytest.mark.parametrize("name", ["qwen3-4b", "llama31-8b", "phi4-mini-4b", "k2-horizon-7b", "gemma3-4b"])
def test_other_families_carry_no_attn_fa(name):
    m = manifest(_spec(name))
    for lt in m["layer_types"].values():
        for gb in [lt.get("gemm_block") or {}, *(lt.get("gemm_block_variants") or {}).values()]:
            assert "attn_fa" not in gb, name
    assert "af" not in m["contexts"] and not [k for k in m["builds"] if k.startswith("af")], name
