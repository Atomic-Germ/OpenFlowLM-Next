"""OPEN-PREFILL-BFP16: the bfp16 block prefill route's packers and its manifest emission.

Traces: OPEN-PREFILL-BFP16 (canonical spec: specs/open-engine/spec.md). The kernel and the full-model checks are manual.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "open_kernels"))
sys.path.insert(0, str(ROOT / "open_kernels" / "designs" / "gemm_bfp16"))

import pack as bp  # noqa: E402
from recipes import pack  # noqa: E402
from recipes.manifest import manifest  # noqa: E402
from recipes.spec import ModelSpec  # noqa: E402
from test_pack_plan import _fnv1a  # noqa: E402
from test_qwen35 import _shared_q8_vector  # noqa: E402


def test_operand_round_trip_is_bfp16_rounding():
    rng = np.random.default_rng(0)
    for packer, unpacker, shape in ((bp.pack_a, bp.unpack_a, (256, 512)), (bp.pack_b, bp.unpack_b, (1024, 512))):
        x = rng.standard_normal(shape, dtype=np.float32)
        y = unpacker(packer(x), *shape)
        assert np.linalg.norm(x - y) / np.linalg.norm(x) < 8e-3


def test_tensor_by_tensor_pack_equals_the_stacked_pack():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((2048, 256), dtype=np.float32)
    dst = np.zeros(2048 * 256 // 8 * 9, np.uint8)
    pack.pack_bfp16_gemm(w[:512], 0, 2048, dst)
    pack.pack_bfp16_gemm(w[512:], 512, 2048, dst)
    assert np.array_equal(dst, bp.pack_b(w))


def test_c_untile_inverts_the_kernels_output_order():
    M, N = 512, 2048
    c = np.arange(M * N, dtype=np.float32).reshape(M, N)
    rb, j = 64 * 4, N // (128 * 8)
    flat = c.reshape(M // rb, rb, j, 8, 128).transpose(3, 0, 2, 1, 4).reshape(-1)     # column c, row block, tile, row, col
    assert np.array_equal(bp.untile_c(flat, M, N), c)


def test_pool_pack_byte_identical_to_pools_test():
    """pools_test.cpp's bfp16_gemm_pack hashes the same 24 q4_1 chunks as a [128, 1536] tensor at row 256 of 1024."""
    q4 = pack.requant_q4_1(_shared_q8_vector(12))
    w = pack.f32_of_q4_1(np.concatenate([q4, q4]), 128, 1536)
    dst = np.zeros(1024 * 1536 // 8 * 9, np.uint8)
    pack.pack_bfp16_gemm(w, 256, 1024, dst)
    assert _fnv1a(dst.tobytes()) == 0xB7AECE12E88C13AF


def _granite():
    return ModelSpec.from_dict(json.loads((ROOT / "open_kernels" / "recipes" / "specs" / "granite42-3b.json").read_text()))


def test_granite_manifest_carries_the_bfp16_variant():
    m = manifest(_granite())
    assert m["manifest_version"] == 4
    lt = m["layer_types"]["dense"]
    v, base = lt["gemm_block_variants"]["bfp16"], lt["gemm_block"]
    assert all(s.get("fmt") is None for s in base["program"])
    assert [s["n_pad"] for s in v["program"]] == [4096, 3072, 8192, 8192, 3072]
    assert [s["kernel"] for s in v["program"]][4] == "gemm_bfp16_n3072_k8192"
    assert v["kind"] == base["kind"] and v["t"] == base["t"] and v["qw"] == base["qw"]
    qkv = v["weights"]["gqkv3_w"]["pack"]
    assert [(o["row0"], o["n_total"]) for o in qkv] == [(0, 4096), (2560, 4096), (3072, 4096)]
    assert all(o["row0"] % 128 == 0 for w in v["weights"].values() for o in w["pack"])
    # one context for every shape, and the globals hold the packed x and the padded y
    assert m["contexts"]["gemmbfp"].endswith("/final.xclbin")
    assert m["globals"]["gbfp_x_k2560"] == 256 * 2560 // 8 * 9 and m["globals"]["gbfp_y_n3072"] == 3072 * 256 * 4
    assert m["builds"]["gemm_bfp16_n8192_k2560"]["env"] == {"GB_M": "256", "GB_K": "2560", "GB_N": "8192"}


@pytest.mark.parametrize("name", ["qwen3-4b", "llama31-8b", "phi4-mini-4b"])
def test_other_families_carry_no_bfp16_variant(name):
    spec = ModelSpec.from_dict(json.loads((ROOT / "open_kernels" / "recipes" / "specs" / f"{name}.json").read_text()))
    m = manifest(spec)
    assert all("bfp16" not in (lt.get("gemm_block_variants") or {}) for lt in m["layer_types"].values())
    assert m["manifest_version"] < 4
