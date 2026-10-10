# Traces: OPEN-PREFILL-GEMM8 (canonical spec: specs/open-engine/spec.md)
"""The dense block route on 8-bit GEMMs (designs/dit_gemm): what the recipe writes for it, and the
pack op that makes its weight copy at load."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "open_kernels"))
SPECS = ROOT / "open_kernels" / "recipes" / "specs"


def _manifest(name):
    from recipes.load import load_spec
    from recipes.manifest import manifest
    return manifest(load_spec(SPECS / name))


def test_qwen3_8b_runs_the_8bit_route_by_default_and_keeps_q4_as_lean():
    m = _manifest("qwen3-8b.json")
    lt = m["layer_types"]["dense"]
    gb, lean = lt["gemm_block"], lt["gemm_block_variants"]["lean"]
    assert m["manifest_version"] == 4
    assert (gb["t"], gb["gemm"], lean["t"]) == (1024, "dit", 256) and "gemm" not in lean
    assert [s["kernel"] for s in gb["program"]] == ["dit_n6144_k4096", "dit_n4096_k4096", "dit_n12288_k4096",
                                                    "dit_n12288_k4096", "dit_n4096_k12288"]
    assert [s["args"] for s in gb["program"]] == [[w, "g8_a", "g8_c"] for w in
                                                  ("g8qkv3_w", "g8o_w", "g8gate_w", "g8up_w", "g8down_w")]
    # q, k and v land side by side as B's columns: 9 bits a weight, 4096 rows of q before k
    qkv = gb["weights"]["g8qkv3_w"]["pack"]
    assert [(o["op"], o["tensor"].split(".")[-2], o["dst"]) for o in qkv] == [
        ("bfp16_dit", "q_proj", 0), ("bfp16_dit", "k_proj", 4096 * 4096 * 9 // 8),
        ("bfp16_dit", "v_proj", 5120 * 4096 * 9 // 8)]
    assert gb["attn_block"] == lean["attn_block"] and gb["attn_block"]["prep"] == "qknorm_rope"
    assert m["contexts"]["dit"] == "dit_n6144_k4096/final.xclbin"
    assert {k: m["kernels"][k]["context"] for k in m["kernels"] if k.startswith("dit")} == dict.fromkeys(
        ("dit_n6144_k4096", "dit_n4096_k4096", "dit_n12288_k4096", "dit_n4096_k12288"), "dit")
    assert m["builds"]["dit_n4096_k12288"]["env"] == {"DG_M": "1024", "DG_K": "12288", "DG_N": "4096", "DG_LAYOUT": "{}"}
    assert (m["globals"]["g8_a"], m["globals"]["g8_c"]) == (1024 * 12288 * 2, 1024 * 12288 * 2)


def test_no_8bit_route_where_a_shape_does_not_fit_or_the_attention_is_not_products():
    # Qwen3-4B's o_proj is 2560 wide (dit_gemm wants N % 1024); Llama 3.1 and K2 declare no products prep
    for name in ("qwen3-4b.json", "llama31-8b.json", "k2-horizon-3.7b.json"):
        m = _manifest(name)
        gb = m["layer_types"]["dense"]["gemm_block"]
        assert gb.get("gemm", "q4") == "q4" and "gemm_block_variants" not in m["layer_types"]["dense"], name
        assert not [k for k in m["kernels"] if k.startswith("dit")] and m["manifest_version"] < 4, name


def test_bfp16_dit_packs_byte_identical_to_pools_test():
    """24 shared chunks at q4_1, [128, 1536]: the hash pools_test.cpp's bfp16_dit_pack gives."""
    from recipes import pack
    from test_pack_plan import _fnv1a
    from test_qwen35 import _shared_q8_vector
    q4 = pack.requant_q4_1(_shared_q8_vector(24))
    b = pack.pack_bfp16_dit(pack.f32_of_q4_1(q4, 128, 1536))
    assert b.size == 128 * 1536 * 9 // 8
    assert _fnv1a(b.tobytes()) == 0xF9C5BB308B534125


def test_per_tensor_ops_write_the_fused_weight_byte_for_byte():
    """Three bfp16_dit ops at their dst offsets equal dit_gemm's pack_b of the concatenated projection,
    which is why a fused q|k|v weight packs one tensor at a time."""
    from recipes import pack
    from test_qwen35 import _shared_q8_vector
    srcs = {n: pack.requant_q4_1(_shared_q8_vector(c)) for n, c in (("q", 48), ("k", 24), ("v", 24))}

    class M:
        def raw(self, name):
            return srcs[name].tobytes()

        def chunk_bytes_of(self, name):
            return pack.CH

    K, rows = 1536, {"q": 256, "k": 128, "v": 128}
    dst, n0 = np.zeros(512 * K * 9 // 8, np.uint8), 0
    for n in ("q", "k", "v"):
        pack.apply_op({"op": "bfp16_dit", "tensor": n, "nch": rows[n] // 32 * (K // 256), "in_dim": K,
                       "dst": n0 * K * 9 // 8}, M(), 0, dst)
        n0 += rows[n]
    w = np.concatenate([pack.f32_of_q4_1(srcs[n], rows[n], K) for n in ("q", "k", "v")])
    assert dst.tobytes() == pack.pack_bfp16_dit(w).tobytes()
