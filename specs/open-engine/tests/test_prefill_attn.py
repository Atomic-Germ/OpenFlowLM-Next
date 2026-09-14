# Traces: OPEN-PREFILL-ATTN (canonical spec: specs/open-engine/spec.md)
"""The block route's attention as two bf16 GEMMs per kv head on the NPU: what the recipe
writes for it, and the procedure that verifies the kernel and the route on hardware."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "open_kernels"))

LMAX = 4096
TIERS = list(range(256, LMAX + 1, 256))


def test_the_route_names_the_attention_gemm_streams():
    from recipes.load import default_spec
    from recipes.manifest import manifest
    from recipes.spec import FULL, LINEAR

    m = manifest(default_spec())
    ab = m["layer_types"][FULL]["gemm_block"]["attn_block"]
    # 8 query heads per kv head x 256 tokens = 2048 rows; one stream per 256 rows of window up to
    # LMAX, both products; the host chunks a longer window and merges the softmax
    assert ab["m"] == 2048 and ab["l_max"] == LMAX and ab["hd"] == 256
    assert ab["args"] == ["ag_a", "ag_b", "ag_c"]
    assert ab["kernels_s"] == {str(L): f"ag_s{L}" for L in TIERS}
    assert ab["kernels_pv"] == {str(L): f"ag_pv{L}" for L in TIERS}
    assert "attn_block" not in m["layer_types"][LINEAR]["gemm_block"]
    # every stream on one xclbin: the bf16 GEMM's core program takes M and K as runtime parameters
    assert m["contexts"]["ag"] == "ag_s256/final.xclbin"
    for L in TIERS:
        for tag, K, N in (("s", 256, L), ("pv", L, 256)):
            name = f"ag_{tag}{L}"
            assert m["kernels"][name] == {"context": "ag", "insts": f"{name}/insts.bin", "build": name}
            b = m["builds"][name]
            assert b["design"] == "attn_block/attn_gemm.py" and b["build_dir"] == f"attn_block/build_{tag}{L}"
            assert b["env"] == {"AG_M": "2048", "AG_K": str(K), "AG_N": str(N)}
    # a: Q or P rows [2048, K] bf16, b: the tiled K^T or V [K, N] bf16, c: [2048, N] f32, all for the widest
    assert (m["globals"]["ag_a"], m["globals"]["ag_b"], m["globals"]["ag_c"]) == (2048 * LMAX * 2, LMAX * 256 * 2, 2048 * LMAX * 4)


# ---- the kernel and the route: manual, on the NPU
# Verification (designs/attn_block, WSL ironenv142 for the builds, run_kernel.exe on Windows):
# 1. python make_test.py --L 2048; build build_s2048 (AG_K=256 AG_N=2048) and build_pv2048
#    (AG_K=2048 AG_N=256) with build_design.py; run_kernel.exe run_s2048.cfg && python compare.py
#    s2048, the same for pv2048: PASS at rel_fro <= 5e-3, per-row cosine printed; the dispatch
#    time is the rate (2.15 GFLOP each). The two final.xclbin differ only in the UUID bytes.
# 2. The full model: open_qwen36_cli --layers 8 --gemm-block --prefill-logits on the 19-token
#    prompt with and without OFLM_OPEN_ATTN_BLOCK=0 agree on argmax / top-5 per position (a prefix
#    that reaches a full-attention layer); the 1020-token prompt at 40 layers gives the same greedy
#    continuation, and the per-block `mid` time of the full-attention layers is recorded in spec.md.
# 3. oflm-test --llm through oflm serve with OFLM_OPEN_GEMM_BLOCK=1.
@pytest.mark.skip(reason="OPEN-PREFILL-ATTN hardware verification: see the procedure above")
def test_the_route_on_hardware():
    pass
