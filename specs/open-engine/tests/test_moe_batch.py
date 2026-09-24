# Traces: OPEN-MOE-BATCH (canonical spec: specs/open-engine/spec.md)
"""The routed experts are already in the band law the prefill GEMM reads.

The expert pools are packed by `expert_stripes` (up|gate, 128-row stripes) and
`expert_down`, neither of which is the 64-row band law `gemm_q4_prefill.py`'s
weight tap walks -- which is why the plan called for a stripe reader and a
repack. They are the same law: a 128-row stripe is two bands interleaved at
k-tile granularity, so a band is a STRIDED read of the pool as packed.

These tests derive the band's byte pattern from the packer's own permutations,
so they fail if either law is ever repermuted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "open_kernels"))

from recipes.pack import down_perm, std_perm, stripe_transpose  # noqa: E402

CH = 5120          # one q4_1 chunk: 32 rows x 256 K
BAND_BYTES = 10240  # gemm_q4_prefill.BAND_BYTES: 64 rows x 256 K = two chunks


def chunk_map(perm, in_dim: int) -> list[tuple[int, int]]:
    """pool chunk index -> (rowblock32, ktile), from a packer permutation."""
    ncol = in_dim // 256
    return [(int(f) // ncol, int(f) % ncol) for f in perm]


def test_a_stripe_is_two_bands_interleaved_at_k_tile_granularity():
    # one 128-row stripe of a [512, 2048] up / gate projection
    stripe = chunk_map(stripe_transpose(2048), 2048)
    law = chunk_map(std_perm(128, 2048), 2048)
    # a band is 16 chunks: rowblocks 2b and 2b+1 across the 8 k-tiles. Inside a 128-row
    # stripe that is band 0 (rowblocks 0, 1) and band 1 (rowblocks 2, 3).
    for band, first in ((0, 0), (1, 2)):
        want = [(rb + 2 * band, kt) for rb, kt in law[:16]]
        got = [stripe.index(w) for w in want]
        assert got == [first + 4 * j + h for j in range(8) for h in range(2)], (band, got)


def test_the_up_gate_band_tap_is_sizes_8_10240_strides_20480_1():
    stripe = chunk_map(stripe_transpose(2048), 2048)
    for band, offset in ((0, 0), (1, BAND_BYTES)):
        want = [(rb + 2 * band, kt) for rb, kt in chunk_map(std_perm(128, 2048), 2048)[:16]]
        byte_offsets = [stripe.index(w) * CH for w in want]
        # every consecutive PAIR of chunks is one 10240 B band element, and the elements
        # step by 20480 B -- exactly TensorAccessPattern(off, [8, 10240], [20480, 1])
        assert byte_offsets[0] == offset
        elems = byte_offsets[0::2]
        assert elems == [offset + 20480 * j for j in range(8)]
        assert byte_offsets[1::2] == [e + CH for e in elems], "a band element is two contiguous chunks"


def test_the_down_band_tap_has_the_same_shape_with_two_elements():
    # one expert's down [2048, 512]: 128 chunks, K = 512 -> 2 band elements per band
    dn = chunk_map(down_perm(128), 512)
    law = chunk_map(std_perm(128, 512), 512)
    for band in range(4):
        want = [(rb + 2 * band, kt) for rb, kt in law[:4]]
        byte_offsets = [dn.index(w) * CH for w in want]
        base = (band // 2) * 40960 + (band % 2) * BAND_BYTES
        assert byte_offsets[0::2] == [base, base + 20480], (band, byte_offsets)
        assert byte_offsets[1::2] == [base + CH, base + 20480 + CH]


@pytest.mark.parametrize("e", [0, 1, 255])
def test_an_experts_bands_land_inside_its_own_pool_region(e):
    """The offsets the kernel will emit, against the pack plan's own arithmetic."""
    stripe_bytes, stripes = 163840, 4
    up_gate_region = 2 * stripes * stripe_bytes          # per expert
    for band in range(8):                                 # 512 rows of up
        off = e * up_gate_region + 2 * (band // 2) * stripe_bytes + (band % 2) * BAND_BYTES
        last = off + 20480 * 7 + BAND_BYTES
        assert e * up_gate_region <= off < (e + 1) * up_gate_region
        assert last <= (e + 1) * up_gate_region
    down_bytes = 655360
    for band in range(32):                                # 2048 rows of down
        off = e * down_bytes + (band // 2) * 40960 + (band % 2) * BAND_BYTES
        assert e * down_bytes <= off < (e + 1) * down_bytes
        assert off + 20480 + BAND_BYTES <= (e + 1) * down_bytes


# ---- what the recipe writes for the kernel (the driver reads it from gemm_block.moe_batch)
def test_the_route_names_the_token_batched_expert_streams():
    from recipes.load import default_spec
    from recipes.manifest import manifest
    from recipes.spec import FULL, LINEAR

    m = manifest(default_spec())
    for lt in (LINEAR, FULL):
        mb = m["layer_types"][lt]["gemm_block"]["moe_batch"]
        # one stream per dispatch length, the shortest that holds the experts still owed tokens is
        # run, so the ladder is binary down from the expert count and the padding stays small
        # nt is the token slots one streaming of an expert serves -- the driver reads it here,
        # the build passes the same number to the design, and nothing else may decide it
        assert mb == {"kernels": {"256": "mb_s256", "128": "mb_s128", "64": "mb_s64", "32": "mb_s32",
                                  "16": "mb_s16", "8": "mb_s8"},
                      "args": ["pool", "mb_x", "mb_h", "mb_y"], "nt": 8}
    # every stream on one xclbin (the core program does not depend on the slot count)
    assert m["contexts"]["mb"] == "mb_s256/final.xclbin"
    for s in (256, 128, 64, 32, 16, 8):
        assert m["kernels"][f"mb_s{s}"] == {"context": "mb", "insts": f"mb_s{s}/insts.bin", "patch": "moebatch",
                                            "build": f"mb_s{s}"}
        b = m["builds"][f"mb_s{s}"]
        assert b["design"] == "moe_batch/moe_batch.py" and b["build_dir"] == f"moe_batch/build_s{s}"
        assert b["env"]["MB_SLOTS"] == str(s) and b["env"]["MB_EXPERTS"] == "256"
        assert b["env"]["MB_NT"] == str(mb["nt"])
        assert int(b["env"]["MB_POOL_DOWN"]) == m["layout"]["moe"]["pool_down"]
        assert int(b["env"]["MB_POOL_BYTES"]) == m["layout"]["pool_bytes"]
    # x / h / y sized for the longest stream: [256 slots, K, nt tokens] bf16 in, [256, 512, nt] bf16 h,
    # [256, 2048, nt] f32 out
    nt = m["layer_types"][LINEAR]["gemm_block"]["moe_batch"]["nt"]
    assert (m["globals"]["mb_x"], m["globals"]["mb_h"], m["globals"]["mb_y"]) == (256 * 2048 * nt * 2, 256 * 512 * nt * 2,
                                                                                 256 * 2048 * nt * 4)


# ---- the kernel itself: manual, on the NPU (the harness measurement is the artifact)
# Verification (designs/moe_batch, WSL ironenv for the builds, run_kernel.exe on Windows):
# 1. MB_SLOTS=64 MB_EXPERTS=64 python build_design.py designs/moe_batch/moe_batch.py designs/moe_batch/build_s64_e64
# 2. python make_test.py --slots 64  (64 random experts packed by recipes/pack.py's own stripe and
#    down laws, 8 distinct token activations per slot, the fp64 reference from the same bytes)
# 3. run_kernel.exe run_s64.cfg && python compare.py s64: PASS at rel_fro <= 5e-3 on y, every
#    slot's and every token column's cosine printed; the dispatch time is the stream rate
#    (64 experts x 2.23 MB per run).
# 4. The full model: open_qwen36_cli --layers 4 --gemm-block --prefill-logits with and without
#    OFLM_OPEN_MOE_BATCH=0 on the 19-token prompt agree on argmax / top-5 per position (the
#    family's near-tie exemption); the 1020-token prompt at 40 layers gives the same greedy
#    continuation and the per-block `moe run` time recorded in spec.md.
@pytest.mark.skip(reason="OPEN-MOE-BATCH hardware verification: see the procedure above")
def test_the_kernel_on_hardware():
    pass
