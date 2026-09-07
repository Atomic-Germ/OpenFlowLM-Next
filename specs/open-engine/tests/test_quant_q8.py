# Traces: OPEN-QUANT-Q8, OPEN-PACK-PLAN, OPEN-SPEC-DERIVE (canonical spec: specs/open-engine/spec.md)
"""q8 weight projections on the main cores: the 16-row half-tile split, the `q8_perm`
band law, the two packers' agreement, the per-role quant map derived from a container,
and the proof that a model whose every role is q4_1 does not move a byte.

The plan is specs/open-engine/plans/q8-gemv.md; these are its section 5.
"""
from __future__ import annotations

import dataclasses
import json
import struct

import numpy as np
import pytest

from recipes import pack
from recipes import qwen36moe as Q
from recipes import qwen35 as Q35
from recipes import dense as DN
from recipes.cache import build_key, source_files
from recipes.catalogue import OpRangeError
from recipes.load import HERE as RECIPES_DIR, default_spec, load_spec
from recipes.manifest import manifest
from recipes.spec import ModelSpec, QUANT_ROLES, quant_map_from_chunk_sizes, quant_map_from_gguf_types

from test_pack_plan import _fnv1a, _q8_vector

Q8 = pack.Q8
CH = pack.CH
SPECS = RECIPES_DIR / "specs"

# One [128, 512] q8 tensor: 8 file chunks of 32 rows x 256 K, 16 pool half-tiles.
OUT_DIM, IN_DIM = 128, 512
NCH = (OUT_DIM // 32) * (IN_DIM // 256)
NHALF = 2 * NCH
Q8_NAME = "model.layers.{l}.linear_attn.ssm_out_proj.weight"

# What recipes/pack.py packs the vector below to, through q8_perm (FNV-1a 64 of the pool
# bytes). src/open_qwen36/pools_test.cpp asserts the same number on the same bytes, so a
# divergence in either packer fails one of the two tests.
Q8_POOL_FNV1A = 0x8d4a3cf75e4cbffa


@pytest.fixture
def unvalidated(monkeypatch):
    """The q8 GEMV has no validated K yet, so a q8 recipe needs the override -- set and
    removed per test, never leaked into the ones that check a refusal."""
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def _bf16(u16) -> np.ndarray:
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def dq_q8_chunks(chunks) -> np.ndarray:
    """[n, 8704] container chunks -> [n, 32, 256] f32 (row, k)."""
    from q4nx import dq_chunks_q8
    return dq_chunks_q8(np.asarray(chunks, np.uint8).reshape(-1)).reshape(-1, 32, 256)


def dq_q8_half(tile) -> np.ndarray:
    """One 5120-byte pool half-tile -> [16, 256] f32, read exactly as gemv_q8.h reads it:
    scales[kb*16 + r] at [0:256], codes[k*16 + r] at [256:4352]."""
    b = np.asarray(tile, np.uint8)
    sc = _bf16(np.ascontiguousarray(b[:256]).view(np.uint16)).reshape(8, 16)     # [kb, r]
    codes = np.ascontiguousarray(b[256:4352]).view(np.int8).reshape(256, 16)     # [k, r]
    return (codes.astype(np.float32) * sc[np.arange(256) // 32]).T               # [r, k]


# ------------------------------------------------------------------ the half-tile split
def test_the_half_tile_split_round_trips_exactly():
    """dq(the two half-tiles) == dq(the container chunk), value for value. The split is a
    byte permutation -- no arithmetic -- so this is exact, not approximate."""
    src = _q8_vector(3)
    halves = pack.q8_half_tiles(src)
    assert halves.shape == (6, CH)
    want = dq_q8_chunks(src)
    for f in range(3):
        got = np.concatenate([dq_q8_half(halves[2 * f]), dq_q8_half(halves[2 * f + 1])], 0)
        assert np.array_equal(got, want[f])


def test_the_half_tile_is_a_byte_permutation_of_the_container_chunk():
    """The codes are copied verbatim (the container's row-block stride is exactly 4096
    codes), the scales are the chunk's 16 rows of each 32-block, and the tail is zero."""
    src = _q8_vector(2)
    halves = pack.q8_half_tiles(src)
    for f in range(2):
        for h in range(2):
            t = halves[2 * f + h]
            assert np.array_equal(t[256:4352], src[f, 512 + h * 4096:512 + (h + 1) * 4096])
            for kb in range(8):
                for r in range(16):
                    assert np.array_equal(t[2 * (kb * 16 + r):2 * (kb * 16 + r) + 2],
                                          src[f, 2 * (kb * 32 + 16 * h + r):2 * (kb * 32 + 16 * h + r) + 2])
            assert not t[4352:].any()


# ------------------------------------------------------------------ the band law
def test_q8_perm_matches_a_brute_force_placement():
    """The pool a q8 projection streams: half-tile c of the band holds output rows
    64*band + 16*(c%4) and columns 256*(c//4). Built by the law and by placing every
    tile where the kernel will read it -- against the dequantised source matrix."""
    src = _q8_vector(NCH)
    W = np.zeros((OUT_DIM, IN_DIM), np.float32)
    ncol = IN_DIM // 256
    dq = dq_q8_chunks(src)
    for f in range(NCH):
        W[32 * (f // ncol):32 * (f // ncol) + 32, 256 * (f % ncol):256 * (f % ncol) + 256] = dq[f]

    halves = pack.q8_half_tiles(src)
    files, hs = pack.q8_perm(NHALF, IN_DIM)
    pool = halves[2 * files + hs]
    per_band = IN_DIM // 64
    for c in range(NHALF):
        band, cc = c // per_band, c % per_band
        r0, k0 = 64 * band + 16 * (cc % 4), 256 * (cc // 4)
        assert np.array_equal(dq_q8_half(pool[c]), W[r0:r0 + 16, k0:k0 + 256]), c


def test_a_q8_band_is_twice_the_q4_1_bytes():
    for K in (2048, 4096, 12288):
        assert pack.q8_band_bytes(K) == 2 * Q.band_bytes(K)
        assert pack.q8_per_band(K) == 2 * (Q.band_bytes(K) // Q.CHUNK) == K // 64


# ------------------------------------------------------------------ the pack op
class _Container:
    def __init__(self, data, chunks):
        self.data, self.chunks = data, chunks

    def raw(self, name):
        return self.data[name]

    def chunk_bytes_of(self, name):
        return self.chunks[name]


def _q8_container():
    n = Q8_NAME.replace("{l}", "0")
    return _Container({n: _q8_vector(NCH).reshape(-1)}, {n: Q8})


def _q8_pool() -> np.ndarray:
    dst = np.zeros(NHALF * CH, np.uint8)
    pack.apply_op({"op": "q8_perm", "tensor": Q8_NAME, "dst": 0, "nch": NHALF, "in_dim": IN_DIM},
                  _q8_container(), 0, dst)
    return dst


def test_the_q8_perm_op_writes_the_law_and_doubles_the_bytes():
    got = _q8_pool().reshape(NHALF, CH)
    halves = pack.q8_half_tiles(_q8_vector(NCH))
    files, hs = pack.q8_perm(NHALF, IN_DIM)
    assert np.array_equal(got, halves[2 * files + hs])
    assert got.nbytes == 2 * NCH * CH


def test_the_numpy_and_cpp_packers_agree_on_the_q8_pool():
    """src/open_qwen36/pools_test.cpp builds the same q8 chunks and asserts this hash."""
    assert _fnv1a(_q8_pool()) == Q8_POOL_FNV1A


def test_a_q8_perm_over_a_container_that_is_not_q8_is_refused_by_name():
    n = Q8_NAME.replace("{l}", "0")
    m = _Container({n: np.zeros(NCH * CH, np.uint8)}, {n: CH})
    with pytest.raises(ValueError, match=r"ssm_out_proj\.weight.*5120"):
        pack.apply_op({"op": "q8_perm", "tensor": Q8_NAME, "dst": 0, "nch": NHALF, "in_dim": IN_DIM},
                      m, 0, np.zeros(NHALF * CH, np.uint8))


def test_a_q8_perm_without_nch_or_in_dim_is_refused():
    with pytest.raises(ValueError, match="q8_perm .* without nch / in_dim"):
        pack.apply_op({"op": "q8_perm", "tensor": Q8_NAME, "dst": 0}, _q8_container(), 0,
                      np.zeros(NHALF * CH, np.uint8))


# ------------------------------------------------------------------ the derived map
def _header(sizes: dict) -> dict:
    return {name: {"dtype": "I8", "shape": [1, ch], "data_offsets": [0, ch]} for name, ch in sizes.items()}


def test_the_quant_map_is_derived_from_a_containers_chunk_sizes():
    """The 35B fine-tunes: q8 attention / linear / out / shared projections, q4_1 routed
    experts. Roles, not tensor names -- the deriver maps the names once."""
    sizes = {}
    for l in (0, 1):
        for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            sizes[f"model.layer.{l}.self_attn.{t}.weight"] = Q8
        sizes[f"model.layer.{l}.linear_attn.qkv_proj.weight"] = Q8
        sizes[f"model.layer.{l}.linear_attn.ssm_out_proj.weight"] = Q8
        for t in ("up", "gate", "down"):
            sizes[f"model.layer.{l}.mlp.{t}_exps_proj.weight"] = CH
            sizes[f"model.layer.{l}.mlp.share_{t}_exps_proj.weight"] = Q8
    got = quant_map_from_chunk_sizes("qwen36moe", sizes)
    assert got == {"attn": "q8", "linear": "q8", "linear_out": "q8", "shared": "q8"}

    # the stock container: nothing but the head is q8, and the head is not a role
    stock = {k: CH for k in sizes}
    stock["lm_head.weight"] = Q8
    assert quant_map_from_chunk_sizes("qwen36moe", stock) == {}


def test_a_qwen35_container_puts_only_the_out_projection_at_q8():
    sizes = {"model.layers.0.linear_attn.qkv_proj.weight": CH,
             "model.layers.0.self_attn.gate_proj.weight": CH,
             "model.layers.0.linear_attn.ssm_out_proj.weight": Q8,
             "model.layers.0.self_attn.q_proj.weight": CH,
             "model.layers.0.mlp.up_proj.weight": CH}
    assert quant_map_from_chunk_sizes("qwen35", sizes) == {"linear_out": "q8"}


def test_a_role_whose_tensors_disagree_is_refused_by_name():
    sizes = {"model.layers.0.self_attn.q_proj.weight": Q8,
             "model.layers.1.self_attn.q_proj.weight": CH}
    with pytest.raises(ValueError, match=r"model.layers.1.self_attn.q_proj.weight"):
        quant_map_from_chunk_sizes("qwen35", sizes)


def test_the_gguf_tensor_types_derive_the_same_map():
    types = {"blk.0.attn_q.weight": "Q8_0", "blk.0.attn_k.weight": "Q8_0", "blk.0.attn_v.weight": "Q8_0",
             "blk.0.attn_output.weight": "Q8_0", "blk.0.ffn_up.weight": "Q4_1",
             "blk.0.ffn_gate.weight": "Q4_1", "blk.0.ffn_down.weight": "Q4_1"}
    assert quant_map_from_gguf_types("llama3", types) == {"attn": "q8"}
    assert quant_map_from_gguf_types("llama3", {k: "Q4_1" for k in types}) == {}


# ------------------------------------------------------------------ the spec surface
def test_an_all_q4_1_map_is_the_string_q4_1_and_moves_no_hash():
    ref = default_spec()
    same = dataclasses.replace(ref, quant={r: "q4_1" for r in QUANT_ROLES})
    assert same.to_dict()["quant"] == "q4_1"
    assert same.spec_hash() == ref.spec_hash()
    assert ModelSpec.from_json(same.to_json()) == ref


def test_a_q8_role_changes_the_spec_hash_and_survives_a_round_trip():
    ref = default_spec()
    q8 = dataclasses.replace(ref, quant={"attn": "q8"})
    assert q8.spec_hash() != ref.spec_hash()
    assert q8.to_dict()["quant"] == {"attn": "q8"}
    assert ModelSpec.from_json(q8.to_json()) == q8
    assert q8.quant_of("attn") == "q8" and q8.quant_of("experts") == "q4_1"
    assert q8.q8_roles == frozenset({"attn"})
    assert ref.q8_roles == frozenset()


def test_every_checked_in_spec_still_reads_as_all_q4_1():
    for p in sorted(SPECS.glob("*.json")):
        s = load_spec(p)
        assert s.quant == "q4_1" and s.q8_roles == frozenset(), p.name
        assert json.loads(p.read_text(encoding="utf-8"))["quant"] == "q4_1"


def test_the_shipped_27b_manifest_is_byte_identical():
    """The whole point of the normalisation: a model with no q8 role must produce exactly
    the manifest it produced before this requirement existed."""
    from pathlib import Path
    fx = Path(__file__).resolve().parent / "fixtures" / "manifest_qwen36.json"
    got = json.dumps(manifest(default_spec(), key="sha256:fixture"), indent=1) + "\n"
    assert got == fx.read_text(encoding="utf-8")


# ------------------------------------------------------------------ layout consequences
def test_a_q8_role_doubles_its_pool_region_and_the_27b_stays_at_512_MB():
    ref = default_spec()
    L0 = Q.layout(ref)
    assert L0.POOL_BYTES == 512 * (1 << 20)
    q8 = dataclasses.replace(ref, quant={"attn": "q8", "linear": "q8", "linear_out": "q8"})
    L1 = Q.layout(q8)
    # the linear projections: qkv then z, each twice the q4_1 bytes
    assert L1.POOL_Z - L1.POOL_QKV == 2 * (L0.POOL_Z - L0.POOL_QKV)
    assert L1.POOL_K - L1.POOL_Q == 2 * (L0.POOL_K - L0.POOL_Q)
    assert L1.POOL_BYTES > L0.POOL_BYTES and L1.POOL_BYTES % (1 << 20) == 0
    # the MoE's out-projection region was already twice the tensor, so consts do not move
    assert L1.C_WOUT == L0.C_WOUT and L1.C_BYTES == L0.C_BYTES


def test_a_q8_routed_expert_is_refused_by_name():
    spec = dataclasses.replace(default_spec(), quant={"experts": "q8"})
    with pytest.raises(OpRangeError, match="routed expert"):
        Q.recipe(spec)


def test_a_q8_shared_expert_is_refused_by_the_moe_recipe():
    """The shared expert rides the routed experts' call sites (`gemv_q4_gup` / `_gdown`,
    one nine-slot loop, for program memory). `spec_from_model_dir` downgrades the role
    rather than refusing the model; a hand-written spec that asks for it is refused."""
    spec = dataclasses.replace(default_spec(), quant={"shared": "q8"})
    with pytest.raises(OpRangeError, match="shared expert"):
        Q.recipe(spec)


def test_a_quant_the_gemv_cannot_read_is_still_refused():
    with pytest.raises(OpRangeError, match="quant='q4_k'"):
        Q.recipe(dataclasses.replace(default_spec(), quant="q4_k"))
    with pytest.raises(OpRangeError, match="quant='q4_k'"):
        Q.recipe(dataclasses.replace(default_spec(), quant={"attn": "q4_k"}))


def test_the_build_dir_names_only_gain_a_hash_when_a_role_is_q8():
    ref = default_spec()
    assert Q.builds(ref)["lx0"]["build_dir"] == "layer_x/build_lx0"
    q8 = dataclasses.replace(ref, quant={"attn": "q8"})
    d = Q.builds(q8)["lx0"]["build_dir"]
    assert d.startswith("layer_x/build_lx0_q") and d != "layer_x/build_lx0"
    assert Q.builds(q8)["ln"] == Q.builds(ref)["ln"]          # the norm does not read weights


def test_the_build_key_takes_a_role_map_and_does_not_move_for_an_all_q4_1_spec():
    """`build_key` hashed `spec.quant` as a string. A derived map is a dict, so every q8
    container raised `AttributeError: 'dict' object has no attribute 'encode'` before the
    canonical form went in -- and an all-q4_1 spec must still hash the bare string."""
    ref = default_spec()
    k = build_key(ref)
    assert build_key(dataclasses.replace(ref, quant={r: "q4_1" for r in QUANT_ROLES})) == k
    q8 = dataclasses.replace(ref, quant={"attn": "q8"})
    assert build_key(q8) != k                                   # a q8 role is a new kernel set
    assert build_key(q8) == build_key(dataclasses.replace(ref, quant={"attn": "q8"}))


def test_gemv_q8_h_enters_the_build_key_only_for_a_q8_spec():
    ref = default_spec()
    names = {f.name for f in source_files(ref)}
    assert "gemv_q8.h" not in names
    q8 = dataclasses.replace(ref, quant={"attn": "q8"})
    assert "gemv_q8.h" in {f.name for f in source_files(q8)}


def test_the_pack_plan_uses_q8_perm_exactly_where_the_role_is_q8(unvalidated):
    spec = dataclasses.replace(default_spec(), quant={"attn": "q8", "linear": "q8", "linear_out": "q8"})
    plan = Q.pack_plan(spec)
    lin = plan["layer_types"]["linear_attention"]
    ops = {o["tensor"]: o for o in lin["pool"] + lin["consts"] if "tensor" in o}
    assert ops["model.layer.{l}.linear_attn.qkv_proj.weight"]["op"] == "q8_perm"
    assert ops["model.layer.{l}.linear_attn.ssm_out_proj.weight"]["op"] == "q8_perm"
    assert ops["model.layer.{l}.mlp.share_up_exps_proj.weight"]["op"] == "std_perm"
    full = plan["layer_types"]["full_attention"]
    q = [o for o in full["pool"] if o.get("tensor", "").endswith("q_proj.weight")]
    assert all(o["op"] == "q8_perm" for o in q)
    assert q[1]["chunk0"] == Q.q4_chunks(spec.attn_q_width, spec.hidden)     # source FILE chunks
    assert q[0]["nch"] == 2 * Q.q4_chunks(spec.attn_q_width, spec.hidden)    # pool HALF chunks


def test_the_dense_recipes_take_the_same_switch(unvalidated):
    """No shipped dense model needs it, but the code path is not MoE-only."""
    ref = load_spec(SPECS / "llama31-8b.json")
    L0 = DN.layout(ref)
    q8 = dataclasses.replace(ref, quant={"attn": "q8"})
    L1 = DN.layout(q8)
    assert L1.POOL_K - L1.POOL_Q == 2 * (L0.POOL_K - L0.POOL_Q)
    assert L1.POOL_UP - L1.POOL_O == 2 * (L0.POOL_UP - L0.POOL_O)     # o is attention too
    assert L1.POOL_GATE - L1.POOL_UP == L0.POOL_GATE - L0.POOL_UP     # the FFN is not
    ops = {o["tensor"]: o["op"] for o in DN.pack_plan(q8)["layer_types"]["dense"]["pool"]}
    assert ops["model.layers.{l}.self_attn.q_proj.weight"] == "q8_perm"
    assert ops["model.layers.{l}.mlp.up_proj.weight"] == "std_perm"


def test_the_qwen35_recipe_takes_the_same_switch(unvalidated):
    ref = load_spec(SPECS / "qwen35-9b.json")
    q8 = dataclasses.replace(ref, quant={"linear_out": "q8"})
    L0, L1 = Q35.layout(ref), Q35.layout(q8)
    assert L1.C_BYTES - L1.C_WOUT == 2 * (L0.C_BYTES - L0.C_WOUT)
    ops = {o["tensor"]: o["op"] for o in Q35.pack_plan(q8)["layer_types"]["linear_attention"]["consts"]
           if "tensor" in o}
    assert ops["model.layers.{l}.linear_attn.ssm_out_proj.weight"] == "q8_perm"


def test_the_catalogue_refuses_an_unvalidated_gemv_q8_point(monkeypatch):
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    from recipes.catalogue import require
    # 2048 and 4096 entered the set with OPEN-QUANT-Q8's hardware pass; 3072 has not run.
    with pytest.raises(OpRangeError, match="gemv_q8: K=3072 is outside the validated set"):
        require("gemv_q8", K=3072, rs=4, rows_per_core=64, per_call=2)
    require("gemv_q8", K=2048, rs=4, rows_per_core=64, per_call=2)
    require("gemv_q8", K=4096, rs=4, rows_per_core=64, per_call=2)


# ------------------------------------------------------------------ the fp64 replica
def _write_container(path, tensors) -> str:
    """A minimal `.q4nx`: 8-byte header length, the safetensors JSON, then the data."""
    hdr, off, blobs = {}, 0, []
    for name, (ch, blob) in tensors.items():
        hdr[name] = {"dtype": "I8", "shape": [len(blob) // ch, ch], "data_offsets": [off, off + len(blob)]}
        off += len(blob)
        blobs.append(blob)
    j = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(j)))
        f.write(j)
        for b in blobs:
            f.write(b)
    return str(path)


def test_the_replica_reads_a_q8_projection_the_way_the_pool_holds_it(tmp_path):
    """The reference must read a projection the plan streams at q8 (`native_q8`) as the
    container's own q8, and every other q8 tensor as the q4_1 the packer writes -- or an
    acceptance number would measure the weights instead of the kernels."""
    from q4nx import Q4NX, dq_chunks_q4_1, dq_chunks_q8

    q8 = _q8_vector(NCH)
    streamed = "model.layers.0.linear_attn.ssm_out_proj.weight"
    requantised = "model.layers.0.mlp.share_down_exps_proj.weight"
    m = Q4NX(_write_container(tmp_path / "m.q4nx",
                              {streamed: (Q8, q8.reshape(-1).tobytes()),
                               requantised: (Q8, q8.reshape(-1).tobytes())}))
    m.native_q8 = lambda n: n == streamed
    assert m.requant_of(streamed) is False and m.requant_of(requantised) is True

    def raster(w):
        w = w.reshape(-1, 32, 256)
        ncol = IN_DIM // 256
        out = np.empty((OUT_DIM, IN_DIM), np.float32)
        for f in range(w.shape[0]):
            out[32 * (f // ncol):32 * (f // ncol) + 32, 256 * (f % ncol):256 * (f % ncol) + 256] = w[f]
        return out

    assert np.array_equal(m.matmul_w(streamed, OUT_DIM, IN_DIM), raster(dq_chunks_q8(q8.reshape(-1))))
    assert np.array_equal(m.matmul_w(requantised, OUT_DIM, IN_DIM),
                          raster(dq_chunks_q4_1(pack.requant_q4_1(q8))))
