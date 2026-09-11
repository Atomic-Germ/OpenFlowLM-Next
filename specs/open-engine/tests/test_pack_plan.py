# Traces: OPEN-PACK-PLAN (canonical spec: specs/open-engine/spec.md)
"""The plan-driven packer (recipes/pack.py over qwen36moe.pack_plan) reproduces
the hand-written packers byte for byte -- on a synthetic container whose
tensors have the 27B's shapes and random bytes (no model file needed)."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

import legacy_pools as LEG
from recipes import pack
from recipes import qwen36moe as Q
from recipes.load import default_spec
from recipes.spec import FULL, LINEAR

CH = 5120
# tensor name -> bytes, as the .q4nx container stores them (raster-order q4_1 chunks; bf16 / f32 small weights)
LAYER_TENSORS = {
    "mlp.up_exps_proj.weight": 256 * 4 * 163840, "mlp.gate_exps_proj.weight": 256 * 4 * 163840,
    "mlp.down_exps_proj.weight": 256 * 655360,
    "mlp.share_up_exps_proj.weight": 655360, "mlp.share_gate_exps_proj.weight": 655360,
    "mlp.share_down_exps_proj.weight": 655360,
    "input_layernorm.weight": 4096, "post_attention_layernorm.weight": 4096,
    "shared_expert_gate.weight": 4096, "moe_router.weight": 1048576,
}
LINEAR_TENSORS = {
    "linear_attn.qkv_proj.weight": 2048 * CH, "self_attn.gate_proj.weight": 1024 * CH,
    "linear_attn.ssm_conv1d.weight": 65536, "linear_attn.ssm_norm.weight": 256,
    "linear_attn.ssm_a": 128, "linear_attn.ssm_dt.bias": 128,
    "linear_attn.ssm_alpha_proj.weight": 131072, "linear_attn.ssm_beta_proj.weight": 131072,
    "linear_attn.ssm_out_proj.weight": 1024 * CH,
}
FULL_TENSORS = {
    "self_attn.q_proj.weight": 2048 * CH, "self_attn.k_proj.weight": 128 * CH, "self_attn.v_proj.weight": 128 * CH,
    "self_attn.o_proj.weight": 1024 * CH, "self_attn.q_norm.weight": 512, "self_attn.k_norm.weight": 512,
}


class FakeContainer:
    """raw(name) -> random bytes of the tensor's size, deterministic per name."""

    def __init__(self, layers: dict[int, str], lm_head_chunks: int = 64):
        self.sizes = {"lm_head.weight": lm_head_chunks * 8704, "model.norm.weight": 4096}
        for l, kind in layers.items():
            for k, v in {**LAYER_TENSORS, **(FULL_TENSORS if kind == FULL else LINEAR_TENSORS)}.items():
                self.sizes[f"model.layer.{l}.{k}"] = v
        self.cache: dict[str, np.ndarray] = {}

    def raw(self, name: str):
        if name not in self.cache:
            seed = abs(hash(name)) % (2 ** 32)
            self.cache[name] = np.random.default_rng(seed).integers(0, 256, self.sizes[name], dtype=np.uint8)
        return self.cache[name]


@pytest.fixture(scope="module")
def m():
    return FakeContainer({0: LINEAR, 1: FULL})


@pytest.fixture(scope="module")
def plan():
    return Q.pack_plan(default_spec())


@pytest.mark.parametrize("layer,kind", [(0, LINEAR), (1, FULL)])
def test_layer_pool_matches_the_legacy_packer(m, plan, layer, kind):
    ours = pack.build_layer_pool(plan, kind, m, layer)
    ref = LEG.build_layer_pool(m, layer, kind == FULL)
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref)


@pytest.mark.parametrize("layer,kind", [(0, LINEAR), (1, FULL)])
def test_consts_match_the_legacy_layer_consts(m, plan, layer, kind):
    L = Q.layout(default_spec())
    nbytes = L.CA_BYTES if kind == FULL else L.C_BYTES
    ours = pack.build_consts(plan, kind, m, layer, nbytes)
    ref = LEG.layer_consts(m, layer, kind == FULL)
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref)


def test_lm_head_pool_matches(m, plan):
    ours = pack.build_lmhead_pool(plan, m)
    ref = LEG.build_lmhead_pool(m)
    assert ours.shape == ref.shape and np.array_equal(ours, ref)


def test_ptab_matches():
    spec = default_spec()
    assert np.array_equal(pack.ptab(300, spec.rotary_dim, spec.rope_theta), LEG.ptab(300))


def test_a_small_weight_that_overflows_its_slot_is_refused(plan):
    class Big(FakeContainer):
        def raw(self, name):
            if name.endswith("input_layernorm.weight"):
                return np.zeros(4097, np.uint8)
            return super().raw(name)

    with pytest.raises(ValueError, match="does not fit its 4096 B slot"):
        pack.build_consts(plan, LINEAR, Big({0: LINEAR}), 0, Q.layout(default_spec()).C_BYTES)


# --------------------------------------------------------------- q8 sources
# Traces: OPEN-PACK-PLAN (the q8-source rule). The same synthetic container is built,
# packed and hashed by src/open_qwen36/pools_test.cpp, so the two packers are held to
# each other byte for byte without a model file.

Q8 = 8704
OUT_DIM, IN_DIM = 128, 512
NCH = (OUT_DIM // 32) * (IN_DIM // 256)        # 8 chunks either format: 32 rows x 256 K each
Q8_NAME = "model.layers.{l}.linear_attn.ssm_out_proj.weight"
Q4_NAME = "model.layers.{l}.mlp.down_proj.weight"
BAD_NAME = "model.layers.{l}.mlp.up_proj.weight"


def _q8_vector(nch: int) -> np.ndarray:
    """The vector pools_test.cpp builds, byte for byte: per chunk one LCG seeded from the
    chunk index, 256 bf16 scales with exponent 0x76 (finite and positive, so the
    requantizer sees real ranges) then 8192 int8 codes."""
    M = 0xFFFFFFFF
    out = np.zeros((nch, Q8), np.uint8)
    for c in range(nch):
        s = (0x9E3779B9 * (c + 1)) & M
        sc = np.zeros(256, np.uint16)
        for i in range(256):
            s = (s * 1664525 + 1013904223) & M
            sc[i] = 0x3B00 | (s >> 24)
        out[c, :512] = sc.view(np.uint8)
        for i in range(512, Q8):
            s = (s * 1664525 + 1013904223) & M
            out[c, i] = s >> 24
    return out


def _lcg_bytes(seed: int, n: int) -> np.ndarray:
    """pools_test.cpp's plain byte stream, for the q4_1 tensor (any bytes will do: the
    q4_1 path is a pure chunk copy and never looks inside)."""
    M = 0xFFFFFFFF
    out = np.zeros(n, np.uint8)
    s = seed & M
    for i in range(n):
        s = (s * 1664525 + 1013904223) & M
        out[i] = s >> 24
    return out


class MixedContainer:
    """A container holding one q8 tensor, one q4_1 tensor and one with 1280-byte chunks --
    the mix the Qwen3.6-35B fine-tunes ship (q8 projections, q4_1 routed experts)."""

    def __init__(self):
        self.data = {
            Q8_NAME.replace("{l}", "0"): _q8_vector(NCH).reshape(-1),
            Q4_NAME.replace("{l}", "0"): _lcg_bytes(0x1234567, NCH * pack.CH),
            BAD_NAME.replace("{l}", "0"): _lcg_bytes(0x89ABCDE, NCH * 1280),
        }
        self.chunks = {Q8_NAME.replace("{l}", "0"): Q8,
                       Q4_NAME.replace("{l}", "0"): pack.CH,
                       BAD_NAME.replace("{l}", "0"): 1280}

    def raw(self, name):
        return self.data[name]

    def chunk_bytes_of(self, name):
        return self.chunks[name]


def _mixed_pool() -> np.ndarray:
    m = MixedContainer()
    dst = np.zeros(2 * NCH * pack.CH, np.uint8)
    pack.apply_op({"op": "std_perm", "tensor": Q8_NAME, "dst": 0, "nch": NCH, "in_dim": IN_DIM}, m, 0, dst)
    pack.apply_op({"op": "std_perm", "tensor": Q4_NAME, "dst": NCH * pack.CH, "nch": NCH, "in_dim": IN_DIM},
                  m, 0, dst)
    return dst


def _fnv1a(b) -> int:
    h = 1469598103934665603
    for x in bytes(b):
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def test_a_q8_source_packs_as_the_requantized_q4_1_in_the_same_chunk_order():
    """(b) The q8 half of the pool is exactly requant_q4_1 of the q8 chunks, put through
    the SAME std_perm the q4_1 path uses -- a q8 chunk and a q4_1 chunk hold the same
    32 x 256 tile, so no index law changes."""
    got = _mixed_pool()[:NCH * pack.CH].reshape(NCH, pack.CH)
    want = pack.requant_q4_1(_q8_vector(NCH))[pack.std_perm(NCH, IN_DIM)]
    assert np.array_equal(got, want)


def test_a_q4_1_source_beside_a_q8_one_is_still_a_verbatim_chunk_copy():
    """(c) The q4_1 half is byte for byte what the packer produced before q8 sources
    existed: the chunks copied in std_perm order, untouched."""
    m = MixedContainer()
    got = _mixed_pool()[NCH * pack.CH:].reshape(NCH, pack.CH)
    src = m.raw(Q4_NAME.replace("{l}", "0")).reshape(NCH, pack.CH)
    assert np.array_equal(got, src[pack.std_perm(NCH, IN_DIM)])


def test_a_container_that_cannot_report_chunk_sizes_is_read_as_q4_1():
    """The frozen pools of the rest of this file come from a container with no
    `chunk_bytes_of`; that path must not change."""
    class Plain:
        def __init__(self, d):
            self.d = d

        def raw(self, name):
            return self.d[name]

    raw = _lcg_bytes(0x1234567, NCH * pack.CH)
    dst = np.zeros(NCH * pack.CH, np.uint8)
    pack.apply_op({"op": "std_perm", "tensor": Q4_NAME, "dst": 0, "nch": NCH, "in_dim": IN_DIM},
                  Plain({Q4_NAME.replace("{l}", "0"): raw}), 0, dst)
    assert np.array_equal(dst.reshape(NCH, pack.CH), raw.reshape(NCH, pack.CH)[pack.std_perm(NCH, IN_DIM)])


def test_the_numpy_and_cpp_packers_agree_on_the_mixed_container():
    """(a) The gate on the two implementations. src/open_qwen36/pools_test.cpp builds the
    same three tensors into a real .q4nx, packs them through pools::apply and asserts this
    same FNV-1a; if either packer changes, one of the two tests fails."""
    assert _fnv1a(_mixed_pool().tobytes()) == 0x548390807A90D2EB, \
        "the mixed-container pool changed; update src/open_qwen36/pools_test.cpp too"


def test_a_chunk_size_none_of_the_three_the_packer_reads_is_refused_by_name():
    """The message names the tensor, the byte count and what the count probably is. It must
    not GUESS Q4_K for a width that is not 4736 -- the reason this assertion exists (the
    message used to read "OFLM 1.0.3 / Q4_K?" for anything unfamiliar). Q4_K appears only in
    the list of widths the packer does read (OPEN-QUANT-Q4K)."""
    m = MixedContainer()
    dst = np.zeros(NCH * pack.CH, np.uint8)
    with pytest.raises(ValueError) as e:
        pack.apply_op({"op": "std_perm", "tensor": BAD_NAME, "dst": 0, "nch": NCH, "in_dim": IN_DIM},
                      m, 0, dst)
    msg = str(e.value)
    assert "mlp.up_proj.weight" in msg and "1280" in msg
    assert "1280 is a smaller chunk geometry" in msg


def _write_container(path, tensors):
    """A minimal `.q4nx` (safetensors) holding the given {name: (chunk_bytes, bytes)}."""
    hdr, data, off = {}, bytearray(), 0
    for name, (ch, b) in tensors.items():
        hdr[name] = {"dtype": "I8", "shape": [len(b) // ch, ch], "data_offsets": [off, off + len(b)]}
        data += bytes(b)
        off += len(b)
    blob = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(data))
    return path


def test_the_python_reader_reads_a_q8_tensor_both_ways(tmp_path):
    """The fp64 replica must read a q8 projection: as the q4_1 the packer writes (the
    acceptance reference, the default) and as the container's own q8 (the quality one)."""
    from q4nx import Q4NX, dq_chunks_q4_1, dq_chunks_q8

    name = "model.layers.0.linear_attn.ssm_out_proj.weight"
    q8 = _q8_vector(NCH)
    m = Q4NX(_write_container(tmp_path / "m.q4nx", {name: (Q8, q8.reshape(-1).tobytes())}))
    assert m.chunk_bytes_of(name) == Q8 and m.chunk_bytes == pack.CH

    def raster(vals):
        w = vals.reshape(-1, 32, 256)
        ncol = IN_DIM // 256
        W = np.empty((OUT_DIM, IN_DIM), np.float32)
        for f in range(w.shape[0]):
            W[32 * (f // ncol):32 * (f // ncol) + 32, 256 * (f % ncol):256 * (f % ncol) + 256] = w[f]
        return W

    assert np.array_equal(m.matmul_w(name, OUT_DIM, IN_DIM),
                          raster(dq_chunks_q4_1(pack.requant_q4_1(q8))))
    m.requant_q8 = False
    assert np.array_equal(m.matmul_w(name, OUT_DIM, IN_DIM), raster(dq_chunks_q8(q8.reshape(-1))))


def test_the_python_reader_refuses_a_chunk_size_it_does_not_know(tmp_path):
    from q4nx import Q4NX

    name = "model.layers.0.mlp.up_proj.weight"
    m = Q4NX(_write_container(tmp_path / "bad.q4nx", {name: (1280, _lcg_bytes(1, NCH * 1280).tobytes())}))
    with pytest.raises(RuntimeError) as e:
        m.matmul_w(name, OUT_DIM, IN_DIM)
    assert "mlp.up_proj.weight" in str(e.value) and "1280" in str(e.value)


# ---- lmhead_q8: the 128-row supertile order is a function of K, not the 27B's 2048
# The 4B slice (K = 2560) came back with correct residuals in all 8 layers and garbage
# logits because the permutation hardcoded 8 k-tiles per band (K = 2048).

class _LmHeadContainer:
    """One `lm_head.weight` of q8 chunks whose first 8 bytes are the file chunk index."""
    def __init__(self, nch: int):
        raw = np.zeros((nch, pack.Q8), np.uint8)
        raw[:, :8] = np.frombuffer(np.arange(nch, dtype="<u8").tobytes(), np.uint8).reshape(nch, 8)
        self._raw = raw.reshape(-1)

    def raw(self, name):
        assert name == "lm_head.weight"
        return self._raw

    def chunk_bytes_of(self, name):
        return pack.Q8


def _lmhead_pool(k: int, rows: int):
    nk = k // 256
    nch = (rows // 32) * nk
    m = _LmHeadContainer(nch)
    dst = np.zeros(nch * pack.Q8, np.uint8)
    pack.apply_op({"op": "lmhead_q8", "tensor": "lm_head.weight", "chunk_bytes": pack.Q8,
                   "in_dim": k, "dst": 0}, m, 0, dst)
    return dst.reshape(nch, pack.Q8)[:, :8].copy().view("<u8").reshape(-1)


@pytest.mark.parametrize("k", [2048, 2560, 4096])
def test_the_lmhead_supertile_order_follows_k(k):
    """Pool chunk c of band b is the file chunk holding rows [128b + 32*(c % 4), +32) and
    K [(c // 4) * 256, +256); the file lays those out as rowblock32 * (K/256) + ktile."""
    rows = 128 * 6
    got = _lmhead_pool(k, rows)
    nk = k // 256
    per_band = 4 * nk
    want = np.array([(4 * (i // per_band) + (i % per_band) % 4) * nk + (i % per_band) // 4
                     for i in range(len(got))], "<u8")
    assert np.array_equal(got, want)


def test_the_lmhead_order_at_k_2048_is_the_shipped_one():
    """The 27B's law, written out: k -> (4*(k//32) + k%4)*8 + (k%32)//4."""
    got = _lmhead_pool(2048, 128 * 6)
    want = np.array([(4 * (i // 32) + i % 4) * 8 + (i % 32) // 4 for i in range(len(got))], "<u8")
    assert np.array_equal(got, want)


def test_lmhead_q8_without_in_dim_is_refused():
    m = _LmHeadContainer(32)
    dst = np.zeros(32 * pack.Q8, np.uint8)
    with pytest.raises(ValueError, match="lmhead_q8 lm_head.weight without in_dim"):
        pack.apply_op({"op": "lmhead_q8", "tensor": "lm_head.weight", "chunk_bytes": pack.Q8, "dst": 0},
                      m, 0, dst)
