# Traces: OPEN-HADAMARD, OPEN-QUANT-T2, OPEN-CONVERT-PRISM-TERNARY (canonical spec: specs/open-engine/spec.md)
"""Ternary Bonsai 2 27B on the open kernels: the rotated-basis spec field, the t2 weight
format's recipe geometry and pack plan, the generated sign table, and the 2-bit packing
against PrismML's own decode rule.

The config is the Qwen3.8-27B fixture (#148) plus the `prism_hadamard` block q4nx-build's
Bonsai path writes; the og signs are a fixed pseudo-random vector (the model's own are in its
GGUF, which is not checked in).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from recipes import qwen35 as Q35, qwen36moe as Q36
from recipes.catalogue import OpRangeError
from recipes.load import spec_from_model_dir
from recipes.spec import ModelSpec, SpecError

FIX = Path(__file__).resolve().parent / "fixtures"
OK = Path(__file__).resolve().parents[3] / "open_kernels"


def _signs(n: int = 6144, seed: int = 7) -> list[int]:
    r = random.Random(seed)
    return [r.choice((1, -1)) for _ in range(n)]


def _cfg(signs=None, head=None) -> dict:
    cfg = json.loads((FIX / "config_qwen35_27b.json").read_text(encoding="utf-8"))
    cfg["prism_hadamard"] = {"block_size": 1024, "og_signs": signs or _signs()}
    if head is not None:
        cfg["prism_hadamard"]["lm_head"] = head
    return cfg


def _spec(quant: str = "q4_1") -> ModelSpec:
    return dataclasses.replace(ModelSpec.from_hf_config(_cfg()), quant=quant)


def test_hadamard_field_is_read_and_hashed_only_when_present():
    plain = ModelSpec.from_hf_config(json.loads((FIX / "config_qwen35_27b.json").read_text(encoding="utf-8")))
    rot = ModelSpec.from_hf_config(_cfg())
    assert plain.hadamard is None and "hadamard" not in plain.to_dict()
    assert plain.quant_hash() == ""
    assert rot.hadamard == {"block": 1024, "og_signs": _signs()}
    assert rot.spec_hash() != plain.spec_hash()
    assert rot.quant_hash() not in ("", plain.quant_hash())
    # a different sign vector is a different kernel set
    other = ModelSpec.from_hf_config(_cfg(_signs(seed=8)))
    assert other.quant_hash() != rot.quant_hash()
    assert ModelSpec.from_dict(rot.to_dict()) == rot


@pytest.mark.parametrize("bad", [{"block_size": 512, "og_signs": _signs()},
                                 {"block_size": 1024, "og_signs": _signs(6143)},
                                 {"block_size": 1024, "og_signs": [2] * 6144}])
def test_hadamard_refuses_what_the_kernels_do_not_implement(bad):
    cfg = _cfg()
    cfg["prism_hadamard"] = bad
    with pytest.raises(SpecError):
        ModelSpec.from_hf_config(cfg)


def test_ternary_format_switch(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(_cfg()), encoding="utf-8")
    monkeypatch.delenv("OFLM_TERNARY_FORMAT", raising=False)
    assert spec_from_model_dir(tmp_path).quant == "t2"
    monkeypatch.setenv("OFLM_TERNARY_FORMAT", "q4_1")
    assert spec_from_model_dir(tmp_path).quant == "q4_1"


def test_t2_geometry_unpadded_chunk_per_element_and_state_slices():
    rq4, rt2 = Q35.recipe(_spec("q4_1")), Q35.recipe(_spec("t2"))
    q4, t2 = rq4.common, rt2.common
    # one unpadded 2176 B chunk per w element: 2048 B of codes + 64 bf16 scales, nothing else
    assert Q36.T2_CHUNK == 2048 + 64 * 2
    assert (t2.TILE, t2.PER_CALL, t2.CALL_BYTES) == (Q36.T2_CHUNK, 1, Q36.T2_CHUNK)
    # 2176 B is 4.25 state rows: the slices are 4 rows at a 2048 B stride, and a head is
    # exactly its 128 rows (no pad rows); the state buffer ends in the 128 B the last head's
    # last element reads past its slice
    assert (t2.DN_ROWS, t2.DN_SLICES, t2.DN_PAD) == (4, 32, 128)
    assert Q36.dn_slice_bytes(t2) == 2048 and Q36.dn_state_tail(t2) == 128
    L = rt2.layout
    assert (L.S_ROWS, L.S_HEAD_BYTES) == (128, 128 * 128 * 4)
    heads = rt2.spec.lin_value_heads
    last_read = L.STATE_S_OFF + (heads - 1) * L.S_HEAD_BYTES + (t2.DN_SLICES - 1) * 2048 + t2.CALL_BYTES
    assert L.STATE_BYTES == L.STATE_S_OFF + heads * L.S_HEAD_BYTES + 128 == last_read
    # q4_1 keeps its geometry: slices are whole elements, 130 rows, no tail
    assert (q4.DN_ROWS, q4.DN_SLICES, q4.DN_PAD) == (10, 13, 130)
    assert Q36.dn_slice_bytes(q4) == q4.CALL_BYTES and Q36.dn_state_tail(q4) == 0
    assert rq4.layout.STATE_BYTES == rq4.layout.STATE_S_OFF + heads * rq4.layout.S_HEAD_BYTES
    assert Q36.xh_l1(_spec()) == 1024 * 4 + 6144 // 8


def test_t2_must_be_every_projection_and_rotated():
    with pytest.raises(OpRangeError):
        Q35.recipe(_spec({"ffn": "t2"}))
    unrotated = dataclasses.replace(_spec("t2"), hadamard=None)
    with pytest.raises(OpRangeError):
        Q35.recipe(unrotated)


def test_t2_pack_plan_and_gemm_weights():
    spec = _spec("t2")
    plan = Q35.pack_plan(spec)["layer_types"]
    projs = [o for lt in plan.values() for k in ("pool", "consts") for o in lt[k] if "nch" in o]
    assert projs and all(o["op"] == "t2_perm" for o in projs)
    # the engine packs at the stride the kernels stream (an op without one means the old 2560)
    assert all(o["chunk_bytes"] == Q36.T2_CHUNK for o in projs)
    route = Q35.gemm_route(spec)
    for g in route["layer_types"].values():
        for key in ("weights", "ffn_weights"):
            for w in g.get(key, {}).values():
                # OPEN-GEMM-T2: the GEMM reads 2-bit chunks, packed back to back in its own buffer
                assert w["from"] == "pack" and all(o["op"] == "t2_perm" for o in w["pack"])
                dst = 0
                for o in w["pack"]:                 # one contiguous run, in the order the GEMM reads
                    assert o["dst"] == dst
                    dst += o["nch"] * Q36.T2_CHUNK


# Traces: OPEN-GEMM-T2
def test_t2_gemm_builds_and_token_major_output():
    plain = ModelSpec.from_hf_config(json.loads((FIX / "config_qwen35_27b.json").read_text(encoding="utf-8")))
    t2_env = {"GQP_BFP16": "1", "GQP_WFMT": "t2", "GQP_KT": "128", "GQP_YT": "1",
              "GQP_T2_STRIDE": str(Q36.T2_CHUNK)}
    for spec, t2 in ((_spec("t2"), True), (_spec("q4_1"), False), (plain, False)):
        gemms = {k: b for k, b in Q35.builds(spec).items() if k.startswith("gemm_")}
        assert gemms, "the 27B has a block route"
        for b in gemms.values():
            extra = {k: v for k, v in b["env"].items() if k not in ("GQP_N", "GQP_K", "GQP_T")}
            assert extra == (t2_env if t2 else {})      # a q4_1 spec's build is the one it always was
            assert b["build_dir"].endswith("_t2ky") is t2
        for g in Q35.gemm_route(spec)["layer_types"].values():
            assert g.get("y_tn", False) is t2
            assert g.get("x_tile_k", 64) == (128 if t2 else 64)


def _gen_kernels():
    p = OK / "designs" / "layer_x" / "gen_kernels.py"
    s = importlib.util.spec_from_file_location("gen_kernels_t", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_generated_sign_table_and_tus():
    gk = _gen_kernels()
    plain = gk.files(Q35.recipe(dataclasses.replace(_spec(), hadamard=None)))
    q4h = gk.files(Q35.recipe(_spec("q4_1")))
    t2 = gk.files(Q35.recipe(_spec("t2")))
    assert "xh_signs.h" not in plain and "#include \"wht.h\"" not in plain["dense_prep.cc"]
    assert set(q4h) - set(plain) == {"xh_signs.h"}
    assert {"gemv_t2_gy.cc", "gemv_t2_gms.cc"} <= set(t2) and "gemv_q4_gy.cc" not in t2
    words = [int(w, 16) for w in __import__("re").findall(r"0x([0-9a-f]{8})u", q4h["xh_signs.h"])]
    bits = [(words[j // 32] >> (j % 32)) & 1 for j in range(6144)]
    assert bits == [1 if v < 0 else 0 for v in _signs()]


def test_t2_chunks_reproduce_prismml_decode():
    sys.path.insert(0, str(OK))
    from q4_1_pack import dequant_pool, pack_q4_1_pool
    from t2_pack import (as_q4_1_blocks, decode_pq2, dequant, dequant_t2_pool, encode_pq2,
                         pack_t2_pool, random_pq2_codes)
    from ml_dtypes import bfloat16
    rng = np.random.default_rng(0)
    codes, s = random_pq2_codes(256, 1024, rng)
    c2, s2 = decode_pq2(encode_pq2(codes, s))
    assert np.array_equal(c2, codes) and np.array_equal(s2.view(np.uint16), s.view(np.uint16))
    # PQ2_0's rule, value = s * code - s, in fp32 from the GGUF scales (PrismML runtime/codec.py)
    ref = (codes.astype(np.float32) - 1.0) * np.repeat(s.astype(np.float32), 128, axis=1)
    np.testing.assert_array_equal(dequant(codes, s), ref)
    want = dequant(codes, s, bfloat16)                     # what both pools hold: bf16 scales
    assert np.array_equal(dequant_t2_pool(pack_t2_pool(codes, s, 2), 256, 1024, 2), want)
    assert np.array_equal(dequant_pool(pack_q4_1_pool(as_q4_1_blocks(codes, s), 2), 256, 1024, 2), want)


# ---- the rotated ternary lm head (OPEN-QUANT-T2's head, OPEN-CONVERT-PRISM-TERNARY's head)

def test_rotated_head_is_a_spec_field_that_moves_no_layer_build():
    unrot = ModelSpec.from_hf_config(_cfg(head="unrotated"))
    rot = ModelSpec.from_hf_config(_cfg(head="rotated"))
    assert unrot == ModelSpec.from_hf_config(_cfg())            # absent reads as "unrotated"
    assert "lm_head" not in unrot.hadamard and rot.hadamard["lm_head"] == "rotated"
    assert rot.spec_hash() != unrot.spec_hash()                 # a different kernel set...
    assert rot.quant_hash() == unrot.quant_hash()               # ...sharing the lx / ax build dirs
    assert ModelSpec.from_dict(rot.to_dict()) == rot
    with pytest.raises(SpecError):
        ModelSpec.from_hf_config(_cfg(head="q8"))


@pytest.mark.parametrize("quant", ["t2", "q4_1"])
def test_rotated_head_runs_as_t2_whatever_the_layers_stream(quant):
    spec = dataclasses.replace(ModelSpec.from_hf_config(_cfg(head="rotated")), quant=quant)
    Q35.recipe(spec)
    head = Q35.pack_plan(spec)["lm_head"]
    nch = spec.vocab // 32 * (spec.hidden // 256)
    assert head["ops"] == [{"op": "t2_perm", "tensor": "lm_head.weight", "dst": 0, "nch": nch,
                            "in_dim": spec.hidden, "chunk_bytes": Q35.LM_T2_CHUNK}]
    # unpadded: exactly PrismML's own PQ2_0 bytes for the head (34 B per 128 weights)
    assert Q35.LM_T2_CHUNK == 2176 and head["pool_bytes"] == nch * 2176 == spec.vocab * spec.hidden // 128 * 34
    prog = Q35.programs(spec)
    assert prog["contexts"]["lm"] == "lm_head_t2/final.xclbin" and prog["kernels"]["lm"]["build"] == "lm_head_t2"
    assert prog["tail"][-1] == {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}
    lay = Q35.manifest_layout(spec, 4096)
    assert prog["globals"]["lmpool"] == lay["lmhead_pool_bytes"] == head["pool_bytes"]
    assert lay["lmhead_chunk_bytes"] == 2176
    b = Q35.builds(spec)
    assert "lm_head_q8" not in b and b["lm_head_t2"]["env"] == {
        "LMHEAD_N": str(spec.vocab), "LMHEAD_K": str(spec.hidden), "LMHEAD_CORES": "8"}


def test_unrotated_head_keeps_the_q8_head():
    spec = _spec("t2")
    assert [o["op"] for o in Q35.pack_plan(spec)["lm_head"]["ops"]] == ["lmhead_q8"]
    assert Q35.programs(spec)["contexts"]["lm"] == "lm_head_q8/final.xclbin"
    assert "lm_head_q8" in Q35.builds(spec) and "lm_head_t2" not in Q35.builds(spec)


def _prism():
    """q4nx-build's q4nx/prism.py, by path: open_kernels/model/q4nx.py owns the name `q4nx` here."""
    path = OK.parent / "utilities" / "q4nx-build" / "q4nx" / "prism.py"
    s = importlib.util.spec_from_file_location("q4nx_build_prism", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_converter_head_bands_are_the_ternary_weights_in_the_rotated_basis():
    """q4nx-build writes output.weight as q4_1 row bands with q = code, d = s, m = -s, untouched
    by any sign or transform (the signs go to output_norm), so the engine can re-derive the 2-bit
    codes exactly."""
    P = _prism()
    sys.path.insert(0, str(OK))
    from t2_pack import encode_pq2, random_pq2_codes
    from types import SimpleNamespace
    rows, cols = 96, 1024
    codes, s = random_pq2_codes(rows, cols, np.random.default_rng(3))
    t = SimpleNamespace(name="output.weight", shape=[cols, rows], data=encode_pq2(codes, s).reshape(-1))
    rot = object.__new__(P.PrismRotation)
    rot.hidden = cols
    bands = list(rot.lm_head_q4_1_bands(t, rows_per_pass=64))
    assert [b[2].shape[0] for b in bands] == [64, 32]           # whole 32-row blocks, ragged last band
    d, m, q = (np.concatenate([b[i].numpy() for b in bands]) for i in range(3))
    assert np.array_equal(q, codes.astype(np.float32)) and np.array_equal(m, -d)
    assert np.array_equal(d, np.repeat(s.astype(np.float32), 4, axis=1))   # one scale per 128, fp16 exact
    w = np.repeat(d, 32, axis=1) * q + np.repeat(m, 32, axis=1)
    assert np.array_equal(w, (codes.astype(np.float32) - 1) * np.repeat(s.astype(np.float32), 128, axis=1))
    with pytest.raises(ValueError):
        next(rot.lm_head_q4_1_bands(t, rows_per_pass=48))


def test_converter_folds_the_hidden_signs_into_output_norm():
    P = _prism()
    import torch
    from types import SimpleNamespace
    rot = object.__new__(P.PrismRotation)
    rot.weight_names, rot.inverse_names = {"output.weight"}, {"token_embd.weight"}
    rot.s_hidden = np.array(_signs(8, seed=5), dtype=np.float32)
    g = torch.arange(1, 9, dtype=torch.float32)
    t = SimpleNamespace(name="output_norm.weight", tensor_type=0, unpack=lambda target: [g])
    (w,), _ = rot.unpack(t, None)
    assert torch.equal(w, g * torch.from_numpy(rot.s_hidden))
    rot.block, rot.s_out = 1024, np.ones(6144, np.float32)
    e = rot.config_entry()
    assert e["lm_head"] == "rotated" and "output_norm" in e["folded"].split(",")
