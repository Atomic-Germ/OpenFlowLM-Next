# Traces: OPEN-HADAMARD, OPEN-QUANT-T2 (canonical spec: specs/open-engine/spec.md)
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


def _cfg(signs=None) -> dict:
    cfg = json.loads((FIX / "config_qwen35_27b.json").read_text(encoding="utf-8"))
    cfg["prism_hadamard"] = {"block_size": 1024, "og_signs": signs or _signs()}
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


def test_t2_geometry_one_chunk_per_element_same_state_layout():
    q4, t2 = Q35.recipe(_spec("q4_1")).common, Q35.recipe(_spec("t2")).common
    assert (t2.TILE, t2.PER_CALL, t2.CALL_BYTES) == (Q36.T2_CHUNK, 1, Q36.T2_CHUNK)
    # the S slices follow the element, and the padded state rows come out the same
    assert t2.DN_ROWS == t2.CALL_BYTES // (4 * t2.DN_DIM) == 5
    assert t2.DN_PAD == q4.DN_PAD == 130
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
    route = Q35.gemm_route(spec)
    for g in route["layer_types"].values():
        for key in ("weights", "ffn_weights"):
            for w in g.get(key, {}).values():
                assert w["from"] == "pack" and all(o["op"] == "std_perm" for o in w["pack"])
                dst = 0
                for o in w["pack"]:                 # one contiguous run, in the order the GEMM reads
                    assert o["dst"] == dst
                    dst += o["nch"] * Q36.CHUNK


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
