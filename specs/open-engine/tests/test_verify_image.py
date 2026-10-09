# Traces: OPEN-DECODE-SPECULATIVE-VERIFY (canonical spec: specs/open-engine/spec.md)
"""Recipe output only; the bit-identity claim is the spec's manual procedure."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from recipes.cache import build_key
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.spec import FULL, LINEAR, ModelSpec

FIX = Path(__file__).resolve().parent / "fixtures"
ARGS = ["pool", "vxres", "consts", "state", "act", "ptab"]


def _bonsai(quant: str = "t2", lm_head: str | None = "rotated") -> ModelSpec:
    cfg = json.loads((FIX / "config_qwen35_27b.json").read_text(encoding="utf-8"))
    cfg["prism_hadamard"] = {"block_size": 1024, "og_signs": [1, -1] * 3072}
    if lm_head:
        cfg["prism_hadamard"]["lm_head"] = lm_head
    return dataclasses.replace(ModelSpec.from_hf_config(cfg), quant=quant)


def test_bonsai_t2_carries_the_verify_image(monkeypatch):
    monkeypatch.delenv("OPEN_LAYER_VERIFY", raising=False)
    spec = _bonsai()
    m = manifest(spec, key="k")
    v = m["verify"]
    L = layout(spec)
    assert v["width"] == 2
    assert v["slot_bytes"] >= L.STATE_BYTES and v["slot_bytes"] % 4096 == 0
    assert v["act_bytes"] == max(L.A_BYTES, L.AA_BYTES)
    assert m["contexts"]["verify"] == "vlx/final.xclbin"
    for k, patch in (("vlx", "stateslot"), ("vax", "attnposv"), ("vlm", None)):
        assert m["kernels"][k]["context"] == "verify"
        assert m["kernels"][k].get("patch") == patch
    assert m["kernels"]["lx"]["patch"] == "stateslot"
    assert v["programs"][LINEAR] == [{"op": "run", "kernel": "vlx", "args": ARGS}]
    assert v["programs"][FULL] == [{"op": "run", "kernel": "vax", "args": ARGS}]
    assert v["tail"][0]["kernel"] == "vlm" and v["xres"] == "vxres" and v["logits"] == "vlogits"
    g = m["globals"]
    assert g["vxres"] == 2 * spec.hidden * 4 and g["vlogits"] == 2 * spec.vocab * 4
    b = m["builds"]
    assert {b[k]["env"]["VUX_PART"] for k in ("vlx", "vax", "vlm")} == {"0", "1", "2"}
    assert all(b[k]["design"] == "verify/vux.py" for k in ("vlx", "vax", "vlm"))


def test_verify_off_and_unsupported(monkeypatch):
    spec = _bonsai()
    k_on = build_key(spec)
    monkeypatch.setenv("OPEN_LAYER_VERIFY", "0")
    m = manifest(spec, key="k")
    assert "verify" not in m and "verify" not in m["contexts"] and "vlx" not in m["kernels"]
    assert "patch" not in m["kernels"]["lx"]
    assert build_key(spec) != k_on
    monkeypatch.delenv("OPEN_LAYER_VERIFY")
    # no rotated ternary head in the layers' chunk law: no in-layer tail, so no verify image
    assert "verify" not in manifest(_bonsai(lm_head=None), key="k")
