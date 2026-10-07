# Traces: OPEN-DECODE-ONE-CONTEXT-DENSE (canonical spec: specs/open-engine/spec.md)
"""The dense composition's merged layer image (designs/layer_x/dux.py): what the qwen35 recipe
emits for Ternary Bonsai 2 (all-t2, one context by default), what OPEN_LAYER_ONE_CTX=0 rolls
back to, and that every other dense spec keeps today's two contexts unless asked.

The hardware claim (no context switch in the layer walk, identical tokens) is manual. This is
the part that is deterministic and silent when wrong: a recipe that quietly went back to two
contexts, forgot the linear program's sixth buffer argument, or sent the Phase 1 (q4_1) build
or another dense model to the merged image nobody has validated, would pass every other test.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from recipes.cache import build_key
from recipes.manifest import manifest
from recipes.spec import FULL, LINEAR, ModelSpec

FIX = Path(__file__).resolve().parent / "fixtures"
ARGS6 = ["pool", "xres", "consts", "state", "act", "ptab"]


def _bonsai(quant: str) -> ModelSpec:
    """The Qwen3.8-27B fixture with the rotated-basis block Bonsai's container carries."""
    cfg = json.loads((FIX / "config_qwen35_27b.json").read_text(encoding="utf-8"))
    cfg["prism_hadamard"] = {"block_size": 1024, "og_signs": [1, -1] * 3072}
    return dataclasses.replace(ModelSpec.from_hf_config(cfg), quant=quant)


def _one_context(m: dict) -> None:
    assert "lx" not in m["contexts"] and "ax" not in m["contexts"]
    assert m["contexts"]["layer"] == "lx/final.xclbin"
    assert m["kernels"]["lx"]["context"] == m["kernels"]["ax"]["context"] == "layer"
    assert m["kernels"]["ax"]["patch"] == "attnpos" and "patch" not in m["kernels"]["lx"]
    # one image, one kernel signature: the linear stream takes ptab and never touches it
    assert m["layer_types"][LINEAR]["program"] == [{"op": "run", "kernel": "lx", "args": ARGS6}]
    assert m["layer_types"][FULL]["program"] == [{"op": "run", "kernel": "ax", "args": ARGS6}]
    b = m["builds"]
    assert (b["lx"]["design"], b["lx"]["env"]) == ("layer_x/dux.py", {"DUX_PART": "0"})
    assert (b["ax"]["design"], b["ax"]["env"]) == ("layer_x/dux.py", {"DUX_PART": "1"})
    assert b["lx"]["build_dir"] != b["ax"]["build_dir"]


def _two_contexts(m: dict) -> None:
    assert m["contexts"]["lx"] == "lx/final.xclbin" and m["contexts"]["ax"] == "ax/final.xclbin"
    assert "layer" not in m["contexts"]
    assert (m["kernels"]["lx"]["context"], m["kernels"]["ax"]["context"]) == ("lx", "ax")
    assert m["layer_types"][LINEAR]["program"][0]["args"] == ARGS6[:5]
    assert m["layer_types"][FULL]["program"][0]["args"] == ARGS6
    assert m["builds"]["lx"]["design"] == "layer_x/lx.py" and m["builds"]["ax"]["design"] == "layer_x/ax.py"


def test_bonsai_t2_is_one_context_by_default(monkeypatch):
    monkeypatch.delenv("OPEN_LAYER_ONE_CTX", raising=False)
    _one_context(manifest(_bonsai("t2"), key="k"))


def test_rollback_and_force(monkeypatch):
    t2 = _bonsai("t2")
    monkeypatch.setenv("OPEN_LAYER_ONE_CTX", "0")
    _two_contexts(manifest(t2, key="k"))
    k0 = build_key(t2)
    monkeypatch.delenv("OPEN_LAYER_ONE_CTX")
    # the explicit switch is in the build key, so a rolled-back set never shares the default's key
    assert build_key(t2) != k0
    # forced on, the Phase 1 (q4_1 + Hadamard) build of the same model merges too
    monkeypatch.setenv("OPEN_LAYER_ONE_CTX", "1")
    _one_context(manifest(_bonsai("q4_1"), key="k"))


@pytest.mark.parametrize("cfg", ["config_qwen35_27b.json", "config_qwen35_9b.json", "config_qwen35_0p8b.json"])
def test_other_dense_models_keep_two_contexts(monkeypatch, cfg):
    """Only the all-t2 spec was measured and gated on the merged image; the rest wait for theirs."""
    monkeypatch.delenv("OPEN_LAYER_ONE_CTX", raising=False)
    spec = ModelSpec.from_hf_config(json.loads((FIX / cfg).read_text(encoding="utf-8")))
    _two_contexts(manifest(spec, key="k"))
    _two_contexts(manifest(_bonsai("q4_1"), key="k"))


# ---- the tail as the image's third stream (dux.py part 2) and the attention walk in whole blocks
# Traces: OPEN-DECODE-ONE-CONTEXT-DENSE, OPEN-ATTN-CONTEXT
TAIL_ARGS = ["lmpool", "xres", "normw", "logits", "lmact", "ptab"]


def _bonsai_rotated_head(quant: str = "t2") -> ModelSpec:
    """Bonsai's container as it ships: the rotated ternary head too (prism_hadamard.lm_head)."""
    cfg = json.loads((FIX / "config_qwen35_27b.json").read_text(encoding="utf-8"))
    cfg["prism_hadamard"] = {"block_size": 1024, "og_signs": [1, -1] * 3072, "lm_head": "rotated"}
    return dataclasses.replace(ModelSpec.from_hf_config(cfg), quant=quant)


def test_bonsai_tail_runs_in_the_layer_context(monkeypatch):
    for k in ("OPEN_LAYER_ONE_CTX", "OPEN_LAYER_TAIL"):
        monkeypatch.delenv(k, raising=False)
    m = manifest(_bonsai_rotated_head(), key="k")
    _one_context(m)
    assert "ln" not in m["contexts"] and "lm" not in m["contexts"] and "ln" not in m["kernels"]
    assert m["kernels"]["lm"] == {"context": "layer", "insts": "lm/insts.bin", "build": "lm"}
    assert m["tail"] == [{"op": "run", "kernel": "lm", "args": TAIL_ARGS}]
    b = m["builds"]
    assert (b["lm"]["design"], b["lm"]["env"]) == ("layer_x/dux.py", {"DUX_PART": "2"})
    assert b["lm"]["build_dir"] not in (b["lx"]["build_dir"], b["ax"]["build_dir"])
    assert "ln" not in b and "lm_head_t2" not in b
    g = m["globals"]
    assert {"lmact", "lmpool", "normw", "logits", "xres"} <= set(g) and not {"zero", "xresf", "hn"} & set(g)


def test_tail_rollback_keeps_its_own_images(monkeypatch):
    monkeypatch.delenv("OPEN_LAYER_ONE_CTX", raising=False)
    spec = _bonsai_rotated_head()
    monkeypatch.delenv("OPEN_LAYER_TAIL", raising=False)
    k_default = build_key(spec)
    monkeypatch.setenv("OPEN_LAYER_TAIL", "0")
    m = manifest(spec, key="k")
    _one_context(m)                                   # the layers stay on one image
    assert (m["kernels"]["ln"]["context"], m["kernels"]["lm"]["context"]) == ("ln", "lm")
    assert [s["kernel"] for s in m["tail"]] == ["ln", "lm"]
    assert "lm_head_t2" in m["builds"] and "ln" in m["builds"] and "lm" not in m["builds"]
    assert build_key(spec) != k_default               # the explicit switch is in the build key


def test_tail_needs_the_rotated_head(monkeypatch):
    """The q8 head (no prism_hadamard.lm_head) is not the layers' chunk law: it keeps lm_head_q8."""
    for k in ("OPEN_LAYER_ONE_CTX", "OPEN_LAYER_TAIL"):
        monkeypatch.delenv(k, raising=False)
    m = manifest(_bonsai("t2"), key="k")
    assert m["kernels"]["lm"]["context"] == "lm" and m["builds"]["lm_head_q8"]


def test_bonsai_attention_walks_whole_blocks(monkeypatch):
    """attn.h ATTN_BLOCK_WIN at RB 4 on the merged t2 image: the manifest tells the driver to pad the
    window to whole blocks (`rb_win`); a kernel and a driver that disagree deadlock the fifo."""
    for k in ("OPEN_LAYER_ONE_CTX", "OPEN_LAYER_TAIL", "ATTN_RB"):
        monkeypatch.delenv(k, raising=False)
    assert manifest(_bonsai("t2"), key="k")["kernels"]["ax"]["rb_win"] == 4
    monkeypatch.setenv("ATTN_RB", "1")                # the probe rolls the walk back to single rows
    assert "rb_win" not in manifest(_bonsai("t2"), key="k")["kernels"]["ax"]
    monkeypatch.delenv("ATTN_RB")
    monkeypatch.setenv("OPEN_LAYER_ONE_CTX", "1")     # the q4_1 build on the merged image keeps RB 1
    assert "rb_win" not in manifest(_bonsai("q4_1"), key="k")["kernels"]["ax"]
