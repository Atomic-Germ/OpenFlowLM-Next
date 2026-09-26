# Traces: OPEN-DECODE-ONE-CONTEXT (canonical spec: specs/open-engine/spec.md)
"""The merged layer image: what the recipe emits by default for the qwen36moe family (one
context), what OPEN_LAYER_ONE_CTX=0 rolls back to (the two-context lx/ax shape), and that a
spec the merged image cannot carry (a q8 projection role) stays on two contexts.

The hardware claim (the layer walk changes context zero times, bit-exact) is manual; this is
the part that is deterministic and silent when wrong -- a recipe that quietly went back to two
contexts, or that forgot the linear program's sixth buffer argument, or that sent a q8 model
to a design that refuses it at build time, would pass every other test here."""
from __future__ import annotations

import dataclasses
import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "open_kernels"))

KERNELS = ("lx0", "lx1", "ax0", "ax1")
UNSET = None


@contextmanager
def flavour(value):
    """OPEN_LAYER_ONE_CTX set to `value` (None = unset). The recipe reads the switch per call,
    but the family module is imported once, so it is re-imported around the switch."""
    old = os.environ.get("OPEN_LAYER_ONE_CTX")
    if value is None:
        os.environ.pop("OPEN_LAYER_ONE_CTX", None)
    else:
        os.environ["OPEN_LAYER_ONE_CTX"] = value
    try:
        import recipes.qwen36moe as Q
        importlib.reload(Q)
        yield
    finally:
        if old is None:
            os.environ.pop("OPEN_LAYER_ONE_CTX", None)
        else:
            os.environ["OPEN_LAYER_ONE_CTX"] = old
        import recipes.qwen36moe as Q
        importlib.reload(Q)


def _manifest(spec=None):
    from recipes.load import default_spec
    from recipes.manifest import manifest

    return manifest(spec or default_spec())


def _assert_two_contexts(m):
    from recipes.spec import FULL, LINEAR

    assert m["contexts"]["lx"] == "lx0/final.xclbin" and m["contexts"]["ax"] == "ax0/final.xclbin"
    assert "layer" not in m["contexts"]
    assert [m["kernels"][k]["context"] for k in KERNELS] == ["lx", "lx", "ax", "ax"]
    assert m["layer_types"][LINEAR]["program"][0]["args"] == ["pool", "xres", "consts", "state", "act"]
    assert m["layer_types"][FULL]["program"][0]["args"] == ["pool", "xres", "consts", "state", "act", "ptab"]
    assert m["builds"]["lx0"]["design"] == "layer_x/lx.py"
    assert m["builds"]["ax0"]["design"] == "layer_x/ax.py"


def _assert_one_context(m):
    from recipes.spec import FULL, LINEAR

    # one context, one image, the same four kernel names the driver already knows
    assert "lx" not in m["contexts"] and "ax" not in m["contexts"]
    assert m["contexts"]["layer"] == "lx0/final.xclbin"
    assert [m["kernels"][k]["context"] for k in KERNELS] == ["layer"] * 4
    assert m["kernels"]["ax0"]["patch"] == "attnpos"
    assert m["kernels"]["lx1"]["patch"] == m["kernels"]["ax1"]["patch"] == "moeroute2"
    # one image means one kernel signature: the linear stream takes the attention layer's
    # sixth buffer argument and never touches it
    args = ["pool", "xres", "consts", "state", "act", "ptab"]
    assert m["layer_types"][LINEAR]["program"][0]["args"] == args
    assert m["layer_types"][FULL]["program"][0]["args"] == args
    # four parts of one design
    for name, part in zip(KERNELS, "0123"):
        b = m["builds"][name]
        assert b["design"] == "layer_x/ux.py"
        assert b["build_dir"] == f"layer_x/build_ux{part}"
        assert b["env"] == {"UX_PART": part}


def test_the_default_is_one_context():
    with flavour(UNSET):
        _assert_one_context(_manifest())


def test_one_is_one_context_too():
    with flavour("1"):
        _assert_one_context(_manifest())


def test_zero_rolls_back_to_two_contexts():
    with flavour("0"):
        _assert_two_contexts(_manifest())


def test_the_merged_image_carries_the_same_attention_row_block():
    """ux.py carries ax.py's attention cores, so the block-only walk's padded row count must
    reach the driver exactly as the two-context export sends it (else the fifo deadlocks
    rather than answering wrongly)."""
    with flavour(UNSET):
        one = _manifest()
    with flavour("0"):
        two = _manifest()
    assert one["kernels"]["ax0"].get("rb") == two["kernels"]["ax0"].get("rb") == 4


def test_a_q8_spec_stays_on_two_contexts():
    """ux.py refuses a q8 projection role (two GEMV entries do not fit the main core), so a
    q8 model of the family (the Ornith / Aquila fine-tunes) keeps the lx/ax layout by default
    instead of failing its export."""
    from recipes.load import default_spec

    q8 = dataclasses.replace(default_spec(), quant={"attn": "q8"})
    assert q8.q8_roles
    with flavour(UNSET):
        _assert_two_contexts(_manifest(q8))


def test_the_flavour_is_in_the_build_key():
    """Otherwise an export of one flavour over the other is skipped as up to date."""
    from recipes.cache import build_key
    from recipes.load import default_spec

    spec = default_spec()
    with flavour("0"):
        off = build_key(spec)
    with flavour(UNSET):
        on = build_key(spec)
    with flavour("1"):
        on1 = build_key(spec)
    assert off != on == on1
