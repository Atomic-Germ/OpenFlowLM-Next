# Traces: OPEN-DECODE-ONE-CONTEXT (canonical spec: specs/open-engine/spec.md)
"""The merged layer image: what the recipe emits by default for the qwen36moe family (one
context), what OPEN_LAYER_ONE_CTX=0 rolls back to (the two-context lx/ax shape), and that a
spec the merged image cannot carry (a q8 projection role) stays on two contexts. The same for
Gemma 4's four layer kinds (gx.py), plus the RTP words and band split its streams depend on.

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

import pytest

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


# ---- Gemma 4 E-series: the four layer kinds as four streams over one image (designs/dense/gx.py)
G4_KERNELS = ("dx_local", "dx", "dx_local_shared", "dx_shared")
G4_PARTS = ("local", "global", "local_shared", "global_shared")


def _g4(stem, monkeypatch, value):
    from recipes.load import load_spec

    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    if value is None:
        monkeypatch.delenv("OPEN_LAYER_ONE_CTX", raising=False)
    else:
        monkeypatch.setenv("OPEN_LAYER_ONE_CTX", value)
    return load_spec(ROOT / "open_kernels" / "recipes" / "specs" / f"{stem}.json")


@pytest.mark.parametrize("stem", ["gemma4-e2b", "gemma4-e4b"])
def test_gemma4_default_is_one_context(stem, monkeypatch):
    m = _manifest(_g4(stem, monkeypatch, UNSET))
    assert set(m["contexts"]) == {"dx", "ln", "lm", "gemm", "fwa256", "fwa512"}   # gemm, fwa: the block route (OPEN-PREFILL-G4, -ATTN-WIDE)
    assert m["contexts"]["dx"] == "dx_local/final.xclbin"
    rows = {"gemma4-e2b": (1024, 2048), "gemma4-e4b": (2048, 4096)}[stem]
    for kn, part in zip(G4_KERNELS, G4_PARTS):
        k = m["kernels"][kn]
        assert (k["context"], k["insts"], k["patch"]) == ("dx", f"dx_{part}/insts.bin", "attnpos")
        assert k["kv_row"] == rows["global" in part]           # each kind keeps its own cache row
        b = m["builds"][f"dx_{part}"]
        assert b["design"] == "dense/gx.py" and b["env"] == {"G4_PART": part}


def test_gemma4_rtp_words(monkeypatch):
    """The counts each kind's stream writes into the shared main-core program: a word that disagreed
    with the stream's own fills would hang the cores or skip a band rather than fail a check."""
    from recipes import gemma4e as G4
    from recipes.spec import DENSE, DENSE_LOCAL, DENSE_LOCAL_SHARED, DENSE_SHARED

    e2b = _g4("gemma4-e2b", monkeypatch, UNSET)
    w = {k: G4.rtp_words(e2b, k) for k in (DENSE_LOCAL, DENSE, DENSE_LOCAL_SHARED, DENSE_SHARED)}
    assert [w[k]["qkv_bands"] for k in w] == [5, 10, 4, 8]
    assert [(w[k]["og_elems"], w[k]["o_k"], w[k]["o_groups"]) for k in w] == [(1, 2048, 16), (2, 4096, 32)] * 2
    assert [(w[k]["up_bands"], w[k]["h_elems"], w[k]["down_k"], w[k]["down_groups"]) for k in w] == \
        [(12, 6, 6144, 48)] * 2 + [(24, 12, 12288, 96)] * 2
    e4b = _g4("gemma4-e4b", monkeypatch, UNSET)
    w4 = {k: G4.rtp_words(e4b, k) for k in (DENSE_LOCAL, DENSE, DENSE_LOCAL_SHARED, DENSE_SHARED)}
    assert [w4[k]["qkv_bands"] for k in w4] == [6, 12, 4, 8]
    assert {(v["up_bands"], v["down_k"], v["down_groups"], v["down_chunks"]) for v in w4.values()} == {(20, 10240, 40, 80)}
    assert len(G4.RTP) <= G4.RTP_WORDS


@pytest.mark.parametrize("stem", ["gemma4-e2b", "gemma4-e4b"])
def test_gemma4_every_qkv_band_lands_once(stem, monkeypatch):
    """Over the image's main cores, each kind streams every q, k and v band exactly once, drains it to its
    own place in `act`, and each core streams as many bands as its RTP word says."""
    from recipes import dense as D, gemma4e as G4

    spec = _g4(stem, monkeypatch, UNSET)
    for kind in G4.kinds(spec):
        L, G = G4.image_geometry(spec, kind)
        bb, yb = D.band_bytes(G.HID, "q4_1"), D.BAND_ROWS * 4
        bands = []
        for c in range(G.N_CORES):
            n = 0
            for po, pn, ao, an in D.qkv_segments(G, L, c, bb):
                assert pn // bb == an // yb
                bands += [(po + i * bb, ao + i * yb) for i in range(pn // bb)]
                n += pn // bb
            assert n == G4.rtp_words(spec, kind)["qkv_bands"]
        want = [(L.POOL_Q + b * bb, L.AD_Q + b * yb) for b in range(G.QW // D.BAND_ROWS)]
        if not G.KV_SHARED:
            want += [(L.POOL_K + b * bb, L.AD_KVN + b * yb) for b in range(G.KVW // D.BAND_ROWS)]
            want += [(L.POOL_V + b * bb, L.AD_KVN + G.KVW * 4 + b * yb) for b in range(G.KVW // D.BAND_ROWS)]
        assert sorted(bands) == sorted(want), kind


def test_e2b_sliding_kinds_move_to_eight_cores(monkeypatch):
    """Their 4-band kv projection kept them on 4 cores as separate images; on the shared image cores 0-3
    take one k band each and cores 4-7 one v band each."""
    from recipes import dense as D, gemma4e as G4
    from recipes.spec import DENSE_LOCAL

    spec = _g4("gemma4-e2b", monkeypatch, UNSET)
    assert D.cores_for(G4.sub_spec(spec, DENSE_LOCAL)) == 4 and G4.image_cores(spec) == 8
    L, G = G4.image_geometry(spec, DENSE_LOCAL)
    bb = D.band_bytes(G.HID, "q4_1")
    kv = [D.qkv_segments(G, L, c, bb)[1][0] for c in range(8)]
    assert kv == [L.POOL_K + c * bb for c in range(4)] + [L.POOL_V + c * bb for c in range(4)]
