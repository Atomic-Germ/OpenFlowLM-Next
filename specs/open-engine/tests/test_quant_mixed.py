# Traces: OPEN-QUANT-Q8, OPEN-LAYOUT-FREEZE, OPEN-OP-RANGE, OPEN-SPEC-DERIVE (canonical spec: specs/open-engine/spec.md)
"""A container that MIXES weight formats -- every Qwen3.5 dense model puts `linear_out`
at q8 and everything else at q4_1 -- needs both GEMV bodies on the same 16 KB main core,
and the Qwen3.5 4B's `lx` overflowed program memory when it tried
(.claude/plans/q8-hw-results.md section 2).

The fix is applied ONLY to such a spec: the two q4_1 entry points (a band into its y
element, a band into the act scratch) become one `gemv_q4_gyms` whose destination is a
runtime argument. An all-q4_1 or an all-q8 spec generates exactly the translation units it
generated before, so no shipped kernel set's object code moves.

What is NOT asserted here: the designs' `MIXED` flag and the `-Oz` GEMV flags live in
`designs/layer_x/xcommon.py` and `designs/dense/dx.py`, which import `aie.iron` and cannot
run without mlir-aie. They read the same condition as `gen_kernels.mixed()` (asserted
below) and are covered by the stub-IRON trace diff in .claude/plans/q8-mixed-handoff.md.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import struct

import pytest

from recipes import dense as DN
from recipes import qwen35 as Q35
from recipes.catalogue import OpRangeError
from recipes.load import HERE as RECIPES_DIR, load_spec
from recipes.spec import ModelSpec

SPECS = RECIPES_DIR / "specs"
DESIGNS = RECIPES_DIR.parent / "designs"
FIXTURES = __import__("pathlib").Path(__file__).resolve().parent / "fixtures"

PAIR = ("gemv_q4_gy.cc", "gemv_q4_gms.cc")
FOLD = "gemv_q4_gyms.cc"


@pytest.fixture
def unvalidated(monkeypatch):
    """Belt and braces for the synthetic specs below (a q8 `ffn`, a q8 dense Qwen3-4B); the
    four published Qwen3.5 sizes at `linear_out: q8` need no override -- see
    `test_every_qwen35_size_at_q8_composes_with_no_override`."""
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def _gen(which: str):
    """designs/<which>/gen_kernels.py, under its own module name (the two files share a
    basename and both define `files` / `mixed`)."""
    path = DESIGNS / which / "gen_kernels.py"
    spec = importlib.util.spec_from_file_location(f"gen_kernels_{which}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _q35(size: str, quant=None) -> ModelSpec:
    cfg = json.loads((FIXTURES / f"config_qwen35_{size}.json").read_text(encoding="utf-8"))
    spec = ModelSpec.from_hf_config(cfg)
    return dataclasses.replace(spec, quant=quant) if quant else spec


# ---- the fold fires exactly on a mixed spec

def test_a_mixed_qwen35_spec_generates_the_folded_entry_and_not_the_pair(unvalidated):
    gk = _gen("layer_x")
    R = Q35.recipe(_q35("4b", {"linear_out": "q8"}))
    assert gk.mixed(R)
    fs = gk.files(R)
    assert FOLD in fs
    assert not (set(PAIR) & set(fs)), "the folded entry replaces the pair, it does not join it"
    assert "gemv_q8_gy.cc" in fs                      # the q8 projection still has its own body
    assert "gemv_q8_gms.cc" not in fs                 # the FFN is q4_1 here
    body = fs[FOLD]
    assert "void gemv_q4_gyms(" in body
    assert "(dst < 0) ? y : ms + dst" in body


def test_every_qwen35_size_that_ships_is_mixed_and_folds(unvalidated):
    """`linear_out` is the only q8 role a Qwen3.5 container has, at every size."""
    gk = _gen("layer_x")
    for size in ("0p8b", "2b", "4b", "9b"):
        R = Q35.recipe(_q35(size, {"linear_out": "q8"}))
        assert gk.mixed(R), size
        assert FOLD in gk.files(R), size


def test_every_qwen35_size_at_q8_composes_with_no_override(monkeypatch):
    """The q8 out projection's GEMV reduces over `lin_value_width` (4096 on the 9B / 4B,
    2048 on the 2B / 0.8B), NOT over `hidden` -- both K are already in `gemv_q8`'s validated
    set from OPEN-QUANT-Q8's 35B pass, so a Qwen3.5 q8 export needs no
    `OPEN_KERNELS_UNVALIDATED` (2026-09-07 hardware pass, .claude/plans/q8m-hw-results.md)."""
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    for size, k in (("9b", 4096), ("4b", 4096), ("2b", 2048), ("0p8b", 2048)):
        spec = _q35(size, {"linear_out": "q8"})
        assert spec.lin_value_width == k, size
        Q35.recipe(spec)                      # composes: no OpRangeError, no override


def test_an_all_q4_1_qwen35_spec_generates_todays_pair(unvalidated):
    """The q4_1 path the hardware stream is validating right now: unchanged."""
    gk = _gen("layer_x")
    for size in ("0p8b", "2b", "4b", "9b"):
        R = Q35.recipe(_q35(size))
        assert not gk.mixed(R), size
        fs = gk.files(R)
        assert set(PAIR) <= set(fs), size
        assert FOLD not in fs, size
        assert not [n for n in fs if n.startswith("gemv_q8")], size


def test_an_all_q8_spec_needs_no_fold(unvalidated):
    """With every GEMV role at q8 there is no q4_1 entry left to fold -- the case the 35B
    fine-tunes built and passed on hardware."""
    gk = _gen("layer_x")
    R = Q35.recipe(_q35("4b", {"attn": "q8", "linear": "q8", "linear_out": "q8", "ffn": "q8"}))
    assert not gk.mixed(R)
    fs = gk.files(R)
    assert FOLD not in fs
    assert {"gemv_q8_gy.cc", "gemv_q8_gms.cc"} <= set(fs)
    # the pair is still WRITTEN (no ExternalFunction names it, so nothing compiles it) --
    # that is what the shipped q8 35B kernel set was built from and it must not move
    assert set(PAIR) <= set(fs)


def test_the_checked_in_specs_generate_exactly_the_translation_units_they_did(unvalidated):
    """Every spec in recipes/specs/ is all-q4_1, so none of them may gain or lose a TU."""
    for name, which, fam in (("qwen36-35b-a3b", "layer_x", None), ("qwen35-9b", "layer_x", Q35),
                             ("qwen3-4b", "dense", DN), ("llama31-8b", "dense", DN),
                             ("gemma3-4b", "dense", DN), ("hy-mt2-7b", "dense", DN)):
        gk = _gen(which)
        spec = load_spec(SPECS / f"{name}.json")
        assert not spec.q8_roles, name
        R = (fam or __import__("recipes.qwen36moe", fromlist=["x"])).recipe(spec)
        assert not gk.mixed(R), name
        fs = gk.files(R)
        assert FOLD not in fs, name
        assert "gemv_q4_gy.cc" in fs, name
        assert not [n for n in fs if n.startswith("gemv_q8")], name


# ---- the dense designs take the same switch

def test_a_mixed_dense_spec_folds_too(unvalidated):
    """No shipped dense container is q8, but designs/dense/dx.py is not MoE-only either."""
    gk = _gen("dense")
    R = DN.recipe(dataclasses.replace(load_spec(SPECS / "qwen3-4b.json"), quant={"attn": "q8"}))
    assert gk.mixed(R)
    fs = gk.files(R)
    assert FOLD in fs and not (set(PAIR) & set(fs))
    assert "gemv_q8_gy.cc" in fs


# ---- the generator leaves the design directory holding only what it compiles

def test_generate_writes_the_fold_and_removes_the_pair(tmp_path, unvalidated):
    gk = _gen("layer_x")
    for n in PAIR:
        (tmp_path / n).write_text("stale\n", encoding="utf-8")
    gk.generate(Q35.recipe(_q35("4b", {"linear_out": "q8"})), tmp_path)
    assert (tmp_path / FOLD).is_file()
    assert not any((tmp_path / n).exists() for n in PAIR)
    # and back again: a q4_1 spec restores the pair and drops the folded entry
    gk.generate(Q35.recipe(_q35("4b")), tmp_path)
    assert all((tmp_path / n).is_file() for n in PAIR)
    assert not (tmp_path / FOLD).exists()


# ---- the refusal: one fold, on the q4_1 side only

def test_a_mixed_spec_with_the_ffn_at_q8_is_refused_by_name():
    """The fold covers the q4_1 pair. An FFN role at q8 beside a q4_1 projection would
    need a SECOND fold on the q8 side, so the recipe refuses it rather than generating one."""
    with pytest.raises(OpRangeError, match="ffn=q8"):
        Q35.recipe(_q35("4b", {"ffn": "q8"}))
    with pytest.raises(OpRangeError, match="ffn=q8"):
        Q35.recipe(_q35("4b", {"ffn": "q8", "linear_out": "q8"}))
    with pytest.raises(OpRangeError, match="ffn=q8"):
        DN.recipe(dataclasses.replace(load_spec(SPECS / "qwen3-4b.json"), quant={"ffn": "q8"}))


def test_an_all_q8_spec_is_not_caught_by_that_refusal(unvalidated):
    """Every role at q8 means no q4_1 GEMV is left, so there is nothing to fold and
    nothing to refuse."""
    Q35.recipe(_q35("4b", {"attn": "q8", "linear": "q8", "linear_out": "q8", "ffn": "q8"}))
    DN.recipe(dataclasses.replace(load_spec(SPECS / "qwen3-4b.json"),
                                  quant={"attn": "q8", "ffn": "q8"}))


# ---- the width where the mixed core does NOT fit: the recipe narrows, it does not build
#
# The 4B's container derives `linear_out: q8` like its three siblings, and its mixed `lx`
# is the one that overflows 16 KB of program memory with both flag levers spent
# (.claude/plans/q8m-hw-results.md section 2). The recipe layer encodes that hardware fact:
# `catalogue.MIXED_CORE_FITS` lists the (family, hidden) widths whose mixed core has been
# built, and `load.narrow_to_buildable` puts `linear_out` back to q4_1 at any other width
# rather than handing the user an export that dies 60 s into aiecc.

Q35_CHUNKS = {"self_attn.q_proj.weight": 5120, "self_attn.k_proj.weight": 5120,
              "self_attn.v_proj.weight": 5120, "self_attn.o_proj.weight": 5120,
              "linear_attn.qkv_proj.weight": 5120, "self_attn.gate_proj.weight": 5120,
              "mlp.up_proj.weight": 5120, "mlp.gate_proj.weight": 5120,
              "mlp.down_proj.weight": 5120,
              "linear_attn.ssm_out_proj.weight": 8704}     # the one q8 role every size ships


def _container(tmp_path, size: str):
    """A model directory shaped like a published Qwen3.5 container: its real config.json
    plus a `model.q4nx` safetensors HEADER whose chunk sizes are the ones those containers
    carry (q4_1 everywhere, q8 on `ssm_out_proj`). No weight byte is written -- the header
    is all `container_chunk_bytes` reads."""
    d = tmp_path / f"Qwen3.5-{size}-NPU2"
    d.mkdir(parents=True)
    (d / "config.json").write_bytes((FIXTURES / f"config_qwen35_{size}.json").read_bytes())
    hdr = {"__metadata__": {"format": "q4nx"}}
    for name, ch in Q35_CHUNKS.items():
        hdr[f"model.layers.0.{name}"] = {"dtype": "I8", "shape": [4, ch]}
    blob = json.dumps(hdr).encode()
    with open(d / "model.q4nx", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    return d


def test_the_4b_narrows_linear_out_to_q4_1_because_its_mixed_core_does_not_fit(tmp_path, capsys):
    """Hidden 2560 is not in MIXED_CORE_FITS, so the derived map loses its q8 role and the
    spec hashes as the bare q4_1 string -- the shipped kernel set's own spec."""
    from recipes.load import spec_from_model_dir
    spec = spec_from_model_dir(_container(tmp_path, "4b"))
    assert spec.hidden == 2560
    assert spec.quant == "q4_1"
    assert not spec.q8_roles


def test_the_9b_2b_and_0p8b_keep_the_containers_q8_out_projection(tmp_path):
    """The three widths whose mixed `lx` built and passed on hardware on 2026-09-07."""
    from recipes.load import spec_from_model_dir
    for size, hidden in (("9b", 4096), ("2b", 2048), ("0p8b", 1024)):
        spec = spec_from_model_dir(_container(tmp_path / size, size))
        assert spec.hidden == hidden, size
        assert spec.quant == {"linear_out": "q8"}, size


def test_the_narrowing_note_names_the_width_the_reason_and_the_fallbacks_cost(tmp_path, capsys):
    """A silent downgrade is worse than no downgrade: the note has to say which width, why
    (program memory), and what it costs (0.999682, OPEN-QUANT-Q8's re-quantizing fallback)."""
    from recipes.load import spec_from_model_dir
    spec_from_model_dir(_container(tmp_path, "4b"))
    note = capsys.readouterr().err
    assert "hidden 2560" in note
    assert "linear_out narrowed to q4_1" in note
    assert "program memory" in note
    assert "0.999682" in note
    assert "q8m-hw-results.md" in note
    assert len([line for line in note.strip().split("\n") if line.strip()]) == 1


def test_a_size_that_fits_prints_no_note(tmp_path, capsys):
    from recipes.load import spec_from_model_dir
    spec_from_model_dir(_container(tmp_path, "9b"))
    assert capsys.readouterr().err == ""


def test_the_validated_set_holds_exactly_the_three_widths_that_built(tmp_path):
    from recipes.catalogue import MIXED_CORE_FITS, mixed_core_fits
    assert MIXED_CORE_FITS == frozenset({("qwen35", 4096), ("qwen35", 2048), ("qwen35", 1024)})
    assert not mixed_core_fits("qwen35", 2560)
    assert mixed_core_fits("qwen35", 4096)


def test_narrowing_leaves_an_all_q8_map_alone(tmp_path):
    """Every role at q8 is not a mixed core -- one format's GEMV bodies, nothing to fold --
    so the 2560 width is no reason to touch it."""
    from recipes.load import narrow_to_buildable
    can = frozenset({"attn", "linear", "linear_out", "ffn"})
    m = {r: "q8" for r in can}
    spec = _q35("4b")
    assert narrow_to_buildable(spec, dict(m), can) == m


def test_force_q4_1_still_wins_over_the_narrowing(tmp_path, monkeypatch):
    """The A/B switch is unchanged: it returns the bare string for every size."""
    from recipes.load import spec_from_model_dir
    monkeypatch.setenv("OPEN_KERNELS_FORCE_Q4_1", "1")
    for size in ("9b", "4b", "2b", "0p8b"):
        assert spec_from_model_dir(_container(tmp_path / size, size)).quant == "q4_1"
