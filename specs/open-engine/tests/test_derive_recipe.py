# Traces: OPEN-FAMILY-ADDITION
"""Recipe constants are derivable from the spec; the design is the human input."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "utilities" / "derive-recipe.py"

_spec = importlib.util.spec_from_file_location("derive_recipe_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
dr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dr
_spec.loader.exec_module(dr)


def test_hardware_invariants_are_universal():
    h = dr.hardware_invariants()
    assert h["N_CORES"] == 8 and h["BAND_ROWS"] == 64 and h["ELEM"] == 4096


def test_quant_law_follows_the_spec_not_the_family():
    assert dr.quant_law("q4_1")["CH"] == 5120
    assert dr.quant_law("q4_0")["CH"] == 4608
    assert dr.quant_law("q8_0")["CH"] == 5120


def test_template_follows_capability():
    assert dr.template_for({"num_experts": 256, "experts_per_tok": 8})["template"] == "moe-router"
    assert dr.template_for({"conv_kernel": 4})["template"] == "hybrid-conv"
    assert dr.template_for({})["template"] == "dense"


def test_capability_constants_come_from_heads_and_experts():
    c = dr.capability_constants({"num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
                                 "num_experts": 512, "experts_per_tok": 8,
                                 "shared_expert_intermediate": 512,
                                 "moe_intermediate": 512})
    assert c["ATTN_CORES"] in (4, 6, 2)
    assert c["TOPK_MAX"] == 8 and c["EXPERTS"] == 512
    assert c["SLOTS"] == 9 and c["POOL_EXPERTS"] == 513


def test_the_design_is_named_as_a_human_input():
    inp = dr.design_inputs({}, dr.template_for({"num_experts": 10}))
    assert "design" in inp and "kernel_source_pattern" in inp
    assert "spec_target_repo" in inp
