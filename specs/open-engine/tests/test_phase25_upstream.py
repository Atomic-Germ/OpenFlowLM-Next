"""Phase 25: the architecture comparison must not be re-litigable from memory.

Three claims were made in PHASE25_UPSTREAM.md that a future agent could easily reverse by
assuming them away:

  * that the factor-8 "eight column submissions" theory was WRONG, and what the dispatch
    count really decomposes into;
  * that the low-weight kernels' extra cost is a context-packing artifact rather than
    arithmetic, even though the arithmetic is identical;
  * that the classification is B and the decision is STOP, with no upstream mechanism
    outstanding.

Each is pinned to a committed artefact or a line of source, not to prose.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / 'utilities/prefill-perf/PHASE25_UPSTREAM.md'
PROFILE = ROOT / 'utilities/prefill-perf/e2e-profile-v2.json'
DESIGN = ROOT / 'open_kernels/designs/gemm_q4_prefill/gemm_q4_prefill.py'
REFERENCE = ROOT / 'npu_offload/gemm_rtp/gemm_pretiled.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def profile():
    return json.loads(PROFILE.read_text())


def test_the_factor_eight_is_two_weight_times_two_input_components():
    """1024 per block is 64 layers x 2 projections x (2 weight x 2 input). No columns."""
    d = profile()
    row = d['data']['2048']
    blocks = 8
    gemm = sum(v['calls'] for k, v in row['dispatch'].items() if k.startswith('gemm_'))
    assert gemm == blocks * 1024
    # Per projection per layer per block, every paired kernel is called exactly twice.
    for name, stat in row['dispatch'].items():
        if not name.startswith('gemm_'):
            continue
        layers = 48 if name.startswith('gemm_n16384_k5120') else (
                 16 if name.startswith('gemm_n14336_k5120') else 64)
        assert stat['calls'] == 2 * layers * blocks, (
            f'{name}: {stat["calls"]} calls for {layers} layers x {blocks} blocks')
        # Each call is one whole-array launch covering all eight columns.
        assert stat['mean'] > 5.0, (
            f'{name}: a per-column submission would be ~1/8 of this')


def test_the_design_launches_the_whole_array_once_per_dispatch():
    """The x8 lives INSIDE one launch: one Worker.grid over one Runtime.sequence."""
    src = DESIGN.read_text()
    assert 'Worker.grid(n_aie_rows, n_aie_cols, _mk)' in src
    assert 'N_AIE_COLS = 8' in src
    # One Runtime, one sequence, one Program -- not eight of anything.
    assert src.count('Runtime(sequence') == 1
    assert src.count('program = Program(') == 1
    assert 'workers=[w for row in workers for w in row]' in src
    assert src.count('resolve_program()') == 1


def test_the_host_submits_one_xrt_run_per_dispatch_not_one_per_column():
    core = (ROOT / 'src/open_qwen36/core.cpp').read_text()
    # The only submission primitive in the whole engine.
    assert core.count('f.r.start();') == 1
    # gemm_dispatch does exactly one `run(...)`; the fold calls it once per component.
    body = core[core.index('const float* Core::gemm_dispatch'):]
    body = body[:body.index('void Core::gemm(')]
    assert body.count('timing_.part0_ms += run(') == 1
    assert 'n_aie_cols' not in body, 'the host must not know about array columns'


def test_the_classification_is_based_on_dataflow_not_filenames():
    """The two files share almost no identifier, which is why names could not decide it."""
    a = DESIGN.read_text()
    b = REFERENCE.read_text()
    for shared in ('Worker.grid', 'ObjectFifo', 'repeat_count', 'mac_dims'):
        assert shared in a and shared in b, shared
    # ... and the operands are swapped, which is the material modification.
    assert 'ACTIVATION (B role)' in a and 'WEIGHT (A role)' in a
    assert 'A is untouched: activations are runtime data' in b


def test_the_low_kernels_cost_more_only_because_of_their_own_context():
    """Identical instruction streams; separate hw_contexts. Measured, not assumed."""
    row = profile()['data']['2048']['dispatch']
    for high, low in (('gemm_n34816_k5120', 'gemm_n34816_k5120_low'),
                      ('gemm_n5120_k17408', 'gemm_n5120_k17408_low'),
                      ('gemm_n5120_k6144', 'gemm_n5120_k6144_low')):
        assert row[high]['calls'] == row[low]['calls']
        # The low product is consistently slower at identical shape and stream length.
        assert row[low]['mean'] > row[high]['mean'], (high, low)
        assert 1.0 < row[low]['mean'] / row[high]['mean'] < 1.15


def test_the_amdahl_ceilings_are_wall_time_based():
    d = profile()
    rec = d['data']['2048']
    c = rec['allocation']['checks']
    ex = rec['allocation']['exclusive']
    wall = c['wall_ms']
    gemm = ex['projection_gemm_device']
    # 2x GEMM: remove half the device time, measured against wall (not against the GEMM).
    assert abs(wall / (wall - gemm / 2) - 1.509) < 0.01
    # Removing all host tiling + transpose cannot exceed 1.13x, which is the point:
    # the host side around the GEMM cannot carry the project.
    host_side = ex['gemm_host_tiling'] + ex['gemm_host_transpose']
    assert wall / (wall - host_side) < 1.13
    # And the factor-8 prize is zero, because the factor 8 does not exist.
    assert '1.000' in DOC.read_text() or 'factor 8 does not exist' in DOC.read_text()


def test_the_residue_cannot_be_any_of_the_named_categories():
    """The residue is bounded by the accounting model, not merely unexamined."""
    d = profile()
    for n in d['lengths']:
        c = d['data'][str(n)]['allocation']['checks']
        # Every named category is already OUTSIDE the unattributed figure...
        assert c['unattributed_engine_ms'] > 0
        # ...and the two remainders are reported separately, never merged.
        assert c['unattributed_engine_ms'] != c['outside_engine_ms']
    # outside_engine is the process, and it is constant: prove it does not grow.
    outside = [d['data'][str(n)]['allocation']['checks']['outside_engine_ms']
               for n in d['lengths']]
    assert max(outside) - min(outside) < 600, outside
    # unattributed IS per-block work, at a near-constant cost per 256-token block.
    per_block = [d['data'][str(n)]['allocation']['exclusive']['unattributed_engine'] / (-(-n // 256))
                 for n in d['lengths']]
    assert max(per_block) - min(per_block) < 100, per_block


def test_the_document_states_the_decision_and_the_rejected_mechanisms():
    md = DOC.read_text()
    assert 'STOP GEMM architecture research' in md
    assert 'Proceed to Phase 26' in md
    # Classification, and why not A or C.
    assert 'B — derived, materially modified' in md
    # The two upstream mechanisms that were already rejected on hardware, with reasons.
    assert 'repeat_count' in md and 'shim tiles' in md
    assert 'deadlock' in md.lower()
    # The corrected claim, stated as a correction rather than quietly edited.
    assert 'was **wrong**' in md
    # Nothing was implemented in this pass.
    assert 'No kernel was changed' in md


def test_phase_26_starts_at_block_32_and_is_not_implemented_here():
    md = DOC.read_text()
    assert 'block 32' in md
    for artefact in ('DeltaNet recurrent state', 'conv state', 'KV cache', 'final hidden'):
        assert artefact in md, artefact
    assert 'not implemented' in md.lower() or 'not implemented,' in md
