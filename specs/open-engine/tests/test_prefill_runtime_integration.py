"""Phase 27: the production prefill policy, and the facts that must not be re-guessed.

Three things here are easy to undo by accident and expensive to rediscover:

  * block prefill is already wired into the runtime and ON by default -- a "let's integrate
    it" pass that adds a second path would duplicate `Engine::prefill`;
  * the tail crossover shipping in the tree (64) was measured for the MoE 35B, where the
    reason was expert streaming, and does NOT hold for this dense model;
  * the block route must not carry oracle overhead, and the physical tile must not become
    a user setting.

Hardware numbers are asserted for shape and provenance only -- no NPU result is faked.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / 'src/open_qwen36/engine.cpp'
CORE_HPP = ROOT / 'src/open_qwen36/core.hpp'
REPORT = ROOT / 'utilities/prefill-block/PHASE27_REPORT.md'
CROSSOVER = ROOT / 'utilities/prefill-block/tail_crossover.py'


def engine() -> str:
    return ENGINE.read_text()


# ---------------------------------------------------------------- already integrated

def test_the_runtime_already_routes_prefill_to_the_block_path():
    """A second integration would duplicate this, so its existence is pinned."""
    src = engine()
    assert 'Engine::prefill' in src
    # Block route, on by default: only "0" turns it off.
    assert 'OFLM_OPEN_GEMM_BLOCK' in src
    assert '!= "0"' in src, 'the default must be ON, with =0 the only opt-out'
    assert 'step_gemm_block' in src
    # The tail is padded with a repeated in-range id and t_real carries the real count.
    assert 'block.resize(GT' in src
    assert 't_real' in src


def test_the_sequential_fallback_is_still_there_and_reached_normally():
    src = engine()
    assert 'for (size_t i = 0; i < ids.size(); ++i) core_->step(' in src, \
        'the sequential path must survive as the fallback'
    # A kernel set with no block program must fall THROUGH to it, not return early.
    assert 'fall through to the sequential path below' in src
    assert '#109' in src, 'the early-return bug that left the prompt unprocessed'


def test_the_dense_route_uses_the_functions_the_oracle_measured():
    """engine.cpp prefers step_gemm_prompt only for MoE; dense takes the block loop."""
    src = engine()
    assert 'layer_major_ok()' in src
    # layer_major_ok requires every layer kind to be linear/full and is documented as the
    # MoE kinds, so a dense set (which has other kinds) takes the step_gemm_block loop.
    hpp = CORE_HPP.read_text()
    assert 'layer_major_ok' in hpp
    core = (ROOT / 'src/open_qwen36/core.cpp').read_text()
    fn = core[core.index('bool Core::layer_major_ok'):]
    fn = fn[:fn.index('Core::~Core')]
    assert 'kind != "linear" && kind != "full"' in fn


def test_the_physical_tile_is_read_from_the_manifest_not_a_setting():
    src = engine()
    assert 'gemm_block_t()' in src
    # And it is not a settings-registry entry.
    settings = json.loads((ROOT / 'src/config/settings.json').read_text())
    keys = json.dumps(settings)
    for forbidden in ('block_size', 'physical_t', 'gemm_block_t', 'tile'):
        assert forbidden not in keys, f'{forbidden} must not be a user setting'


# ---------------------------------------------------------------- no oracle overhead

def test_the_block_logits_dump_defaults_off():
    hpp = CORE_HPP.read_text()
    assert 'bool block_logits_all_ = false;' in hpp, \
        'per-token lm_head dumps must not be on by default'


def test_the_state_accessors_are_bring_up_only():
    """read_state / write_state_canary must not be called from the production path."""
    core = (ROOT / 'src/open_qwen36/core.cpp').read_text()
    start = core.index('size_t Core::state_region_bytes')
    # They are defined at the very end of core.cpp, next to the other bring-up accessors.
    assert start > core.index('void Core::read_act')
    for call in ('read_state(', 'write_state_canary('):
        # Only the definitions themselves and the act/state helpers may mention them.
        hits = [m.start() for m in re.finditer(re.escape(call), core)]
        assert hits, call
        assert all(h > start for h in hits), f'{call} is called outside its definition'


def test_the_oracle_flags_are_not_in_the_shipped_cli():
    """--dump-state / --state-canary belong to the standalone harness only."""
    vm = (ROOT / 'src/include/utils/vm_args.hpp').read_text()
    for flag in ('dump-state', 'state-canary', 'gemm-block'):
        assert flag not in vm, f'{flag} must not be a shipped oflm option'
    # They DO exist in the standalone harness, which is where the oracle lives.
    cli = (ROOT / 'src/open_qwen36/cli.cpp').read_text()
    assert '"--dump-state"' in cli and '"--state-canary"' in cli


# ---------------------------------------------------------------- the crossover

def test_the_report_records_that_64_was_inherited_not_measured():
    """The shipped constant came from the MoE family, with a different reason."""
    src = engine()
    assert 'Qwen3.6-35B' in src, 'the provenance of the 64 must stay visible in the source'
    md = REPORT.read_text()
    assert 'inherited' in md
    assert 'C = 128' in md, 'the measured crossover must be recorded'


def test_the_measured_crossover_is_conservative_and_inside_the_measured_gap():
    """64 is still sequential-cheaper; 128 is the first block-cheaper point measured."""
    md = REPORT.read_text()
    # The table must contain both the last sequential-cheaper and the first block-cheaper
    # point, so the crossover cannot be claimed outside the interval actually measured.
    assert re.search(r'\|\s*64\s*\|', md) and re.search(r'\|\s*\**128\**\s*\|', md)
    assert '1.18' in md, 'n=64 block/seq ratio, the regression the shipped 64 causes'
    assert '0.72' in md, 'n=128 block/seq ratio'
    # The committed data file must agree with the claimed crossover and its provenance.
    d = json.loads((ROOT / 'utilities/prefill-block/tail-crossover.json').read_text())
    assert d['physical_block'] == 256
    assert d['shipped_crossover'] == 64
    assert d['measured_crossover'] == 128
    assert 'Qwen3.6-35B' in d['shipped_crossover_provenance']
    n64 = next(r for r in d['rows'] if r['n'] == 64)
    n128 = next(r for r in d['rows'] if r['n'] == 128)
    assert n64['ratio'] > 1.0, 'n=64 must be the sequential-cheaper side'
    assert n128['ratio'] < 1.0, 'n=128 must be the first block-cheaper side'
    # The mechanism: the block costs the same whatever t_real holds.
    assert d['block_cost_flat_ms'][0] <= 7000 and d['block_cost_flat_ms'][1] <= 7000


def test_the_block_cost_is_recorded_as_flat_across_active_rows():
    """The mechanism, not just the number: a full-width dispatch whatever t_real is."""
    md = REPORT.read_text()
    assert 'flat' in md.lower()
    assert '6.4' in md and '6.8' in md


def test_the_crossover_harness_declares_it_was_not_run_green():
    """A tool that did not produce the shipped table must say so."""
    src = CROSSOVER.read_text()
    assert 'Engine::prefill' in src, 'it is meant to measure the production path'
    md = REPORT.read_text()
    assert 'not run green' in md
    assert 'Core`-level measurement' in md or '`Core`-level' in md


# ---------------------------------------------------------------- the trajectory

def _d16():
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob('decode-*-d16-*/result.json'))
    if not runs:
        pytest.skip('no 16-token trajectory run committed')
    return json.loads(runs[-1].read_text())


def test_the_sixteen_token_trajectory_agrees_token_for_token():
    d = _d16()
    assert d['decode_steps'] == 16
    assert d['layers'] == 64
    assert d['verdict'] == 'PASS'
    gt = next(t for t in d['tensors'] if t['tensor'] == 'greedy_tokens')
    assert gt['sequential'] == gt['block']
    assert len(gt['sequential']) == 16


def test_the_trajectory_errors_are_bounded_not_compounding():
    """27.9: identical tokens plus a max_abs trend that does not explode."""
    d = _d16()
    logits = [t for t in d['tensors'] if t['tensor'].startswith('logits@decode')]
    assert len(logits) == 16
    assert all(t['cosine'] >= 0.9999 for t in logits)
    ma = [t['max_abs'] for t in logits]
    first = sorted(ma[:5])[2]
    last = sorted(ma[-5:])[2]
    # Growth factor stays small: the plan's "monotonically/explosively" stop condition.
    assert last < 2.0 * first, f'max_abs compounding: {first} -> {last}'
    # And no new absolute threshold was introduced to hide it.
    assert 'new absolute' in REPORT.read_text().lower() or 'no new absolute-error' in \
        REPORT.read_text().lower()


def test_no_weak_new_tolerance_was_invented():
    """The gate is the repository's, unchanged, in both the oracle and the report."""
    oracle = (ROOT / 'utilities/prefill-block/oracle.py').read_text()
    assert 'GATE_COSINE = 0.9999' in oracle
    md = REPORT.read_text()
    assert '0.9999' in md and 'gate' in md.lower()


# ---------------------------------------------------------------- the open gates

def test_the_report_names_what_is_blocked_rather_than_claiming_it():
    md = REPORT.read_text()
    assert 'BLOCKED' in md
    for item in ('oflm run passes', 'oflm serve passes', 'production runtime benchmark'):
        assert item in md, item
    assert 'PARTIAL' in md, 'the overall status must not read as PASS'


def test_the_report_states_the_existing_fallback_gap():
    """27.11's second branch is not implemented; saying nothing would be worse."""
    md = REPORT.read_text()
    assert 'restore' in md.lower()
    assert 'unexpected' in md.lower()
