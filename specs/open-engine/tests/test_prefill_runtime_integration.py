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


def test_the_crossover_is_selected_by_what_the_set_can_run_not_by_model_name():
    """27.1: a shape-identical derivative must be correct without an update.

    Two schedules, two measured crossovers. Picking by capability is what makes that
    possible; a model-name test would need editing for every new tag.
    """
    src = engine()
    assert 'block_loop_crossover = 128' in src, 'the dense block-loop crossover'
    assert 'layer_major_crossover = 64' in src, 'the MoE layer-major crossover is a different schedule'
    assert 'core_->layer_major_ok() ? layer_major_crossover : block_loop_crossover' in src, \
        'the choice must be capability-driven'
    # No model-name or family-name test may decide it.
    prefill = src[src.index('buffer<bf16> Engine::prefill'):]
    prefill = prefill[:prefill.index('prefill_images')]
    for fingerprint in ('27b', 'qwen3.8', 'family ==', 'model_name =='):
        assert fingerprint not in prefill, f'{fingerprint} must not steer the route choice'


def test_the_operator_override_still_wins_over_both_crossovers():
    src = engine()
    assert 'OFLM_OPEN_GEMM_BLOCK_MIN' in src
    after = src[src.index('const size_t block_loop_crossover'):]
    after = after[:after.index('mrope_active')]
    assert 'OFLM_OPEN_GEMM_BLOCK_MIN' in after, \
        'the diagnostic override must apply after the capability choice'


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


def _flat(md: str) -> str:
    """Markdown wraps phrases; collapse whitespace so a line break cannot break a claim."""
    return re.sub(r'\s+', ' ', md)


def test_the_crossover_harness_declares_it_was_not_run_green():
    """A tool that did not produce the shipped table must say so."""
    src = CROSSOVER.read_text()
    assert 'Engine::prefill' in src, 'it is meant to measure the production path'
    md = _flat(REPORT.read_text())
    assert 'not run green' in md
    # The table's level must stay labelled, so it is never read as an oflm throughput claim.
    assert 'Core`-level measurement' in md


def test_the_report_retracts_the_interactive_only_claim():
    """The earlier report called oflm run interactive-only. It was wrong; say so."""
    md = REPORT.read_text()
    assert 'That was wrong' in md, 'the retraction must be explicit, not quietly deleted'
    # And the mechanism that makes it wrong is real.
    posix = (ROOT / 'src/runner/cli_wide_posix.cpp').read_text()
    assert 'std::getline(std::cin, fallback)' in posix
    assert 'return "/bye";' in posix, 'EOF must end the session, which is what a pipe relies on'


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

def test_the_report_does_not_claim_a_gate_it_did_not_run():
    """The 1024/2048 sweep was not run; the report must say so, and say which were."""
    md = REPORT.read_text()
    assert 'NOT RE-RUN' in md, 'the build-blocked changes must be labelled'
    assert 'not run as its own pass' in md
    for n in ('1024', '2048'):
        assert n in md


def test_the_build_block_that_stopped_the_rebuild_is_recorded():
    """A syntax check is not a build; the distinction has to survive in the report."""
    md = REPORT.read_text()
    assert '-fsyntax-only' in md
    assert 'root-owned' in md, 'the actual reason the rebuild could not happen'
    # And the claim that the matrix is unaffected must carry its justification.
    assert '≥ 128 tokens' in md or '>= 128 tokens' in md


def test_2711_both_branches_are_in_the_code():
    """An advertised capability failing is not the same as a set without it."""
    src = engine()
    prefill = src[src.index('buffer<bf16> Engine::prefill'):]
    prefill = prefill[:prefill.index('prefill_images')]
    # The advertised-capability failure path rewinds rather than continuing on a bad cache.
    assert 'rewind' in prefill
    assert 'core_->reset()' in prefill, 'position-0 case undoes the prompt for free'
    assert 'core_->restore(snapshot_)' in prefill
    # And it throws when the rewind itself failed, instead of falling back onto it.
    assert 'could not be restored' in prefill
    assert 'NOT rewound' in prefill


def test_the_pre_prompt_snapshot_is_not_taken_on_the_common_path():
    """27.14: a full state readback per request would be real overhead."""
    src = engine()
    assert 'const bool multi_block = ids.size() > GT;' in src
    assert '!recover_by_reset && multi_block && save_pre_prompt_state()' in src
    # No unconditional checkpoint in prefill.
    prefill = src[src.index('buffer<bf16> Engine::prefill'):]
    prefill = prefill[:prefill.index('prefill_images')]
    assert '= core_->checkpoint();' not in prefill


def test_the_failure_message_is_emitted_only_in_the_handler():
    """A message before the try announces a failure on every successful run."""
    src = engine()
    prefill = src[src.index('buffer<bf16> Engine::prefill'):]
    prefill = prefill[:prefill.index('prefill_images')]
    body = prefill[prefill.index('const bool recover_by_reset'):]
    catch_at = body.index('catch (const std::exception& e)')
    assert 'block prefill failed' not in body[:catch_at], \
        'the diagnostic must not be emitted before the block runs'
    assert 'block prefill failed' in body[catch_at:]


# ---------------------------------------------------------------- the committed runs

def _run(pattern: str):
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob(pattern))
    if not runs:
        pytest.skip(f'no committed run matching {pattern}')
    return json.loads(runs[-1].read_text())


def test_the_run_matrix_is_committed_and_passes_every_boundary_length():
    d = _run('oflm-run-matrix-*/result.json')
    assert d['verdict'] == 'PASS'
    assert d['matrix']['verdict'] == 'PASS'
    assert d['matrix']['failed'] == []
    got = sorted(r['text_tokens'] for r in d['matrix']['rows'])
    assert got == [255, 256, 257, 511, 512, 513], 'every length the plan names must be present'
    for r in d['matrix']['rows']:
        assert r['verdict'] == 'PASS'
        # Both routes must have run on hardware, not just one.
        for side in ('block', 'sequential'):
            assert r[side]['rc'] == 0, (r['text_tokens'], side)
            assert r[side]['reached_model'], (r['text_tokens'], side)
            assert r[side]['matches_expected'], (r['text_tokens'], side)
            assert r[side]['output'], (r['text_tokens'], side)
        assert r['outputs_agree'], f"the two routes disagreed at {r['text_tokens']} tokens"


def test_the_matrix_pins_the_prompt_length_rather_than_approximating_it():
    """An approximate length would not be the boundary the plan asked for."""
    d = _run('oflm-run-matrix-*/result.json')
    off = d['matrix']['template_offset']
    assert isinstance(off, int) and off >= 0
    for r in d['matrix']['rows']:
        assert r['block']['prefilled'] == r['text_tokens'] + off
        assert r['sequential']['prefilled'] == r['text_tokens'] + off


def test_the_two_block_cases_are_distinguishable_from_one_block():
    """511/512/513 must actually take the multi-block path, not one padded block."""
    d = _run('oflm-run-matrix-*/result.json')
    one = [r for r in d['matrix']['rows'] if r['text_tokens'] in (255, 256, 257)]
    two = [r for r in d['matrix']['rows'] if r['text_tokens'] in (511, 512, 513)]
    assert one and two
    # A second block is a second pass over every layer; it cannot be faster than one.
    assert max(r['block']['wall_s'] for r in one) < min(r['block']['wall_s'] for r in two)


def test_the_serve_run_is_committed_and_passes_what_it_could_exercise():
    d = _run('oflm-serve-*/result.json')
    assert d['verdict'] == 'PASS'
    graded = [r for r in d['rows'] if r['verdict'] != 'SKIP']
    assert graded and all(r['verdict'] == 'PASS' for r in graded)
    checks = ' '.join(r['check'] for r in d['rows'])
    for need in ('fresh_request_255', 'fresh_request_256', 'fresh_request_257',
                 'second_request'):
        assert need in checks, need


def test_a_skipped_reset_case_is_not_counted_as_a_pass():
    """serve has no reset endpoint; that is a limitation, never a pass."""
    d = _run('oflm-serve-*/result.json')
    skips = [r for r in d['rows'] if r['verdict'] == 'SKIP']
    assert skips, 'the reset case must be present as SKIP, not silently dropped'
    assert all('no reset endpoint' in (r.get('reason') or '') for r in skips)
    assert d.get('reset_endpoint') is None, 'if serve gains a reset endpoint, exercise it'


def test_the_serve_timings_show_the_flat_block_cost_through_the_runtime():
    """The padding signature, measured through oflm rather than at Core level."""
    d = _run('oflm-serve-*/result.json')
    walls = [r['wall_s'] for r in d['rows']
             if r['check'].startswith('fresh_request_') and r.get('wall_s')]
    assert len(walls) >= 3
    lo, hi = min(walls), max(walls)
    assert hi - lo < 0.25 * lo, f'a padded block should be flat in the prompt length: {walls}'
