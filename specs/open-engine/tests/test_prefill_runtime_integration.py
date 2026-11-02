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
    """The layer-major constant came from the MoE family, with a different reason."""
    src = engine()
    assert 'Qwen3.6-35B' in src, 'the provenance of the 64 must stay visible in the source'
    md = REPORT.read_text()
    assert 'inherited' in md
    assert '96' in md, 'the measured block-loop crossover must be recorded'


def test_the_crossover_is_selected_by_what_the_set_can_run_not_by_model_name():
    """27.1: a shape-identical derivative must be correct without an update.

    Two schedules, two measured crossovers. Picking by capability is what makes that
    possible; a model-name test would need editing for every new tag.
    """
    src = engine()
    assert 'block_loop_crossover = 96' in src, 'the block-loop crossover'
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


def test_the_measured_crossover_comes_from_a_fit_not_from_a_grid_point():
    """The regression that produced 128, kept as a test so it cannot come back.

    The old rule was "the smallest SAMPLED length at which the block wins", over a grid of
    1/8/32/64/128/192/255. Its own table crosses near 81, so the rule returned the next grid
    point (128) and overcharged every prompt in between. A crossover must come from where
    the two cost curves meet.
    """
    d = json.loads((ROOT / 'utilities/prefill-block/tail-crossover.json').read_text())
    assert d['physical_block'] == 256
    # The superseded grid measurement is recorded, with the reason it was wrong.
    assert d['v1_was_a_grid_artifact'] is True
    assert d['v1_grid'] == [1, 8, 32, 64, 128, 192, 255]
    assert 'grid point' in d['v1_why_wrong'] or 'grid' in d['v1_why_wrong']
    # The closure measurement is a fit, and the applied constant follows from it.
    fit = d['evidence']['linear_fit']
    assert fit['crossing_tokens'] < 96, 'the crossing is below the applied constant'
    assert d['applied']['block_loop_crossover'] == 96
    assert d['applied']['rebuilt'] is True, 'a constant nobody built is not a shipped constant'
    # The mechanism: the block costs the same whatever t_real holds.
    lo, hi = d['evidence']['block_cost_flat_ms']
    assert 0.9 <= hi / lo <= 1.1, f'the flat block cost must stay flat: {lo}..{hi}'
    # And every decisive point really ran a padded pass -- a point where the block side fell
    # back to sequential is two sequential runs and its ratio is noise.
    for r in d['evidence']['rows']:
        assert r['block_passes'] >= 1


def test_the_block_cost_is_recorded_as_flat_across_active_rows():
    """The mechanism, not just the number: a full-width dispatch whatever t_real is."""
    md = REPORT.read_text()
    assert 'flat' in md.lower()
    # The closure sweep's flat band, at full depth through the shipped binary.
    assert '33.1' in md and '34.5' in md


def _flat(md: str) -> str:
    """Markdown wraps phrases; collapse whitespace so a line break cannot break a claim."""
    return re.sub(r'\s+', ' ', md)


def test_the_superseded_harness_is_kept_as_the_record_of_the_grid_artifact():
    """The v1 tool is not deleted: it is the evidence of how 128 was produced."""
    src = CROSSOVER.read_text()
    assert 'Engine::prefill' in src, 'it was meant to measure the production path'
    md = _flat(REPORT.read_text())
    assert 'grid artifact' in md, 'the reason 128 was wrong must survive in the report'
    assert 'tail_crossover.py' in md, 'the superseded harness is still pointed at'
    # The report must name the level of every number, so none is read as a throughput claim.
    assert 'Core`-level' in md or 'sanity check, not a performance claim' in md


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
    """The closure trajectory, measured with logits through the harness CLI.

    Two runs exist and they measure different things on purpose. `decode-*-d16-*` drove the
    shipped `oflm`, which reports no logits, so it could only compare emitted text.
    `decode16-logits-*` drives the same `Engine::prefill` through the standalone CLI at full
    depth and does report them, which is what a cosine needs.
    """
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob('decode16-logits-*/result.json'))
    if not runs:
        pytest.skip('no 16-token logits trajectory committed')
    return json.loads(runs[-1].read_text())


def test_the_sixteen_token_trajectory_agrees_token_for_token():
    d = _d16()
    assert d['decode_steps'] == 16
    assert d['layers'] == 64
    assert d['verdict'] == 'PASS'
    tokens = d['tokens']
    assert len(tokens) == 16, 'one token comparison per decode step'
    assert all(t['agree'] for t in tokens), 'a single flipped token is a failure'
    assert d['summary']['tokens_agreeing'] == 16


def test_the_trajectory_reports_the_numbers_the_plan_asked_for():
    """cosine, max_abs AND max_rel, for every step -- not just "PASS"."""
    d = _d16()
    logits = [t for t in d['tensors'] if t['tensor'].startswith('logits@decode')]
    assert len(logits) == 16
    assert all(t['cosine'] >= 0.9999 for t in logits), 'the repository gate, unchanged'
    assert all('max_abs' in t and 'max_rel' in t for t in logits)
    s = d['summary']
    assert s['min_cosine'] >= 0.9999
    assert s['max_abs'] < 1e-2, 'absolute error stays at the 1e-2 scale'
    # max_rel is reported, and is large only because it divides by near-zero logits.
    assert s['max_rel'] > 1.0


def test_the_trajectory_errors_are_bounded_not_compounding():
    """The plan's stop condition: a trend that compounds, not one that oscillates."""
    d = _d16()
    ma = [t['max_abs'] for t in d['tensors'] if t['tensor'].startswith('logits@decode')]
    mid = len(ma) // 2
    early = max(ma[:mid])
    late = max(ma[mid:])
    assert late < 2.0 * early, f'max_abs compounding: {early} -> {late}'
    # And no new absolute threshold was invented to hide it.
    md = REPORT.read_text().lower()
    assert 'no new tolerance was introduced' in md


def test_no_weak_new_tolerance_was_invented():
    """The gate is the repository's, unchanged, in both the oracle and the report."""
    oracle = (ROOT / 'utilities/prefill-block/oracle.py').read_text()
    assert 'GATE_COSINE = 0.9999' in oracle
    md = REPORT.read_text()
    assert '0.9999' in md and 'gate' in md.lower()


# ---------------------------------------------------------------- the open gates

def test_the_report_does_not_claim_a_gate_it_did_not_run():
    """Every gate is either green with an artefact, or it says why not."""
    md = REPORT.read_text()
    assert 'PASS — hardware closure complete' in md
    for n in ('128', '256', '512', '1024'):
        assert f'| {n} |' in md, f'{n} must appear as a measured row, not just in prose'
    # The one case that could not be exercised must be named, not implied.
    assert 'N/A' in md and 'reset' in md.lower()


def test_the_rebuild_happened_and_its_provenance_is_recorded():
    """The earlier report said the binary could not be built. It was built; say how."""
    md = REPORT.read_text()
    assert 'openflowlm-build:ubuntu26' in md, 'the supported container build'
    assert '45ed9846122748f686fb77357f4255da0a6deb5a63b8567ad4da1c3d46eedd01' in md, \
        'the binary the closure ran on'
    assert '/opt/xilinx/xrt/bin/c++' in md, 'the compiler actually used'
    # And the older attempt is not erased: it is the reason the build note exists.
    assert '-fsyntax-only' in md or 'syntax check' in md


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


def test_the_old_matrix_is_recorded_as_a_route_comparison_not_boundary_evidence():
    """Its lengths were TEXT tokens, so a fixture "255" reached the engine as 266.

    It did run both routes on hardware and the outputs agreed, which is still worth having:
    it is a functional route comparison. What it cannot be is boundary evidence, and the
    report must say so rather than letting the labels imply otherwise.
    """
    d = _run('oflm-run-matrix-*/result.json')
    assert d['verdict'] == 'PASS'
    assert d['matrix']['failed'] == []
    off = d['matrix']['template_offset']
    # This is the proof that it was not the boundary: every length is offset.
    assert off > 0, 'the fixture lengths were text tokens, not engine tokens'
    for r in d['matrix']['rows']:
        assert r['block']['prefilled'] == r['text_tokens'] + off
    md = REPORT.read_text()
    assert 'functional route comparison' in md, 'the relabelling must be explicit'
    assert 'not' in md and 'boundary evidence' in md


def test_the_route_is_asserted_at_exact_engine_lengths():
    """The closure matrix: the engine's own count must equal the requested count."""
    d = _run('route-matrix-block-*/result.json')
    assert d['verdict'] == 'PASS', d.get('failed')
    assert d['route_reported_by'].startswith('OFLM_OPEN_PREFILL_ROUTE'), \
        'the route must be read from the engine, never timed'
    for r in d['rows']:
        assert r['token_match'], f"{r['requested']} did not reach the engine as itself"
        assert r['engine_tokens'] == r['requested']
        assert r['verdict'] == 'PASS', r


def test_the_boundary_lengths_are_the_policys_own_cases():
    """257, 383/384/385 and 513 are the plan's worked examples; check the arithmetic."""
    d = _run('route-matrix-block-*/result.json')
    got = {r['requested']: r['actual'] for r in d['rows']}
    # 256 = exactly one block, no tail.
    assert got[256]['blocks'] == 1 and got[256]['tail'] == 0
    # 257 = one block plus ONE stepped token, not a second padded pass.
    assert got[257]['blocks'] == 1 and got[257]['padded'] == 0
    assert got[257].get('tail_sequential') == 1
    # 383 = full block + a 127-token tail. 127 CLEARS the 96 crossover, so it is padded and
    # the prompt is two passes -- the first post-block tail that would be stepped is 351.
    assert got[383]['padded'] == 1 and got[383]['passes'] == 2
    assert got[384]['padded'] == 1 and got[384]['passes'] == 2
    # 512 = two full blocks; 513 = two blocks plus one stepped token.
    assert got[512]['blocks'] == 2 and got[512]['padded'] == 0
    assert got[513]['blocks'] == 2 and got[513].get('tail_sequential') == 1


def test_the_two_block_cases_are_distinguishable_from_one_block():
    """Two passes over every layer cannot be faster than one."""
    d = _run('oflm-sanity-*/result.json')
    rows = {r['text_tokens']: r for r in d['rows']}
    assert rows[128]['passes'] == 1 and rows[512]['passes'] == 2
    assert rows[128]['ttft_ms'] < rows[512]['ttft_ms']
    # And the cost is per BLOCK, not per active row: one block at 128 and at 256 is the
    # same dispatch, so their TTFTs stay within a small band.
    band = rows[256]['ttft_ms'] / rows[128]['ttft_ms']
    assert 0.75 <= band <= 1.35, f'one padded block should cost about the same: {band}'


def test_request_isolation_is_A_B_A_not_a_repeated_A():
    """A repeated identical request can match for the wrong reason; a different one cannot."""
    d = _run('fail-inject-serve-*/result.json')
    assert d['verdict'] == 'PASS'
    checks = {r['check']: r for r in d['rows']}
    assert checks['A_after_B_equals_A_before_B']['ok'], \
        'resident state leaked across requests if A changed after B ran'
    assert checks['A_and_B_are_different_prompts']['ok'], \
        'if A and B matched, the isolation check would pass vacuously'


def test_the_reset_case_is_recorded_as_not_applicable_not_as_a_pass():
    """serve has no reset endpoint; saying so is the honest outcome."""
    d = _run('oflm-serve-*/result.json')
    skips = [r for r in d['rows'] if r['verdict'] == 'SKIP']
    assert skips, 'the reset case must be present as SKIP, not silently dropped'
    assert all('no reset endpoint' in (r.get('reason') or '') for r in skips)
    assert d.get('reset_endpoint') is None, 'if serve gains a reset endpoint, exercise it'
    assert 'N/A' in REPORT.read_text()


def test_the_sanity_run_is_labelled_a_sanity_run_not_a_profile():
    """Phase 28 owns the profile; this run must never be quoted as one."""
    d = _run('oflm-sanity-*/result.json')
    assert d['verdict'] == 'PASS'
    assert 'not a production profile' in d['gate']
    assert 'Phase 28' in d['not_a_claim']
    assert 'OFLM_OPEN_LAYER_MAJOR=0' in d['note'], \
        'block-loop numbers must state that they are not the shipped default route'


def test_the_block_cost_is_linear_in_blocks_not_in_active_rows():
    """Both halves of the claim, from the committed sanity run.

    Flat in the active rows inside a block -- that is why a tail is padded or avoided.
    Linear in the number of blocks -- that is why a long prompt costs what it costs.
    """
    d = _run('oflm-sanity-*/result.json')
    rows = d['rows']
    # Every row's pass count is the policy's own prediction, so none of them is a surprise.
    for r in rows:
        assert r['passes'] == r['expected_passes'], r
    per_block = [r['ttft_ms'] / r['passes'] for r in rows if r['passes']]
    assert max(per_block) / min(per_block) <= 1.6, \
        f'per-block cost is not consistent across block counts: {per_block}'


def test_the_crossover_was_measured_with_the_block_side_forced():
    """The methodological trap: under the shipped policy the block below the crossover
    never runs, so both sides are sequential and their ratio is noise."""
    d = _run('crossover-local-*/result.json')
    assert d['verdict'] == 'PASS'
    assert d['block_min'] == 1, 'the block side must be forced below the crossover'
    assert 'OFLM_OPEN_GEMM_BLOCK_MIN' in d['why_forced']
    # Every decisive point really ran a padded pass.
    decisive = [r for r in d['rows'] if r['decisive']]
    assert len(decisive) >= 6
    assert all(r['block_passes'] >= 1 for r in decisive)
    # And the crossing comes from the fit, not from the nearest sampled point.
    assert d['linear_fit']['crossing_tokens'] is not None
    assert d['linear_fit']['crossing_tokens'] < min(
        r['engine_tokens'] for r in decisive if r['block_wins'])


def test_the_failure_handler_was_exercised_on_hardware():
    """Item 6: three cases, each proving a different branch actually ran."""
    d = _run('fail-inject-serve-*/result.json')
    assert d['verdict'] == 'PASS'
    checks = {r['check']: r for r in d['rows']}
    # Case B reached the SNAPSHOT branch, not the reset one.
    assert checks['B_injector_fired_on_a_non_zero_position']['ok']
    assert checks['B_reports_the_pre_prompt_state_was_restored']['ok']
    # Case C refused: the plan's "must not continue on dirty state".
    assert checks['C_refuses_and_generates_nothing']['ok']


def test_case_A_is_committed_too_through_run():
    """The reset branch, exercised on the CLI path rather than argued from the source."""
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob('fail-inject-2*/result.json'))
    assert runs, 'the case-A run through oflm run must be committed'
    d = json.loads(runs[-1].read_text())
    assert d['tool'] == 'prefill-block-fail-inject', \
        'fail-inject-serve-* is a different harness with a different schema'
    cases = d['cases']
    assert cases['A_before_first_block']['injected'] is True
    assert cases['A_before_first_block']['said_reset'] is True
    assert cases['A_before_first_block']['output'] == d['reference']['output']
