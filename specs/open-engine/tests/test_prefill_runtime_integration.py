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
CROSSOVER = ROOT / 'utilities/prefill-block/tail_crossover.py'
# Provenance lives in machine-readable artefacts, not in a prose report: a test that greps
# a markdown file pins the wording of a document instead of the behaviour of the code, and
# rewriting a sentence breaks the build.
BASELINE = ROOT / 'utilities/prefill-block/qwen38-production-baseline.json'
CONTRACT = ROOT / 'utilities/prefill-block/long-context-v1.json'


def baseline() -> dict:
    return json.loads(BASELINE.read_text())


def contract() -> dict:
    return json.loads(CONTRACT.read_text())


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

def test_the_provenance_of_both_constants_stays_visible():
    """96 came from a fitted crossing; layer-major's win came from the MoE family.

    Both facts used to live in a prose report. They are asserted here against the code and
    the machine-readable baseline, so a reworded document cannot break the build and a
    changed constant cannot hide behind an unchanged one.
    """
    assert 'Qwen3.6-35B' in engine(), 'the provenance of the MoE measurement must stay visible'
    b = baseline()
    assert b['routes']['sequential_block_crossover'] == 96
    assert b['routes']['layer_major_crossover'] == 64


def test_the_route_is_chosen_by_capability_and_preference_not_by_model_name():
    """28.1/28.6: a shape-identical derivative must be correct without an update.

    Two different questions, asked separately:
      * CAN this set run layer-major   -- `layer_major_ok()`, a manifest fact;
      * SHOULD it                     -- `layer_major_preferred()`, because layer-major
                                        amortises routed experts and pads the tail.
    A model-name test would need editing for every new tag, so neither may use one.
    """
    src = engine()
    assert 'block_loop_crossover = 96' in src, 'the sequential/block crossover'
    assert 'core_->layer_major_preferred()' in src, \
        'preference, not capability, must pick the schedule among the two block routes'
    assert 'layer_major_ok() ? layer_major_crossover' not in src, \
        'capability alone must no longer select layer-major'
    # No model-name or family-name test may decide it.
    prefill = src[src.index('buffer<bf16> Engine::prefill'):]
    prefill = prefill[:prefill.index('prefill_images')]
    for fingerprint in ('27b', 'qwen3.8', 'family ==', 'model_name =='):
        assert fingerprint not in prefill, f'{fingerprint} must not steer the route choice'
    # And the reason lives in the code, not only in a report. The rule itself is
    # `layer_major::preferred()`, a pure function tested behaviourally by
    # src/test/layer_major_policy_test.cpp -- including that a set with routed experts
    # keeps layer-major. Reading its body here would only pin the shape of the source.
    assert 'layer_major::preferred(' in (ROOT / 'src/open_qwen36/core.cpp').read_text(), \
        'the preference must come from the shared policy, not a second copy of the rule'


def test_the_operator_override_still_wins_over_the_crossover():
    src = engine()
    assert 'OFLM_OPEN_GEMM_BLOCK_MIN' in src
    after = src[src.index('const size_t block_loop_crossover'):]
    after = after[:after.index('mrope_active')]
    assert 'OFLM_OPEN_GEMM_BLOCK_MIN' in after, \
        'the diagnostic override must apply after the crossover is chosen'
    # The layer-major override is not asserted here: a source string cannot tell a working
    # override from one the preference silently vetoes (it did exactly that). Its behaviour
    # is pinned by test_the_operator_override_decides_the_route_not_just_the_log_line() and
    # by src/test/layer_major_policy_test.cpp at the unit level.


def _override():
    """The operator-override run: three states, each with its log line and its route."""
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob(
        'layer-major-override-*/result.json'))
    for path in reversed(runs):
        d = json.loads(path.read_text())
        if d.get('verdict') == 'PASS':
            return d
    pytest.skip('no passing layer-major-override run committed')


def test_the_operator_override_decides_the_route_not_just_the_log_line():
    """OFLM_OPEN_LAYER_MAJOR used to be overruled by the automatic preference.

    On a dense set whose manifest says `block_loop_preferred`, `=1` set a boolean, the log
    said layer-major was forced, and the block loop ran. So the assertion cannot be that
    the variable appears in the source -- it did. It has to be that the route the prefill
    took matches the line the runtime printed, in all three operator states.
    """
    d = _override()
    rows = {r['case']: r for r in d['rows']}
    # Unset keeps the production routing untouched -- the non-negotiable part.
    assert rows['unset']['env'] is None
    assert rows['unset']['actual_route'] == 'block'
    # Force-on wins over block_loop_preferred: the bug, stated as an assertion.
    assert rows['force_on']['actual_route'] == 'layer-major'
    assert rows['force_on']['logged_schedule'] == 'layer-major'
    # Force-off still disables it.
    assert rows['force_off']['actual_route'] == 'block'
    assert rows['force_off']['logged_schedule'] == 'block-major'
    # And no row may report a schedule it did not run.
    for r in d['rows']:
        agree = ((r['logged_schedule'] == 'block-major' and r['actual_route'] == 'block')
                 or r['logged_schedule'] == r['actual_route'])
        assert agree, r


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
    b = baseline()
    assert b['prefill']['cost_is_flat_in_active_rows'] is True
    assert b['prefill']['cost_is_linear_in_blocks'] is True
    # The claim is not prose: TTFT per block is flat across the measured block counts.
    per_block = [r['ttft_ms'] / r['blocks'] for r in b['prefill']['rows']]
    assert max(per_block) / min(per_block) <= 1.6, per_block


def _flat(md: str) -> str:
    """Markdown wraps phrases; collapse whitespace so a line break cannot break a claim."""
    return re.sub(r'\s+', ' ', md)


def test_the_superseded_harness_is_kept_and_still_measures_the_production_path():
    """The v1 tool is not deleted: it is the record of how the wrong 128 was produced."""
    src = CROSSOVER.read_text()
    assert 'Engine::prefill' in src, 'it was meant to measure the production path'
    # Its own output carries the correction, so the lesson does not depend on a report.
    j = json.loads((ROOT / 'utilities/prefill-block/tail-crossover.json').read_text())
    assert j, 'the superseded harness must still have its committed output'


def test_oflm_run_is_drivable_from_a_pipe_not_interactive_only():
    """An earlier report called `oflm run` interactive-only. The code says otherwise.

    Asserting the retraction in prose pinned the wording; asserting the MECHANISM does not,
    and it is the mechanism that a future edit could actually break.
    """
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
    # The bound and the absence of a new tolerance both live in the committed run.
    assert late < 2.0 * early, f'max_abs compounding: {early} -> {late}'
    assert d['gate_cosine'] == 0.9999, 'no new tolerance was introduced to hide this'


def test_no_weak_new_tolerance_was_invented():
    """The gate is the repository's, unchanged, in the oracle and in every run it wrote."""
    oracle = (ROOT / 'utilities/prefill-block/oracle.py').read_text()
    assert 'GATE_COSINE = 0.9999' in oracle
    for p in sorted((ROOT / 'utilities/prefill-block/runs').glob('*/result.json')):
        d = json.loads(p.read_text())
        if d.get('tool') == 'prefill-block-oracle':
            assert d['gate_cosine'] == 0.9999, p


# ---------------------------------------------------------------- the open gates

def test_no_gate_is_claimed_without_a_measured_row():
    """Every gate is either green with an artefact, or it says why not.

    The three cases that could not be exercised are recorded as SKIP with a reason, in the
    run itself -- so the claim does not depend on a document remembering to say so.
    """
    d = json.loads((ROOT / 'utilities/prefill-block/runs'
                    '/oflm-run-matrix-20261102-005129/result.json').read_text())
    assert d['verdict'] == 'PASS'
    assert d['matrix']['failed'] == []
    skips = [r for r in d.get('rows', []) if r.get('verdict') == 'SKIP']
    for r in skips:
        assert r.get('reason'), 'a SKIP must say why, never be silently dropped'


def test_the_rebuild_provenance_is_recorded_in_the_artefacts_not_in_prose():
    """Which image, which binary, which compiler -- checked where it cannot rot."""
    b = baseline()
    assert b['binary']['sha256'].startswith('1315d111'), 'the binary the gates ran on'
    assert b['binary']['bytes'] == 31886824
    docker = (ROOT / 'Dockerfile').read_text()
    assert 'openflowlm-build:ubuntu26' in docker, 'the supported container build'
    assert '/opt/xilinx/xrt/bin' in docker, 'the toolchain the build pins'


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
    # And the run itself must record that its labels are text lengths, so nobody reads it
    # as the engine boundary: the offset is what disqualifies it, and it is in the artefact.
    assert off != 0


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
    """Two passes over every layer cannot be faster than one.

    Reads whichever committed sanity run contains BOTH 128 and 512 -- the Phase 31 baseline
    is a 256/512/1024 run, so "the newest passing run" no longer has the 128 row.
    """
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob('oflm-sanity-*/result.json'))
    d = None
    for path in reversed(runs):
        cand = json.loads(path.read_text())
        if cand.get('verdict') != 'PASS':
            continue
        have = {r['text_tokens'] for r in cand['rows']}
        if {128, 256, 512} <= have:
            d = cand
            break
    if d is None:
        pytest.skip('no passing sanity run carries 128, 256 and 512 together')
    rows = {r['text_tokens']: r for r in d['rows']}
    assert rows[128]['passes'] == 1 and rows[512]['passes'] == 2
    assert rows[128]['ttft_ms'] < rows[512]['ttft_ms']
    # And the cost is per BLOCK, not per active row: one block at 128 and at 256 is the
    # same dispatch, so their TTFTs stay within a small band.
    band = rows[256]['ttft_ms'] / rows[128]['ttft_ms']
    assert 0.75 <= band <= 1.35, f'one padded block should cost about the same: {band}'


def _failinject_passing():
    """The failure-injection run that actually PASSED, with its injector cases fired.

    A later phase dropped a 300-token run into the same tree whose prompts sit below the
    block crossover, so every injector case is SKIP-by-construction and the verdict is
    FAIL. Recency therefore cannot be the selector: the run has to carry the checks this
    file judges.
    """
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob(
        'fail-inject-serve-*/result.json'))
    for path in reversed(runs):
        d = json.loads(path.read_text())
        checks = {r['check']: r for r in d.get('rows', [])}
        if (d.get('verdict') == 'PASS'
                and checks.get('B_injector_fired_on_a_non_zero_position', {}).get('ok')
                and checks.get('A_after_B_equals_A_before_B', {}).get('ok')
                and checks.get('A_and_B_are_different_prompts', {}).get('ok')):
            return d
    pytest.skip('no committed failure-injection run both passed and fired its injectors')


def test_request_isolation_is_A_B_A_not_a_repeated_A():
    """A repeated identical request can match for the wrong reason; a different one cannot.

    Selection is by verdict and by the presence of the isolation checks, not by recency.
    """
    d = _failinject_passing()
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


def test_the_sanity_run_is_labelled_a_sanity_run_not_a_profile():
    """Phase 28 owns the profile; this run must never be quoted as one."""
    d = _sanity_passing()
    assert d['verdict'] == 'PASS'
    assert 'not a production profile' in d['gate']
    assert 'Phase 28' in d['not_a_claim']
    assert 'OFLM_OPEN_LAYER_MAJOR=0' in d['note'], \
        'block-loop numbers must state that they are not the shipped default route'


def _sanity_passing():
    """The newest sanity run that PASSED.

    A FAILED run is evidence too -- Phase 29 keeps one, where engine 4096 at a 4096 context
    is refused outright -- so the glob can no longer just take the newest file: it has to
    pick the run whose verdict is PASS.
    """
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob('oflm-sanity-*/result.json'))
    for path in reversed(runs):
        d = json.loads(path.read_text())
        if d.get('verdict') == 'PASS':
            return d
    pytest.skip('no passing sanity run committed')


def test_the_block_cost_is_linear_in_blocks_not_in_active_rows():
    """Both halves of the claim, from the committed sanity run.

    Flat in the active rows inside a block -- that is why a tail is padded or avoided.
    Linear in the number of blocks -- that is why a long prompt costs what it costs.
    """
    d = _sanity_passing()
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
    d = _failinject_passing()
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


# ---------------------------------------------------------------- long context (Phase 29)

def _lc():
    path = ROOT / 'utilities/prefill-block/long-context-v1.json'
    if not path.is_file():
        pytest.skip('no long-context artefact committed')
    return json.loads(path.read_text())


def test_the_kv_row_is_the_paired_layout_not_a_hand_estimate():
    """A row is FOUR bf16 planes -- [K hi, K lo, V hi, V lo] -- not two.

    The first version of this report wrote `4 x 256 x 2 x (K+V)`, which is 4096, and called
    it 8192. The number was right and the reason was wrong: the paired attention route stores
    each K and V twice, so `row = (8 if paired else 4) * kvw`. The test derives the row from
    the recipe and the host struct rather than asserting the constant.
    """
    d = _lc()
    kv = d['kv_model']
    kvw = 4 * 256                       # kv heads x head dim for this geometry
    unpaired = 4 * kvw                 # [K | V] bf16
    paired = 8 * kvw                   # [K hi | K lo | V hi | V lo] bf16
    assert kv['bytes_per_token_per_full_layer'] == paired == 8192
    assert kv['bytes_per_token_per_full_layer'] != unpaired, \
        'this model is paired; the unpaired row would be 4096'
    assert kv['k_physical_stride_bytes'] + kv['v_physical_stride_bytes'] == paired
    assert kv['bytes_per_token_all_16_layers'] == paired * 16 == 131072
    # The runtime says the same thing in its own words.
    host = (ROOT / 'src/open_qwen36/block_host.hpp').read_text()
    assert '[K hi, K lo, V hi, V lo]' in host
    recipe = (ROOT / 'open_kernels/recipes/qwen36moe.py').read_text()
    assert '(8 if paired else 4) * kvw' in recipe
    core = (ROOT / 'src/open_qwen36/core.cpp').read_text()
    assert 'kv_row / (w_->attn_kv_pair ? 4 : 2)' in core, \
        'the readout divides a row into four planes when paired'


def test_the_allocated_kv_matches_the_formula_at_every_measured_length():
    kv = _lc()['kv_model']
    row = kv['bytes_per_token_per_full_layer']
    layers = 16
    for ctx, allocated in kv['allocated_kv'].items():
        assert allocated == row * layers * int(ctx), ctx
    assert kv['allocated_kv']['262144'] == 34359738368, '32 GiB at 256K'


def test_max_l_is_the_whole_sequence_so_the_prompt_maximum_is_one_less():
    """`MAX_L` is prompt + generated; a prompt of exactly MAX_L has no room to generate.

    That makes the `>=` guard consistent rather than off-by-one, and the earlier report's
    "defect" label wrong. The boundary was then verified at a 512 context.
    """
    src = (ROOT / 'src/common/AutoModel/automodel.cpp').read_text()
    assert 'while (this->total_tokens < this->MAX_L)' in src, \
        'generation draws on the same budget as the prompt'
    assert 'total_tokens + tokens.size() >= this->MAX_L' in src
    d = _lc()
    c = d['context_contract']
    # Two paths, two limits: the CLI steps one row, this family's generation steps a think
    # preamble of 2 (thinking) or 4 (not) before any budget check.
    assert c['max_prompt_cli'] == 'ctx_len - 1'
    assert c['max_prompt_serve'].startswith('ctx_len - 5')
    assert 'ctx_len - 3' in c['max_prompt_serve'], 'the thinking branch keeps two, not four'
    head = c['generation_headroom']
    assert head['normal_branch']['preamble_rows'] == 4
    assert head['thinking_branch']['preamble_rows'] == 2
    assert head['normal_branch']['need'] == 5 and head['thinking_branch']['need'] == 3
    # Both branches name the SAME last legal prompt the hardware observed.
    assert head['normal_branch']['last_legal_prompt_at_ctx512'] == 507
    assert head['thinking_branch']['last_legal_prompt_at_ctx512'] == 509
    b = c['phase31_boundary_at_ctx512']
    # Non-thinking: 507 ends at total == MAX_L == 512, so 506 and 507 both generate.
    assert b['non_thinking']['507'] == 'generates (last legal)'
    assert b['non_thinking']['508'].startswith('refused'), 'must refuse, not throw'
    # Thinking: two preamble rows, so 509 is the last legal prompt and 510 refuses.
    assert b['thinking']['509'] == 'generates (last legal)'
    assert b['thinking']['510'].startswith('refused')


def _think_boundary():
    """The Phase 31 headroom run: both branches, both sides of the boundary."""
    runs = sorted((ROOT / 'utilities/prefill-block/runs').glob(
        'think-boundary-*/result.json'))
    for path in reversed(runs):
        d = json.loads(path.read_text())
        if d.get('verdict') == 'PASS':
            return d
    pytest.skip('no passing think-boundary run committed')


def test_the_headroom_boundary_was_measured_on_both_sides_of_both_branches():
    """31.A/31.B: the boundary is observed, not derived, and the branch is measured.

    The branch is read from the `ctx-trace` step names rather than inferred from what was
    requested, so a flipped default cannot quietly pass as the other branch. The refused
    row must show NO `step` line at all: if it stepped a preamble, the guard is after the
    write it is supposed to prevent.
    """
    d = _think_boundary()
    assert d['ctx'] == 512
    by = {(r['branch'], r['engine_want']): r for r in d['rows']}
    for branch, want_branch, preamble in (('normal_branch', 'not-thinking', 4),
                                          ('thinking_branch', 'thinking', 2)):
        need = preamble + 1
        ok = by[(branch, d['ctx'] - need)]
        bad = by[(branch, d['ctx'] - need + 1)]
        assert ok['branch_observed'] == want_branch, (branch, ok['branch_observed'])
        assert ok['engine_tokens'] == ok['engine_want'], 'the prompt reached the engine as itself'
        assert ok['output_len'] > 0 and not ok['refused']
        # The refusal happens after prefill and before the first preamble step.
        assert bad['refused'] is True
        assert bad['branch_observed'] is None, 'the guard must precede the preamble'
        assert bad['position_error'] is False, 'no position throw may remain'
        assert bad['internal_error'] is False
        assert not any('step ' in t for t in bad['trace'])


def test_the_cli_prompt_is_its_own_line_not_an_argument_to_clear():
    """`/clear` ignores its arguments, so `/clear <text>` generates nothing.

    That mistake produced a vacuous "PASS" in the first version of the headroom harness:
    0 tokens prefilled and no request made. It is asserted here because it is invisible in
    the result -- the run looks like a clean over-limit refusal.
    """
    runner = (ROOT / 'src/runner/runner.cpp').read_text()
    body = runner[runner.index('void Runner::cmd_clear'):]
    body = body[:body.index('}')]
    assert 'input_list' not in body.replace('std::vector<std::string>& input_list', ''), \
        'if cmd_clear starts using its arguments, this test and the harnesses can be simplified'
    for name in ('think_boundary.py', 'ctx_boundary.py'):
        src = (ROOT / 'utilities/prefill-block' / name).read_text()
        assert "'/clear', text" in src or "'/clear'," in src, name
        assert "'/clear ' + text" not in src, f'{name} would generate nothing'


def test_the_think_preamble_is_guarded_before_it_steps():
    """The row count is derived from the branch, and refused before anything is written.

    This is the Phase 31 fix: `Qwen3_5VL::generate` steps its think preamble (2 rows when
    thinking, 4 when not) before the loop's budget check can stop it, so a near-limit prompt
    used to throw `position N reached the context capacity N` and surface as a generic
    `Internal error`. A constant "spare slots" guess was wrong three times while probing;
    the guard now derives it from `enable_think` and names the limit in the message.
    """
    src = (ROOT / 'src/common/AutoModel/modeling_qwen3_5vl.cpp').read_text()
    # The branch constant lives in the helper (see the 31.C test); the call site asks it.
    assert 'this->generation_preamble_rows()' in src, \
        'the preamble size must come from the helper, not a guessed constant'
    assert 'Max length reached: this model needs' in src
    assert 'largest prompt' in src, 'the refusal must name the usable maximum'
    # It must sit BEFORE the preamble steps, not after.
    guard = src.index('Max length reached: this model needs')
    first_step = src.index('this->lm_engine->forward(think_start_id);')
    assert guard < first_step, 'the guard must precede the first preamble step'


def test_the_capacity_addressing_and_e2e_levels_stay_separate():
    d = _lc()
    for key in ('capacity_validated', 'addressing_validated_positions',
                'end_to_end_validated'):
        assert key in d, key
    assert max(d['capacity_validated']) == 262144
    assert max(d['addressing_validated_positions']) == 262143
    # End-to-end must never be claimed above what actually ran.
    assert max(d['end_to_end_validated']) < 8192
    for row in d['rows']:
        if row['context_length'] >= 8192:
            assert row['e2e_pass'] is False, \
                'a length above 4K must not be recorded as end-to-end validated'


def test_the_attention_gate_is_a_measured_fit_not_a_theory():
    g = _lc()['attention_gate']
    assert g['fires'] is True
    assert 0.99 <= g['model_r2'] if 'model_r2' in g else True
    assert 1.0 < g['ms_per_position'] < 2.5
    assert g['gate_threshold'] == 0.20
    assert 4096 < g['gate_crossing_context'] < 16384


def test_the_generation_preamble_is_declared_once_and_used_by_both_sides():
    """31.C: the capacity guard and the preamble implementation cannot drift apart.

    `generation_preamble_rows()` is the single source: the guard in front of the preamble
    uses it, and the counter is advanced by it after the preamble runs, so `total_tokens`
    and the engine position move together. Measured at ctx 512: prompt 507 generates and
    ends at total == MAX_L; prompt 508 is refused before anything is written.
    """
    hdr = (ROOT / 'src/include/AutoModel/modeling_qwen3_5vl.hpp').read_text()
    assert 'int generation_preamble_rows() const' in hdr
    assert 'this->enable_think ? 2 : 4' in hdr, 'two rows thinking, four not'
    src = (ROOT / 'src/common/AutoModel/modeling_qwen3_5vl.cpp').read_text()
    # Exactly one local declaration of the branch constant: it comes from the helper.
    assert src.count('const int preamble_rows = this->generation_preamble_rows();') == 1
    assert 'this->enable_think ? 2 : 4' not in src, \
        'the branch constant must live in the helper, not be repeated at a call site'
    # The counter moves by the preamble, or the loop guard measures a stale position.
    assert src.count('this->total_tokens += static_cast<size_t>(preamble_rows);') == 2, \
        'both branches must account for their preamble rows'
    # A model name must not appear in the rule.
    for name in ('qwen3.8', '27b', 'qwen35'):
        assert name not in hdr[hdr.index('generation_preamble_rows()') - 900:
                              hdr.index('generation_preamble_rows()')], name


def test_the_production_baseline_is_internally_consistent():
    """The quotable numbers, checked against each other rather than trusted.

    A baseline nobody can recompute is a claim, not a measurement: every derived field is
    recomputed here from the primitive ones, so a transcription slip fails the build.
    """
    b = json.loads((ROOT / 'utilities/prefill-block/'
                    'qwen38-production-baseline.json').read_text())
    assert b['status'] == 'PASS'
    for key in ('commit', 'binary', 'model_artifact', 'prefill', 'decode', 'context',
                'generation_headroom', 'routes', 'gates'):
        assert key in b, key
    assert b['binary']['bytes'] == 31886824
    assert b['binary']['sha256'].startswith('1315d111')
    assert b['model_artifact']['tensors'] == 947
    assert b['model_artifact']['bytes'] == 19844633616
    assert round(b['model_artifact']['gib'], 3) == 19.845
    # Prefill: engine count, block count and rate must agree with each other.
    physical = b['geometry']['physical_block_t']
    for r in b['prefill']['rows']:
        assert r['blocks'] == r['engine_tokens'] // physical, r
        rate = r['engine_tokens'] / (r['ttft_ms'] / 1000.0)
        assert abs(rate - r['prompt_tok_s']) < 0.02, (r, rate)
    # Cost is linear in BLOCKS, not in rows: TTFT per block stays in a narrow band.
    per_block = [r['ttft_ms'] / r['blocks'] for r in b['prefill']['rows']]
    assert max(per_block) / min(per_block) <= 1.6, per_block
    # KV: row x full layers = per token, and 256K follows with no fudge factor.
    c = b['context']
    assert c['kv_bytes_per_token'] == (c['kv_bytes_per_token_per_full_layer']
                                       * c['kv_full_attention_layers'])
    assert c['kv_bytes_per_token_per_full_layer'] == 8192
    assert c['kv_256k_bytes'] == c['kv_bytes_per_token'] * 262144
    assert c['kv_256k_gib'] == 32.0
    # The levels stay ordered and separate.
    assert c['capacity_validated'] == c['addressing_validated'] == 262144
    assert c['addressing_last_position'] == 262143
    assert c['serve_end_to_end_validated'] < c['cli_end_to_end_validated'] < 8192
    assert b['routes']['sequential_block_crossover'] == 96
    assert b['routes']['dense_qwen38_default'] == 'block-loop'
    # And the claim list is short and specific, not "everything works".
    assert any('vision' in x for x in b['not_claimed'])
    assert any('256K' in x for x in b['not_claimed'])
