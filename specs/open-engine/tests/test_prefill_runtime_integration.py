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
    # The crossover now comes from `prefill::kDenseBlockLoop`; that it equals 96, and that
    # the baseline agrees, is asserted by the threshold test above rather than by a literal
    # here, so a shared constant is possible at all.
    assert 'block_loop_crossover = prefill::kDenseBlockLoop' in src, \
        'the sequential/block crossover must come from the shared constant'
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


def test_the_documented_crossover_matches_the_baseline_and_the_header():
    """One number, three places: the C++ constant, the baseline, and the public header.

    The header said 64 for years while the code used 96. Both numbers exist -- 64 is the
    layer-major crossover the MoE families measured, 96 is the dense block-loop one -- but
    presenting 64 as the dense default made a 64-95 token prompt look block-eligible.
    """
    hdr = (ROOT / 'src/open_qwen36/prefill_thresholds.hpp').read_text()
    assert 'inline constexpr size_t kDenseBlockLoop = 96;' in hdr
    assert 'inline constexpr size_t kLayerMajor = 64;' in hdr
    # The baseline records the production dense value, and it must be the same number.
    b = baseline()
    assert b['routes']['sequential_block_crossover'] == 96
    assert b['routes']['layer_major_crossover'] == 64
    # The implementation takes the boundary from the constant rather than repeating it.
    eng = (ROOT / 'src/open_qwen36/engine.cpp').read_text()
    assert 'const size_t block_loop_crossover = prefill::kDenseBlockLoop;' in eng, \
        'the crossover must come from the shared constant, not a second literal 96'
    # And the public header must not still claim the MoE number for the dense route.
    public = (ROOT / 'src/open_qwen36/engine.hpp').read_text()
    assert 'kDenseBlockLoop` (96)' in public
    assert 'GEMM_BLOCK_MIN (64)' not in public, \
        'the header described 64 as the dense default; that is the bug this pins'


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


# ---------------------------------------------------------------- the open gates


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


# ---------------------------------------------------------------- long context (Phase 29)


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
