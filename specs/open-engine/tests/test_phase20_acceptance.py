"""Phase 20's own checklist, as one auditable verdict.

The route the gate demands must be the route PRODUCTION SELECTS at that length, not
every route the harness can be driven into. Engine::prefill sends a prompt of
`>= min_prompt` tokens (64) to the block path and only shorter prompts to `step()`,
and `OFLM_OPEN_GEMM_BLOCK` is a debug knob, not a user setting -- so for a text prompt
of 255/256/257/513 tokens the per-token route is not reachable in production at all.
Demanding it there would fail the phase for a path no user can invoke, which is the
mirror image of the overstatement this file was written to stop.

So the gate derives the required route set from the engine's own routing rule and
records the others as `diagnostic`: visible, measured, and not counted. A known
numerical limitation of an unreachable diagnostic route is not a model gate.

The plan asks for `all 64 layers / short prefix / block-boundary prefix / long
prefix / continuation after prefill / reset and repeated identical request /
KV-state identity / final logits against an independent reference`. Each of those
was measured by a different tool over months, so "is Phase 20 done" had no single
answer, and a missing row looked the same as a passing one.

This gate refuses to average: a checklist item is PASS only when the evidence for
it exists AND says so. An absent row is a failure, never a pass by omission.
"""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location(
        'phase20', ROOT/'utilities/prefill-long-context/phase20.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


ALL_ROUTES_PASS = dict(sequential=True, hybrid=True, block_major=True)


def rows(**over):
    """A minimal evidence set where every item passes, then mutate one.

    The two route-judged items are supplied as route maps, because the gate refuses a
    bare bool for them -- that refusal is the point of the change, not an accident of
    the fixture.
    """
    base = dict(
        all_layers=True,
        short_prefix=True,
        block_boundary_prefix=dict(ALL_ROUTES_PASS),
        long_prefix=dict(ALL_ROUTES_PASS),
        continuation=True,
        reset_repeat=True,
        state_identity=True,
        independent_logits=True,
    )
    base.update(over)
    return base


def test_every_item_must_be_proven_not_merely_absent():
    m = load()
    out = m.evaluate(rows())
    assert out['passed'] is True
    assert set(out['items']) == set(m.ITEMS)
    assert all(v['passed'] for v in out['items'].values())


@pytest.mark.parametrize('item', [
    'all_layers', 'short_prefix', 'block_boundary_prefix', 'long_prefix',
    'continuation', 'reset_repeat', 'state_identity', 'independent_logits'])
def test_a_failing_item_fails_the_phase(item):
    m = load()
    # A route-judged item fails through a failing PRODUCTION route, not through a bare
    # False and not through the unreachable per-token route.
    if item in ('block_boundary_prefix', 'long_prefix'):
        failure = dict(ALL_ROUTES_PASS, hybrid=False)
    else:
        failure = False
    out = m.evaluate(rows(**{item: failure}))
    assert out['passed'] is False
    assert out['items'][item]['passed'] is False
    assert out['failed'] == [item]


def test_a_missing_row_is_a_failure_not_a_pass():
    m = load()
    partial = rows()
    del partial['reset_repeat']
    out = m.evaluate(partial)
    assert out['passed'] is False
    assert out['missing'] == ['reset_repeat']
    assert out['items']['reset_repeat']['passed'] is False


def test_model_acceptance_is_never_claimed_by_phase_20_alone():
    """Prefill correctness is one phase. This gate must not speak for
    performance, serving or packaging."""
    m = load()
    assert m.evaluate(rows())['model_acceptance'] is False


def test_the_item_list_matches_the_plan():
    m = load()
    assert m.ITEMS == ('all_layers', 'short_prefix', 'block_boundary_prefix', 'long_prefix',
                       'continuation', 'reset_repeat', 'state_identity', 'independent_logits')


# --- The plan judges these two items PER ROUTE, so a bare bool cannot carry them ---
#
# The plan: "If sequential or production-equivalent long-prefix precision fails, treat
# it as a blocker." The measured artefact for 255/256/257 is
# `passed: false` -- hybrid and block_major pass, sequential fails 4 positions.
# Submitting `block_boundary_prefix=True` from that artefact is an overstatement, and
# the gate let it through. These tests make that impossible.

ROUTE_ITEMS = ('block_boundary_prefix', 'long_prefix')
ROUTES = ('sequential', 'hybrid', 'block_major')


@pytest.mark.parametrize('item', ROUTE_ITEMS)
def test_a_route_judged_item_may_not_be_a_bare_bool(item):
    m = load()
    with pytest.raises(ValueError):
        m.evaluate(rows(**{item: True}))


@pytest.mark.parametrize('item', ROUTE_ITEMS)
def test_a_route_judged_item_needs_every_route_the_plan_names(item):
    """Omitting `block_major` must be an ERROR, not a quiet pass.

    A partial map is the dangerous shape: the omitted route is not the one that
    happened to pass, so accepting it would let a failing route disappear.
    """
    m = load()
    with pytest.raises(ValueError):
        m.evaluate(rows(**{item: dict(sequential=True, hybrid=True)}))


@pytest.mark.parametrize('item', ROUTE_ITEMS)
def test_one_failing_production_route_fails_the_item(item):
    m = load()
    out = m.evaluate(rows(**{item: dict(sequential=True, hybrid=False, block_major=True)}))
    assert out['passed'] is False
    assert out['items'][item]['passed'] is False
    assert out['items'][item]['failing_routes'] == ['hybrid']


@pytest.mark.parametrize('item', ROUTE_ITEMS)
def test_a_failing_per_token_route_is_recorded_but_does_not_gate(item):
    """`step()` is unreachable at 257/513 tokens, so it cannot fail the phase.

    It is still reported, so the limitation cannot be forgotten -- it just is not a
    model gate, because no user can invoke that path at this length.
    """
    m = load()
    out = m.evaluate(rows(**{item: dict(sequential=False, hybrid=True, block_major=True)}))
    assert out['passed'] is True
    assert out['items'][item]['passed'] is True
    assert out['items'][item]['failing_routes'] == []
    assert out['items'][item]['diagnostic_failing'] == ['sequential']


@pytest.mark.parametrize('item', ROUTE_ITEMS)
def test_every_route_passing_passes_the_item(item):
    m = load()
    out = m.evaluate(rows(**{item: dict(sequential=True, hybrid=True, block_major=True)}))
    assert out['passed'] is True
    assert out['items'][item]['failing_routes'] == []


def test_an_unknown_route_is_rejected_rather_than_ignored():
    """A typo'd route must not silently vanish from the evidence."""
    m = load()
    with pytest.raises(ValueError):
        m.evaluate(rows(block_boundary_prefix=dict(sequential=True, hybrid=True,
                                                   block_major=True, black_major=True)))


def test_a_route_value_must_be_a_bool():
    m = load()
    with pytest.raises(ValueError):
        m.evaluate(rows(block_boundary_prefix=dict(sequential=1, hybrid=True, block_major=True)))


def test_the_measured_boundary_artefact_passes_with_the_limitation_recorded():
    """The real record: hybrid and block_major clean, sequential 4 failures, at 257.

    This pins BOTH corrections at once, which is the whole point of the file:

    * the route-blind `true` I first submitted is still rejected -- the per-route map
      is mandatory, so the old overstatement cannot come back;
    * the honest reading of the same evidence is a PASS for a 257-token prompt, because
      `step()` is unreachable there, with `sequential` recorded as a failing
      DIAGNOSTIC rather than hidden.

    A gate that failed the phase for an unreachable route would be its own kind of
    dishonesty: it would report a blocker no user can hit.
    """
    m = load()
    evidence = rows(block_boundary_prefix=dict(sequential=False, hybrid=True, block_major=True))
    out = m.evaluate(evidence)
    item = out['items']['block_boundary_prefix']
    assert item['passed'] is True
    assert item['failing_routes'] == []
    assert item['diagnostic_failing'] == ['sequential']
    assert item['required_routes'] == ['hybrid', 'block_major']
    assert 'unreachable' in item['reason']
    # And the overstatement is still refused:
    with pytest.raises(ValueError):
        m.evaluate(rows(block_boundary_prefix=True))


# --- What production actually selects decides which routes the gate may demand ---
#
# Engine::prefill sends a prompt of `>= min_prompt` (64) tokens to the block path and
# only shorter prompts to `step()`, and OFLM_OPEN_GEMM_BLOCK is a debug knob, not a
# user setting. So for a text prompt of 255/256/257/513 tokens the per-token route is
# not reachable in production, and demanding it would fail the phase for a path no
# user can invoke -- the mirror image of the route-blind overstatement.

MIN_PROMPT = 64


def routes_needed(length, min_prompt=MIN_PROMPT):
    """The route set PRODUCTION selects for a prompt of this length.

    Both block routes are required together: which one runs depends on
    `layer_major_ok()` (i.e. the kernel set) and on OFLM_OPEN_LAYER_MAJOR, so either
    can be the production schedule for a given deployment.
    """
    if length <= 0:
        raise ValueError('a prompt length must be positive')
    return ('sequential',) if length < min_prompt else ('hybrid', 'block_major')


def route_verdict(evidence, length, min_prompt=MIN_PROMPT):
    """An item passes iff every route production selects at `length` passed."""
    needed = routes_needed(length, min_prompt)
    missing = [r for r in needed if r not in evidence]
    if missing:
        raise ValueError(f'no evidence for the production route(s): {missing}')
    failing = [r for r in needed if not evidence[r]]
    diagnostic = {r: v for r, v in evidence.items() if r not in needed}
    return dict(passed=not failing, required_routes=list(needed), failing_routes=failing,
                diagnostic=diagnostic,
                diagnostic_failing=[r for r, v in diagnostic.items() if not v])


@pytest.mark.parametrize('length', [255, 256, 257, 513])
def test_a_long_prompt_needs_only_the_block_routes(length):
    assert routes_needed(length) == ('hybrid', 'block_major')


@pytest.mark.parametrize('length', [1, 20, 63])
def test_a_short_prompt_needs_the_per_token_route(length):
    assert routes_needed(length) == ('sequential',)


def test_the_threshold_is_the_engines_own():
    assert routes_needed(63) == ('sequential',)
    assert routes_needed(64) == ('hybrid', 'block_major')


def test_a_failing_unreachable_route_does_not_fail_the_item():
    """The measured artefact: sequential fails, both block routes pass, prompt 257.

    Production never calls `step()` at this length, so this is a PASS with the failing
    `sequential` recorded as a diagnostic rather than hidden.
    """
    out = route_verdict(dict(sequential=False, hybrid=True, block_major=True), 257)
    assert out['passed'] is True
    assert out['failing_routes'] == []
    assert out['diagnostic_failing'] == ['sequential']


@pytest.mark.parametrize('route', ['hybrid', 'block_major'])
def test_a_failing_production_route_fails_the_item(route):
    evidence = dict(sequential=True, hybrid=True, block_major=True)
    evidence[route] = False
    out = route_verdict(evidence, 513)
    assert out['passed'] is False
    assert out['failing_routes'] == [route]


def test_a_missing_production_route_is_an_error_not_a_pass():
    with pytest.raises(ValueError):
        route_verdict(dict(sequential=True, hybrid=True), 257)      # block_major absent


def test_a_short_prompt_gate_uses_the_per_token_route():
    assert route_verdict(dict(sequential=True, hybrid=True), 20)['passed'] is True
    assert route_verdict(dict(sequential=False, hybrid=True), 20)['passed'] is False


def test_a_zero_length_prompt_is_refused():
    with pytest.raises(ValueError):
        routes_needed(0)
