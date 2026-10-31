"""Phase 20's own checklist, as one auditable verdict.

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


def rows(**over):
    """A minimal evidence set where every item passes, then mutate one."""
    base = dict(
        all_layers=True,
        short_prefix=True,
        block_boundary_prefix=True,
        long_prefix=True,
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
    out = m.evaluate(rows(**{item: False}))
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
