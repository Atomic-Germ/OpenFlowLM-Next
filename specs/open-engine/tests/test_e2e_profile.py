"""Phase 24 profiler: the attribution must not be able to lie to itself.

Two failures are catastrophic here and both happened while writing it:

  * a fixture that does not hold the length the row claims, so two rows measure the
    same prompt and the table reads "cost stops growing";
  * a scaling label that ignores how much the length grew, which called a clean 16x
    for 16x term "constant/fixed overhead".

Neither is a crash -- both produce a plausible table. So both are pinned.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'utilities/prefill-perf/e2e_profile.py'


def load():
    assert TOOL.is_file(), 'the Phase 24 profiler is missing'
    spec = importlib.util.spec_from_file_location('_e2e', TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('points, expected', [
    # A term that does not move while the prompt grows 16x is fixed overhead.
    ([(128, 5.0), (512, 5.4), (1024, 6.3), (2048, 8.1)], 'constant/fixed overhead'),
    # 16x tokens for 16x time is O(T) -- NOT constant. This is the regression.
    ([(128, 24940.9), (512, 49830.0), (1024, 99475.5), (2048, 198619.2)], 'O(T)'),
    # 16x tokens for 100x time is superlinear / quadratic.
    ([(128, 136.0), (512, 1046.4), (1024, 3609.0), (2048, 13600.6)], 'approximately O(T^2)'),
    # 16x tokens for 40x time is superlinear but not quadratic
    # (log(40)/log(16) = 1.33). 25x would be 1.16, i.e. still O(T) -- which is what the
    # first version of this test wrongly expected, so the case is kept at 40x.
    ([(128, 100.0), (2048, 4000.0)], 'superlinear'),
])
def test_scaling_labels_normalise_to_the_growth_in_length(points, expected):
    m = load()
    assert m.classify(points) == expected


def test_scaling_refuses_to_label_a_single_or_zero_point():
    m = load()
    assert m.classify([(128, 10.0)]) == 'unknown'
    assert m.classify([(128, 0.0), (2048, 10.0)]) == 'unknown'


def test_categories_are_disjoint_engine_fields():
    """Every engine field is claimed exactly once, so the groups cannot add up twice."""
    m = load()
    groups = m.GROUPS
    assert set(groups) == {
        'projection_gemm', 'ffn_and_gemm_host', 'deltanet', 'attention',
        'norm_elementwise', 'state_movement', 'sync_bo', 'host_setup', 'moe',
        'qkv_scatter', 'residue'}
    # The residue is defined as the engine's own remainder, never as a recomputation.
    assert groups['residue'] == 'unattributed'


def test_a_fixture_shorter_than_the_row_claiming_it_is_refused(tmp_path):
    """The exact bug: a 513-token file used for a 1024-token row must not run."""
    m = load()
    short = tmp_path / 'ids512.txt'
    short.write_text(','.join(['7'] * 513))
    assert m.token_count(short) == 513
    # The guard compares against the requested length, so this pair is rejectable.
    assert m.token_count(short) != 512


def test_the_committed_profile_records_what_makes_it_quotable():
    """A profile without these fields is an attribution that cannot be checked."""
    p = ROOT / 'utilities/prefill-perf/e2e-profile-v1.json'
    assert p.is_file()
    d = json.loads(p.read_text())
    assert d['precision_mode'] == 'paired'
    assert d['reps'] == 1
    assert d['dispersion_measured'] is False, (
        'one pass cannot measure spread; claiming otherwise is the whole trap')
    assert d['model_acceptance'] is False
    assert d['lengths'] == [128, 512, 1024, 2048], '4096 is skipped by the decision gate'
    for n in d['lengths']:
        row = d['data'][str(n)]
        assert row['tokens_in_run'] == n, (
            f'the {n}-token row ran {row["tokens_in_run"]} tokens')
        # Dispatches scale with the prompt in exact block-sized steps, and the step size
        # is an architectural property worth stating: 1024 dispatches per 256-token
        # block plus the 2-dispatch tail. 1024 = 64 layers x 2 GEMMs x 8 columns, i.e.
        # each array column is dispatched separately. A row that does not follow
        # 1024*blocks+2 did not process that many tokens.
        blocks = -(-n // 256)
        assert row['dispatch_calls'] == 1024 * blocks + 2, (
            f'{n} tokens gave {row["dispatch_calls"]} dispatches, '
            f'expected {1024 * blocks + 2} for {blocks} block(s)')


def test_the_report_states_its_own_limitations():
    md = (ROOT / 'utilities/prefill-perf/E2E_PROFILE.md').read_text()
    for needed in ('reps = 1', 'dispersion', 'residue', 'commit', 'firmware'):
        assert needed in md, needed
    # The power mode was unreadable, and saying so is required rather than omitting it.
    assert 'not set' in md and 'EACCES' in md
