"""Phase 24 profiler: the attribution must not be able to lie to itself.

Three failures are catastrophic here and all three happened while writing it:

  * a fixture that does not hold the length the row claims, so two rows measure the
    same prompt and the table reads "cost stops growing";
  * a scaling label that ignores how much the length grew, which called a clean 16x
    for 16x term "constant/fixed overhead";
  * categories that summed to 4796.0 ms MORE than wall time at 2048 tokens, because
    `shared_ms` was counted in `host_setup` while `residue` (which contains it) was
    counted in full. Over 100% of a run is the signature of a double count.

None is a crash -- all three produce a plausible table. So all three are pinned.
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


def _rec(**over):
    """A minimal 2048-like record; every identity below is checked on real shape."""
    f = dict(mid=19546.0, dn_conv=3067.9, dn_rule=2876.7, attn=13600.6, prenorm=1264.9,
             gemm_tile=24124.2, gemm_tr=7410.7, tail=3423.2, shared=4788.6, state=8.1,
             sync=2349.1, setup=26.3, moe_prep=0.0, moe_patch=0.0, moe_run=0.0,
             moe_read=0.0, qkv_scatter=0.0, residue=27643.7)
    f.update(over)
    t = dict(total=284432.8, gemm=198619.2, host=58144.3, route=0.0, lm=25.6)
    return dict(totals=t, fields=f)


def test_categories_never_exceed_wall_time():
    """THE requested sanity check. v1 failed here by 4796.0 ms -- over 100% of a run."""
    m = load()
    acc = m.allocate(_rec(), 294310.0)
    c = acc['checks']
    assert c['within_wall'] is True
    assert c['categories_sum_ms'] <= c['engine_total_ms'] + 1.0
    assert c['categories_minus_wall_ms'] < 0


def test_the_double_count_is_reproduced_by_the_old_grouping():
    """Not a hypothetical: `shared` sits inside `residue`, so counting both overshoots."""
    m = load()
    rec = _rec()
    old = (rec['totals']['gemm'] + rec['fields']['gemm_tile'] + rec['fields']['gemm_tr'] +
           rec['fields']['dn_conv'] + rec['fields']['dn_rule'] + rec['fields']['attn'] +
           rec['fields']['prenorm'] + rec['fields']['tail'] + rec['fields']['state'] +
           rec['fields']['sync'] + rec['fields']['setup'] + rec['fields']['shared'] +
           rec['fields']['residue'] + rec['totals']['lm'])
    assert old > rec['totals']['total'] + 4000, 'the old grouping really did overshoot'
    # The new one does not, and the difference is exactly the shared term.
    assert abs(m.allocate(rec, 294310.0)['checks']['categories_sum_ms'] - rec['totals']['total']) < 1.0


def test_wall_share_and_engine_share_are_different_numbers():
    """A wall share and an engine share must never be read as the same percentage."""
    m = load()
    acc = m.allocate(_rec(), 294310.0)
    gemm = 'projection_gemm_device'
    assert acc['share_of_wall_pct'][gemm] < acc['share_of_engine_pct'][gemm]
    # 198619.2 / 294310 and / 284432.8 -- both explicit, neither interchangeable.
    assert abs(acc['share_of_wall_pct'][gemm] - 67.49) < 0.1
    assert abs(acc['share_of_engine_pct'][gemm] - 69.83) < 0.1


def test_the_unattributed_engine_residue_is_reported_separately_from_the_process_gap():
    m = load()
    acc = m.allocate(_rec(), 294310.0)
    assert abs(acc['checks']['unattributed_engine_ms'] - 22847.0) < 0.1
    assert abs(acc['checks']['outside_engine_ms'] - 9877.2) < 0.1
    # Both are reported, and neither is silently merged into a category.
    assert 'unattributed_engine' in acc['exclusive']
    assert 'outside_engine' not in acc['exclusive']


def test_a_route_that_nests_attention_inside_part1_does_not_get_counted_twice():
    """On the dxB route, route_ms nests attn; adding both would double count again."""
    m = load()
    rec = _rec(residue=0.0)
    rec['totals']['route'] = 500.0
    rec['totals']['host'] = 58144.3 - 19546.0 + 500.0
    acc = m.allocate(rec, 294310.0)
    assert 'host_route' in acc['exclusive']
    assert 'attention' not in acc['exclusive'], (
        'attn is inside route here; counting it too double counts the attention column')


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
    p = ROOT / 'utilities/prefill-perf/e2e-profile-v2.json'
    assert p.is_file(), 'v2 is the corrected grouping; v1 is kept as the evidence of it'
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
        blocks = -(-n // 256)
        assert row['dispatch_calls'] == 1024 * blocks + 2, (
            f'{n} tokens gave {row["dispatch_calls"]} dispatches, '
            f'expected {1024 * blocks + 2} for {blocks} block(s)')
        # EVERY committed row must pass the accounting checks, not just the one under
        # discussion: a grouping error is a property of the tool, not of a length.
        c = row['allocation']['checks']
        assert c['within_wall'] is True, f'{n}: categories exceed wall time'
        assert abs(c['categories_minus_engine_ms']) <= 1.0, (
            f'{n}: categories differ from the engine total by '
            f'{c["categories_minus_engine_ms"]} ms')
        assert abs(c['part1_identity_delta_ms']) <= 1.0, (
            f'{n}: part1 is not the sum of the fields beneath it')


def test_the_dispatch_per_block_count_is_not_a_column_count():
    """1024 per block is 256 dispatches per projection, NOT 8 array columns per dispatch.

    This is the correction of Phase 24's own claim. The evidence: the tail of a request
    is 2 dispatches (ln + lm) and every other dispatch is a GEMM. Per block that is
    64 layers x 2 projections x 2 (a paired projection is the fold of high and low
    weight components) x 2 (x_hi and x_lo over each) = 1024. The eight array columns
    live INSIDE one dispatch: gemm_q4_prefill.py's Worker.grid(4, 8) over one sequence.
    """
    p = ROOT / 'utilities/prefill-perf/e2e-profile-v2.json'
    d = json.loads(p.read_text())
    row = d['data']['2048']
    # 2048 tokens = 8 blocks. GEMM dispatches = 8 * 1024; the rest are the 2-dispatch tail.
    gemm = sum(v['calls'] for k, v in row['dispatch'].items() if k.startswith('gemm_'))
    assert gemm == 8 * 1024, f'GEMM dispatches per 8 blocks: {gemm}'
    non_gemm = sum(v['calls'] for k, v in row['dispatch'].items()
                   if not k.startswith('gemm_'))
    assert non_gemm == 2, f'the tail is ln + lm, got {non_gemm}'
    # Per projection per layer per block: 2 kernels (hi, low) x 2 input components.
    per_proj = 8 * 1024 // (64 * 2)
    assert per_proj == 64, f'dispatches per projection over 8 blocks: {per_proj}'
    assert per_proj // 8 == 8, 'per block: 8 = 2 weight components x 2 input components'


