"""Phase 21 must turn the dispatch table into efficiency, not just into milliseconds.

The lesson these pin: a "host blocked on GEMM" column reads as a synchronization
problem until the FLOPs are divided by it. On Qwen3.8 the GEMMs sustain ~1.9 TFLOP/s,
i.e. ~12 % of a 16-TOPS part, so the cost is inside the GEMM. A tool that reported only
the milliseconds would have sent the next agent to fix the wrong layer of the stack.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location(
        'dispatch', ROOT / 'utilities' / 'prefill-perf' / 'dispatch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = ('  gemm-prompt [0,513) t_real=513: 2000.0 ms (GEMM 1200.0, host 400.0, per-token 0.0, '
         'lm_head 25.8)')
TABLE = '''      gemm_n16384_k5120         6 calls    137.1 ms total  22.857 mean  20.849 min
      gemm_n16384_k5120_low     6 calls    142.5 ms total  23.748 mean  21.061 min
      gemm_n34816_k5120         6 calls    281.0 ms total  46.836 mean  41.418 min
      gemm_n34816_k5120_low     6 calls    279.9 ms total  46.645 mean  43.752 min
      gemm_n5120_k17408         6 calls    135.5 ms total  22.577 mean  19.884 min
      gemm_n5120_k17408_low     6 calls    144.5 ms total  24.087 mean  21.320 min
      gemm_n5120_k6144          6 calls     57.1 ms total   9.512 mean   7.902 min
      gemm_n5120_k6144_low      6 calls     57.5 ms total   9.580 mean   7.898 min
      lm                        1 calls     25.4 ms total  25.431 mean  25.431 min
      ln                        1 calls      0.3 ms total   0.284 mean   0.284 min'''


def test_the_stage_line_yields_tokens_and_the_gemm_host_split():
    d = load()
    s = d.parse_stage(STAGE)
    assert s['t_real'] == 513 and s['n0'] == 0 and s['n1'] == 513
    assert s['gemm_ms'] == 1200.0 and s['host_ms'] == 400.0


def test_a_run_without_a_stage_line_is_none_not_an_error():
    """The per-token route emits no stage line; that is a finding, not a crash."""
    d = load()
    assert d.parse_stage('prefill 513 tokens: 232469 ms (453 ms/token)') is None


def test_every_dispatch_row_is_parsed_including_non_gemm_kernels():
    d = load()
    rows = d.parse_dispatches(TABLE)
    assert len(rows) == 10
    assert [r['name'] for r in rows][:2] == ['gemm_n16384_k5120', 'gemm_n16384_k5120_low']
    assert rows[-1] == dict(name='ln', calls=1, total_ms=0.3, mean_ms=0.284, min_ms=0.284)


def test_a_non_gemm_kernel_carries_no_flops_and_is_not_scored():
    d = load()
    rows = d.parse_dispatches(TABLE)
    eff = d.efficiency(rows, 256)
    assert all(r['name'] not in ('lm', 'ln') for r in eff)
    assert len(eff) == 8


def test_flops_are_two_n_k_tokens_and_the_low_companion_counts_too():
    d = load()
    assert d.flops('gemm_n16384_k5120', 256) == 2 * 16384 * 5120 * 256
    assert d.flops('gemm_n16384_k5120_low', 256) == 2 * 16384 * 5120 * 256
    assert d.flops('lm', 256) is None


def test_efficiency_is_on_one_call_not_the_aggregate():
    """42.9 GFLOP in a 22.857 ms MEAN call is 1.88 TFLOP/s.

    Not 42.9 GFLOP over the 137.1 ms of six calls -- that would be the aggregate, and it
    is exactly the mistake that produced a 200x understatement in the first version.
    """
    d = load()
    rows = d.parse_dispatches(TABLE)
    k = {r['name']: r for r in d.efficiency(rows, 256)}['gemm_n16384_k5120']
    expected = (2 * 16384 * 5120 * 256 / 1e12) / (22.857 / 1e3)
    assert k['tflops'] == pytest.approx(expected, rel=1e-6)
    assert k['tflops'] == pytest.approx(1.88, rel=0.02)
    assert k['gflop_per_call'] == pytest.approx(42.9, rel=0.02)


def test_each_call_carries_one_compiled_block_not_the_whole_prompt():
    """The bug this pins: a 513-token prompt is THREE 256-wide calls per GEMM.

    Dividing the prompt's FLOPs by every call's aggregate time reports 0.011 TFLOP/s
    for a GEMM that sustains ~2 TFLOP/s per dispatch -- a 200x understatement that
    would have argued against the fix the measurement actually supports. Efficiency is
    per CALL against the compiled block width, and the prompt's useful throughput is a
    separate number.
    """
    d = load()
    rows = d.parse_dispatches(TABLE)
    # 96 calls of a 256-wide block: per-call FLOPs must not include the 513 either.
    per_call = d.efficiency(rows, 256)
    k = {r['name']: r for r in per_call}['gemm_n16384_k5120']
    assert k['calls'] == 6
    assert k['gflop_per_call'] == pytest.approx(2 * 16384 * 5120 * 256 / 1e9, rel=1e-9)


def test_prompt_throughput_is_reported_separately_and_is_smaller():
    """Useful throughput folds in the paired doubling and the block padding."""
    d = load()
    rows = d.parse_dispatches(TABLE)
    p = d.prompt_throughput(rows, tokens=513, block=256)
    # 513 real tokens occupy ceil(513/256)=3 blocks of 256 slots = 768 token-slots.
    assert p['blocks'] == 3
    assert p['padded_tokens'] == 768
    assert p['padding_factor'] == pytest.approx(768 / 513)
    assert p['useful_tflops'] > 0
    # useful < achieved, because only 513/768 of the work is real and half the products
    # are the paired correction.
    assert p['useful_tflops'] < p['achieved_tflops'] / 2


def test_a_zero_block_width_is_refused():
    d = load()
    with pytest.raises(ValueError):
        d.prompt_throughput(d.parse_dispatches(TABLE), tokens=513, block=0)


def test_peak_fraction_states_utilisation_against_an_explicit_peak():
    d = load()
    rows = d.parse_dispatches(TABLE)
    k = {r['name']: r for r in d.efficiency(rows, 256, peak=16.0)}['gemm_n34816_k5120']
    assert k['peak_fraction'] == pytest.approx(k['tflops'] / 16.0)


def test_a_zero_token_count_is_refused():
    d = load()
    with pytest.raises(ValueError):
        d.efficiency(d.parse_dispatches(TABLE), 0)


def test_the_paired_low_products_are_split_out():
    """The plan's first suspect is the four-product route; this makes it a number."""
    d = load()
    p = d.paired_split(d.parse_dispatches(TABLE), 256)
    assert p['high_ms'] == pytest.approx(137.1 + 281.0 + 135.5 + 57.1)
    assert p['low_ms'] == pytest.approx(142.5 + 279.9 + 144.5 + 57.5)
    assert p['low_share'] == pytest.approx(p['low_ms'] / p['total_ms'])


def test_budget_explains_the_total_and_reports_coverage():
    d = load()
    s = d.parse_stage(STAGE)
    b = d.budget(s, dispatch_ms=1160.0)
    assert b['total_ms'] == 2000.0
    assert b['unexplained_ms'] == pytest.approx(400.0)
    assert b['coverage'] == pytest.approx(0.8)


def test_a_budget_without_a_stage_line_is_an_error():
    d = load()
    with pytest.raises(ValueError):
        d.budget(None, 1.0)


def test_coverage_below_the_phase_threshold_fails_and_above_passes():
    d = load()
    low = d.verdict(d.budget(d.parse_stage(STAGE), 1160.0))
    assert low['covered'] is False
    assert low['coverage_min'] == 0.90
    good_stage = d.parse_stage(
        '  gemm-prompt [0,256) t_real=256: 1000.0 ms (GEMM 600.0, host 350.0, per-token 0.0, lm_head 0.0)')
    assert d.verdict(d.budget(good_stage, 600.0))['covered'] is True


def test_a_dispatch_column_that_disagrees_with_the_gemm_column_is_flagged():
    """The two come from different clocks; a gap is a measurement bug, not a win."""
    d = load()
    s = d.parse_stage(STAGE)
    agree = d.verdict(d.budget(s, 1200.0))
    assert agree['dispatch_agree'] is True
    disagree = d.verdict(d.budget(s, 600.0))
    assert disagree['dispatch_agree'] is False
    assert disagree['dispatch_gap_ms'] == pytest.approx(600.0)


def test_an_unparseable_table_says_what_to_do(tmp_path):
    d = load()
    log = tmp_path / 'r.log'
    log.write_text('prefill 513 tokens: 232469 ms (453 ms/token)\n')
    with pytest.raises(SystemExit) as e:
        d.run(type('A', (), dict(log=log, tokens=513, coverage_min=0.9, out=None))())
    assert 'OFLM_OPEN_DISPATCH_LOG' in str(e.value)
