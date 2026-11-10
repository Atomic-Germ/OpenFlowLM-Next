"""Acceptance gates for the matched OpenMP measurement driver."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
try:
    spec = importlib.util.spec_from_file_location('omp_policy', ROOT / 'utilities/prefill-block/omp_policy.py')
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
except ModuleNotFoundError as exc:
    if exc.name in ('fcntl', 'numpy', 'tokenizers'):
        pytest.skip(f'{exc.name} required by the NPU measurement driver', allow_module_level=True)
    raise


def samples():
    return [dict(rep=i, warmup=i < 0, arm=arm, prefill_ms=ms, decode_ms=450,
                 rss_kib=30000000, ids=[1, 2])
            for i in range(-1, 5) for arm, ms in [('before', 100), ('after', 85)]]


def test_warmup_excluded_and_all_pairs_required():
    rows = samples()
    rows[1]['prefill_ms'] = 1000
    result = M.summary(rows)
    assert result['pass'] and result['prefill_wins'] == 5
    assert result['prefill_ms']['after_median'] == 85
    with pytest.raises(ValueError):
        M.summary(rows[:-2])


@pytest.mark.parametrize('metric,value', [('prefill_ms', 110), ('decode_ms', 500), ('rss_kib', 35000000)])
def test_regressions_fail(metric, value):
    rows = samples()
    for r in rows:
        if r['arm'] == 'after': r[metric] = value
    assert not M.summary(rows)['pass']


def test_different_tokens_and_nonfinite_measurements_rejected():
    rows = samples(); rows[-1]['ids'] = [2, 1]
    with pytest.raises(ValueError): M.summary(rows)
    rows = samples(); rows[-1]['prefill_ms'] = float('nan')
    with pytest.raises(ValueError): M.summary(rows)


def test_inherited_policy_and_profiler_do_not_contaminate_comparison(monkeypatch):
    for k in ['OMP_WAIT_POLICY', 'OFLM_OPEN_OMP_REEXEC', 'OFLM_OPEN_OMP_PASSIVE', 'OFLM_OPEN_PROFILE']:
        monkeypatch.setenv(k, 'contamination')
        assert k not in M.clean_env()


def test_missing_or_duplicate_repetition_rejected():
    rows = samples(); rows[-1]['rep'] = 0
    with pytest.raises(ValueError): M.summary(rows)


def test_production_outputs_exclude_following_status_and_streamed_turn():
    log = ('[OFLM]  Model RAW Output: \nfirst\nline\n\n>>>   Statistics:\n'
           '    Prefill time: 123 seconds\n>>> [OFLM] cleared\n'
           '>>> next streamed answer\n[OFLM]  Model RAW Output: \nsecond\n\n>>> Goodbye!\n')
    assert M.production_outputs(log) == ['first\nline', 'second']
    assert M.production_outputs(log.replace('123', '456')) == ['first\nline', 'second']


def test_production_outputs_preserve_model_text_and_reject_incomplete_log():
    assert M.production_outputs('[OFLM]  Model RAW Output: \nStatistics:\nvalue\n\n>>> ') == ['Statistics:\nvalue']
    with pytest.raises(ValueError):
        M.production_outputs('[OFLM]  Model RAW Output: \ntruncated')
