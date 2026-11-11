"""Acceptance gates for the matched OpenMP measurement driver."""
import importlib.util
from pathlib import Path
import subprocess

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


def test_source_provenance_records_both_requested_commits(monkeypatch):
    calls = iter([subprocess.CompletedProcess([], 0, stdout='current\n'),
                  subprocess.CompletedProcess([], 0, stdout='')])
    monkeypatch.setattr(M.subprocess, 'run', lambda *args, **kwargs: next(calls))
    result = M.source_provenance('before', 'after')
    assert result == dict(before_commit='before', after_commit='after',
                          current_commit='current', tracked_worktree_clean=True)


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


def test_exact_files_require_the_same_nonempty_set(tmp_path):
    left = tmp_path / 'before'; right = tmp_path / 'after'
    left.mkdir(); right.mkdir()
    assert not M.exact_files(left, right, 'state_*.bin')
    (right / 'state_0.bin').write_bytes(b'after')
    assert not M.exact_files(left, right, 'state_*.bin')
    (left / 'state_0.bin').write_bytes(b'after')
    assert M.exact_files(left, right, 'state_*.bin')
    (left / 'state_1.bin').write_bytes(b'extra')
    assert not M.exact_files(left, right, 'state_*.bin')


def server_rows():
    rows = []
    for label in ('A', 'B', 'A', 'B', 'A', 'B', 'A'):
        rows.append(dict(prompt=label, rss_kib=1000 if label == 'A' else 2000,
                         response=dict(ok=True, content='same answer',
                                       usage=dict(completion_tokens=2))))
    return rows


def test_server_result_allows_same_answer_for_different_prompts():
    result = M.server_result(server_rows(), 2)
    assert result['verdict'] == 'PASS'


def test_server_result_rejects_incomplete_or_unstable_responses():
    rows = server_rows(); rows[0]['response'] = dict(ok=False, error='failed')
    assert M.server_result(rows, 2)['verdict'] == 'FAIL'
    rows = server_rows(); rows[2]['response']['content'] = 'different'
    assert M.server_result(rows, 2)['verdict'] == 'FAIL'
    rows = server_rows(); rows[-1]['rss_kib'] = 1100
    assert M.server_result(rows, 2)['verdict'] == 'FAIL'


def test_server_result_compares_rss_within_each_prompt_size():
    rows = server_rows()
    rows[-1]['rss_kib'] = 1020
    result = M.server_result(rows, 2)
    assert result['verdict'] == 'PASS'
    assert result['rss_by_prompt']['A']['change_pct'] == pytest.approx(2)
    assert result['rss_by_prompt']['B']['change_pct'] == 0
