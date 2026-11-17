"""Acceptance gates for the matched OpenMP measurement driver."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def test_same_binary_benchmark_changes_only_the_policy(tmp_path, monkeypatch):
    pytest.importorskip('fcntl', reason='hardware driver uses POSIX locking')
    a = SimpleNamespace(tokens=2, generate=2, reps=5, before=tmp_path/'cli',
                        after=tmp_path/'cli', before_commit='same', after_commit='same',
                        model=tmp_path, kernels=tmp_path)
    monkeypatch.setattr(M.O, 'LOCK', tmp_path/'lock')
    monkeypatch.setattr(M.O, 'fixture', lambda *args, **kwargs: [1, 2])
    monkeypatch.setattr(M.P, 'provenance', lambda *args: {})
    monkeypatch.setattr(M.P, 'conditions', lambda: {})
    monkeypatch.setattr(M, 'source_provenance', lambda *args: {})
    monkeypatch.setattr(M, 'platform', lambda: dict(mode='default'))
    policies = []
    def invoke(command, env, timeout):
        policy = env['OFLM_OPEN_OMP_PASSIVE']
        policies.append(policy)
        assert 'OMP_WAIT_POLICY' not in env
        Path(command[4]).write_text('1000')
        ms = 100 if policy == '0' else 85
        return subprocess.CompletedProcess(command, 0, stdout=(
            f'prefill 2 tokens: {ms} ms\ndecode 2 tokens: 10 ms/token\ntoken 1\ntoken 2\n'), stderr=''), \
            ['/usr/lib/libxrt_coreutil.so.2.25.00']
    monkeypatch.setattr(M, 'run_sampled', invoke)
    result = M.benchmark(a, tmp_path)
    assert result['verdict'] == 'PASS'
    assert policies == ['1', '0', '0', '1', '1', '0', '0', '1', '1', '0', '0', '1']
    assert all(row['omp_passive'] == ('0' if row['arm'] == 'before' else '1')
               for row in result['rows'])
    assert all(row['runtime_libraries'] == ['/usr/lib/libxrt_coreutil.so.2.25.00']
               for row in result['rows'])


def test_kernel_ablation_changes_only_kernel_path_not_policy(tmp_path, monkeypatch):
    pytest.importorskip('fcntl', reason='hardware driver uses POSIX locking')
    a = SimpleNamespace(tokens=2, generate=2, reps=5, before=tmp_path/'cli',
                        after=tmp_path/'cli', before_commit='same', after_commit='same',
                        model=tmp_path, kernels=tmp_path/'unused',
                        before_kernels=tmp_path/'control', after_kernels=tmp_path/'candidate')
    monkeypatch.setattr(M.O, 'LOCK', tmp_path/'lock')
    monkeypatch.setattr(M.O, 'fixture', lambda *args, **kwargs: [1, 2])
    monkeypatch.setattr(M.P, 'provenance', lambda *args: {})
    monkeypatch.setattr(M.P, 'conditions', lambda: {})
    monkeypatch.setattr(M, 'source_provenance', lambda *args: {})
    monkeypatch.setattr(M, 'platform', lambda: dict(mode='default'))
    calls = []
    def invoke(command, env, timeout):
        calls.append((command, env))
        assert env['OFLM_OPEN_OMP_PASSIVE'] == '1'
        kernel = command[command.index('--kernels') + 1]
        Path(command[4]).write_text('1000')
        ms = 100 if kernel == str(a.before_kernels) else 95
        return subprocess.CompletedProcess(command, 0, stdout=(
            f'prefill 2 tokens: {ms} ms\ndecode 2 tokens: 10 ms/token\ntoken 1\ntoken 2\n'), stderr=''), []
    monkeypatch.setattr(M, 'run_sampled', invoke)
    result = M.benchmark(a, tmp_path)
    assert result['verdict'] == 'PASS'
    assert result['check'] == 'kernel-benchmark'
    assert len(calls) == 12
    for before, after in zip(calls[::2], calls[1::2]):
        assert before[1] == after[1]
        left, right = list(before[0][5:]), list(after[0][5:])
        index = left.index('--kernels') + 1
        assert {left[index], right[index]} == {str(a.before_kernels), str(a.after_kernels)}
        left[index] = right[index]
        assert left == right


def test_partial_kernel_ablation_is_refused():
    a = SimpleNamespace(before_kernels=Path('control'), after_kernels=None)
    with pytest.raises(ValueError, match='both kernel directories'):
        M.kernel_comparison(a)


def test_runtime_paths_read_resolved_libraries_not_sonames():
    maps = '\n'.join([
        '7f00-7f01 r-xp 0 08:02 1 /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2.25.00',
        '7f02-7f03 r-xp 0 08:02 1 /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2.26.0',
        '7f04-7f05 r-xp 0 08:02 1 /opt/xilinx/xrt/python/pyxrt.cpython-314-x86_64-linux-gnu.so',
        '7f06-7f07 r-xp 0 08:02 1 /usr/lib/x86_64-linux-gnu/libstdc++.so.6',
    ])
    assert M.runtime_paths(maps) == [
        '/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2.26.0',
        '/opt/xilinx/xrt/python/pyxrt.cpython-314-x86_64-linux-gnu.so',
        '/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2.25.00',
    ]


def test_run_sampled_records_child_libraries_and_exit(tmp_path):
    if not Path('/proc/self/maps').is_file():
        pytest.skip('live runtime mappings require Linux procfs')
    code = 'import time; time.sleep(0.3)'
    command = [sys.executable, '-c', code]
    result, libraries = M.run_sampled(command, dict(os.environ), 10)
    assert result.returncode == 0
    assert isinstance(libraries, list)


def test_runtime_sampler_follows_wrapper_and_drains_large_output(tmp_path):
    if not Path('/proc/self/maps').is_file():
        pytest.skip('live runtime mappings require Linux procfs')
    import math
    import shutil
    _ = math.sin(1)
    paths = [line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if '/libm.so.' in line]
    if not paths:
        pytest.skip('a mapped libm shared library is needed for the fake runtime fixture')
    library = tmp_path / 'libxrt_coreutil.so.2.25.00'
    shutil.copyfile(paths[0], library)
    child = ('import ctypes,sys,time; ctypes.CDLL(sys.argv[1]); '
             'print("x"*200000); print("e"*200000,file=sys.stderr); time.sleep(0.4)')
    wrapper = 'import subprocess,sys; sys.exit(subprocess.call(sys.argv[1:]))'
    command = [sys.executable, '-c', wrapper, sys.executable, '-c', child, str(library)]
    result, libraries = M.run_sampled(command, dict(os.environ), 10)
    assert result.returncode == 0
    assert str(library) in libraries
    assert len(result.stdout) > 200000 and len(result.stderr) > 200000


def test_runtime_sampler_times_out_and_reaps_child():
    command = [sys.executable, '-c', 'import time; time.sleep(30)']
    with pytest.raises(subprocess.TimeoutExpired):
        M.run_sampled(command, dict(os.environ), 0.1)


def server_tool():
    spec = importlib.util.spec_from_file_location(
        'server_candidate_check', ROOT / 'utilities/prefill-block/server_candidate_check.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def server_arms():
    def arm():
        libraries = {f'/runtime/{name}.so.2.25.00': {'sha256': 'a' * 64}
                     for name in ('libxrt_coreutil', 'libxrt_driver_xdna')}
        run = dict(libraries=libraries, a1={'content': 'A'}, b={'content': 'B'}, a2={'content': 'A'})
        return dict(verdict='PASS', artifacts={'manifest': 'identical'}, main=run,
                    fresh={'libraries': libraries})
    return {'before': arm(), 'after': arm()}


def test_server_comparison_requires_runtime_and_output_identity():
    import copy
    tool = server_tool()
    arms = server_arms()
    assert tool.comparison_passes(arms)
    for section, key, value in [
            ('main', 'b', {'content': 'DIFFERENT'}),
            ('main', 'libraries', {}),
            ('fresh', 'libraries', {'/missing.so': {}})]:
        candidate = copy.deepcopy(arms)
        candidate['after'][section][key] = value
        assert not tool.comparison_passes(candidate)
    candidate = copy.deepcopy(arms)
    candidate['after']['main']['libraries']['/runtime/libxrt_driver_xdna.so.2.25.00']['sha256'] = 'b' * 64
    assert not tool.comparison_passes(candidate)


def test_server_artifacts_must_match_retained_evidence():
    tool = server_tool()
    evidence = {'cases': [{'name': 'gemm_n256_k256',
                          'builds': {'after': {'binaries': {'final.xclbin': 'expected'}}}}]}
    artifacts = {'gemm': {'gemm_n256_k256': {'final.xclbin': 'expected'}}}
    assert tool.matches_evidence(artifacts, evidence, 'after')
    artifacts['gemm']['gemm_n256_k256']['final.xclbin'] = 'changed'
    assert not tool.matches_evidence(artifacts, evidence, 'after')
    assert not tool.matches_evidence({'gemm': {}}, evidence, 'after')


def test_runtime_identity_requires_core_plugin_and_hashes():
    from runtime_provenance import runtime_identity
    libraries = server_arms()['before']['main']['libraries']
    assert runtime_identity(libraries) is not None
    assert runtime_identity({}) is None
    libraries.pop('/runtime/libxrt_driver_xdna.so.2.25.00')
    assert runtime_identity(libraries) is None


def test_boundary_state_coverage_includes_every_layer(monkeypatch):
    monkeypatch.setattr(M.O, '_LAYER_KINDS', ['linear'] * 3 + ['full'] + ['linear'] * 3 + ['full'])
    slices = M.boundary_slices(8, 273)
    assert set(slices) == set(range(8))
    assert [s[0] for s in slices[0]] == ['conv', 'S']
    assert slices[7] == [('kv', 0, 273 * M.O.KV_ROW_BYTES)]
    with pytest.raises(ValueError, match='every requested layer'):
        M.boundary_slices(9, 273)


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
