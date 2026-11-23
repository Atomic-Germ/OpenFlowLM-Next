"""CPU-only PR4A capture/evidence behaviour; no model, NPU, or benchmark runs."""
import builtins
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'utilities' / 'prefill-block'
sys.path.insert(0, str(TOOLS))
C = importlib.import_module('pipeline_capture')
E = importlib.import_module('pipeline_evidence')


def test_dedup_explicit_opt_in_sanitizes_inheritance(monkeypatch, tmp_path):
    monkeypatch.setenv('OFLM_OPEN_FOLD_DEDUP_INPUT_SPLIT', '1')
    args = (tmp_path / 'profile', None, 'prefill', '0', 'block', None)
    off, removed = C.environment(*args)
    assert 'OFLM_OPEN_FOLD_DEDUP_INPUT_SPLIT' not in off
    assert removed['OFLM_OPEN_FOLD_DEDUP_INPUT_SPLIT'] == '1'
    on, _ = C.environment(*args, dedup_input_split=True)
    assert on['OFLM_OPEN_FOLD_DEDUP_INPUT_SPLIT'] == '1'
    with pytest.raises(E.EvidenceError):
        C.environment(*args, dedup_input_split='1')


def test_process_memory_observation():
    procfs()
    observation = C.process_cpu(os.getpid())
    assert observation['rss_bytes'] > 0
    assert observation['hwm_bytes'] >= observation['rss_bytes']


def test_fold_report_uses_paired_gain_and_keeps_incomplete_set():
    report = importlib.import_module('fold_dedup_report')
    result = report.paired_observations([(100, 90), (200, 202)])
    assert result['paired_percent_reductions'] == [10, -1]
    assert result['paired_median_percent'] == 4.5
    assert result['spread_percentage_points'] == 11
    assert result['wins'] == 1
    assert not result['complete_initial_set']


def posix_lock():
    if os.name != 'posix':
        pytest.skip('POSIX advisory locks are unavailable')
    return pytest.importorskip('fcntl', reason='fcntl advisory locks are unavailable')


def procfs():
    if os.name != 'posix' or not Path('/proc/self/maps').is_file():
        pytest.skip('live-process provenance requires POSIX /proc')


def dump(path, value):
    path.write_text(json.dumps(value) + '\n')


def dump_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def geometry():
    return dict(family='qwen35', spec_hash='sha256:' + 'a' * 64, layers=2,
                kinds=['linear_attention', 'full_attention'], block_tokens=4,
                fold_calls_per_block=8, real_vocab=C.REAL_VOCAB)


def evidence(phase='prefill', tokens=5, max_tokens=1, route='block'):
    info = geometry()
    if phase == 'decode':
        schedule = [('step', 1, 1)] * (max_tokens - 1)
    elif route == 'layer-major':
        schedule = [('gemm-prompt', tokens, (tokens + 3) // 4)]
    else:
        schedule = [('gemm-block', min(4, tokens - at), 1) for at in range(0, tokens, 4)]
    rows = [dict(record='header', version=1, **{k: info[k] for k in ('family', 'spec_hash', 'layers')})]
    requests, folds, blocks = [], [dict(record='header', version=1)], []
    start = 0
    for index, (what, real, nblocks) in enumerate(schedule):
        lmhead = 2 if phase == 'decode' or index == len(schedule) - 1 else 0
        request = dict.fromkeys(E.REQUEST_MS, 0)
        request.update(record='request', what=what, tokens=real, blocks=nblocks, layers=2,
                       total_ms=20, gemm_ms=4, host_ms=6, lmhead_ms=lmhead,
                       dispatch_ms=8, layer_wall_ms=10, residue_ms=10)
        requests.append(request)
        rows.append(request)
        for layer, kind in enumerate(info['kinds']):
            row = dict.fromkeys(E.LAYER_MS, 0)
            row.update(record='layer', what=what, kind=kind, layer=layer, wall_ms=5)
            rows.append(row)
        if phase == 'prefill':
            end = tokens if route == 'layer-major' else start + 4
            blocks.append(f'  {what} [{start},{end}) t_real={real}: 20.0 ms '
                          f'(GEMM 4.0, host 6.0, per-token 0.0, lm_head {lmhead:.1f})')
            folds.append(dict(record='fold', what=what, real_tokens=real,
                              block_tokens=nblocks * 4, calls=8 * nblocks, split_ms=1,
                              copy_ms=2, accumulate_ms=3, total_ms=6))
            start += real
    text = 'resident after 1.0 s\n' + '\n'.join(blocks) + '\n'
    text += (f'prefill {tokens} tokens: {len(schedule) * 20} ms '
             f'({len(schedule) * 20 / tokens:.0f} ms/token)\n')
    for index in range(max_tokens):
        text += f'token {index + 2}\n'
        if index < max_tokens - 1:
            text += (f'  step @{tokens + index} [npu]: 20.0 ms (part0 4.0, route 0.00, '
                     'part1 6.0, lm_head 2.0; embed 0.00, patch 0.00, dispatches 8.0)\n')
    if max_tokens > 1:
        text += f'decode {max_tokens - 1} tokens: 20 ms/token (50.00 tok/s)\n'
    return rows, folds, text, requests


def validate(tmp_path, rows, text, phase='prefill', tokens=5, max_tokens=1, route='block'):
    path = tmp_path / 'layers.jsonl'
    dump_rows(path, rows)
    _, requests = E.validate_profile(path, geometry(), phase, tokens, max_tokens, route)
    return E.validate_log(text, phase, tokens, max_tokens, requests, geometry(), route)


def test_prefill_unbounded_legitimate_wrapper_overhead(tmp_path):
    rows, _, text, _ = evidence()
    text = text.replace('40 ms (8 ms/token)', '5040 ms (1008 ms/token)')
    assert validate(tmp_path, rows, text)['prompt_wall_ms'] == 5040


@pytest.mark.parametrize('wall,average', [(39, 8), (40, 9), (5040, 8)])
def test_prefill_impossible_outer_summary(tmp_path, wall, average):
    rows, _, text, _ = evidence()
    text = text.replace('40 ms (8 ms/token)', f'{wall} ms ({average} ms/token)')
    with pytest.raises(E.EvidenceError):
        validate(tmp_path, rows, text)


@pytest.mark.parametrize('location', ['before-residency', 'after-prefill'])
def test_prefill_block_must_be_inside_phase(tmp_path, location):
    rows, _, text, _ = evidence()
    lines = text.splitlines()
    block = lines.pop(1)
    lines.insert(0 if location == 'before-residency' else len(lines), block)
    with pytest.raises(E.EvidenceError, match='boundaries'):
        validate(tmp_path, rows, '\n'.join(lines) + '\n')


@pytest.mark.parametrize('field', ['total_ms', 'gemm_ms', 'host_ms', 'per_token_ms', 'lmhead_ms'])
def test_wrapper_overhead_does_not_hide_changed_inner_values(tmp_path, field):
    rows, _, text, _ = evidence()
    text = text.replace('40 ms (8 ms/token)', '5040 ms (1008 ms/token)')
    rows[1][field] += 1
    if field == 'total_ms':
        rows[1]['residue_ms'] += 1
    with pytest.raises(E.EvidenceError):
        validate(tmp_path, rows, text)


def test_finished_raw_without_final_metadata_remains_invalid(tmp_path):
    _, _, text, _ = evidence()
    (tmp_path / 'run.log').write_text(text)
    result = E.audit_capture(tmp_path)
    assert not result['valid']
    assert any('capture.json' in error for error in result['errors'])


@pytest.mark.parametrize('delta,accepted', [(.2505, True), (.25051, False)])
def test_prefill_containment_uses_only_print_rounding(tmp_path, delta, accepted):
    rows, _, text, requests = evidence()
    # Two three-decimal request times and one integer outer wall.
    for row in requests:
        row['total_ms'] += delta
        row['residue_ms'] += delta
    text = text.replace('20.0 ms', '20.3 ms')
    if accepted:
        validate(tmp_path, rows, text)
    else:
        with pytest.raises(E.EvidenceError, match='shorter'):
            validate(tmp_path, rows, text)


@pytest.mark.parametrize('phase,tokens,max_tokens,route', [
    ('prefill', 1, 1, 'block'), ('prefill', 5, 1, 'block'),
    ('prefill', 8, 3, 'block'), ('prefill', 5, 1, 'layer-major'),
    ('decode', 5, 6, 'block'),
])
def test_complete_phase_evidence(tmp_path, phase, tokens, max_tokens, route):
    rows, folds, text, _ = evidence(phase, tokens, max_tokens, route)
    result = validate(tmp_path, rows, text, phase, tokens, max_tokens, route)
    assert result['generated_tokens'] == max_tokens - 1
    if phase == 'prefill':
        dump_rows(tmp_path / 'fold.jsonl', folds)
        assert len(E.validate_fold(tmp_path / 'fold.jsonl', geometry(), tokens, route)) == len(folds) - 1


@pytest.mark.parametrize('mutation', [
    lambda r: r.pop(0), lambda r: r.insert(1, r[0]), lambda r: r[0].update(version=2),
    lambda r: r[0].update(version=True), lambda r: r[0].update(family='other'),
    lambda r: r[0].update(spec_hash='old'), lambda r: r[0].update(layers=64),
    lambda r: r[1].update(tokens=3), lambda r: r[1].update(blocks=2),
    lambda r: r[1].update(what='step'), lambda r: r[1].update(layers=1),
    lambda r: r[2].update(layer=1), lambda r: r[2].update(layer=False),
    lambda r: r[3].update(layer=0), lambda r: r[2].update(what='step'),
    lambda r: r[2].update(kind='wrong'), lambda r: r[2].update(record='request'),
    lambda r: r[1].update(record='unknown'), lambda r: r[1].update(lmhead_ms=1),
    lambda r: r[4].update(lmhead_ms=0), lambda r: r[1].update(total_ms=21),
    lambda r: r[1].update(layer_wall_ms=11), lambda r: r.pop(), lambda r: r.extend(r[1:4]),
    lambda r: r.__delitem__(slice(1, None)),
])
def test_bad_profile_coverage_and_identity(tmp_path, mutation):
    rows, _, text, _ = evidence()
    mutation(rows)
    with pytest.raises(E.EvidenceError):
        validate(tmp_path, rows, text)


@pytest.mark.parametrize('kind,field', [('request', k) for k in E.REQUEST_MS] +
                         [('layer', k) for k in E.LAYER_MS])
def test_every_timing_column_is_required(tmp_path, kind, field):
    rows, _, text, _ = evidence()
    del rows[1 if kind == 'request' else 2][field]
    with pytest.raises(E.EvidenceError, match='missing timing'):
        validate(tmp_path, rows, text)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -1, True, '1', None])
@pytest.mark.parametrize('index', [1, 2])
def test_invalid_timing_values(tmp_path, value, index):
    rows, _, text, _ = evidence()
    rows[index]['host_ms'] = value
    with pytest.raises(E.EvidenceError):
        validate(tmp_path, rows, text)


@pytest.mark.parametrize('text', ['', '{}', '{}\n{', '[]\n', '{"record":1,"record":2}\n',
                                      '{"value":1e999}\n', 'null\n', '{"a":NaN}\n'])
def test_malformed_incomplete_nonfinite_jsonl(tmp_path, text):
    path = tmp_path / 'bad.jsonl'
    path.write_text(text)
    with pytest.raises(E.EvidenceError):
        E.jsonl(path)


@pytest.mark.parametrize('change', [
    lambda f: f.pop(0), lambda f: f[0].update(version=2), lambda f: f[0].update(version=True),
    lambda f: f.insert(1, f[0]),
    lambda f: f.pop(), lambda f: f.append(f[1]), lambda f: f[1].update(what=''),
    lambda f: f[1].update(what='step'), lambda f: f[1].update(real_tokens=0),
    lambda f: f[1].update(real_tokens=5), lambda f: f[1].update(block_tokens=0),
    lambda f: f[1].update(block_tokens=8), lambda f: f[1].update(calls=0),
    lambda f: f[1].update(calls=True), lambda f: f[2].update(calls=16),
    lambda f: f[1].update(total_ms=7),
    lambda f: f[1].update(split_ms=float('inf')), lambda f: f[1].update(copy_ms=-1),
] + [lambda f, key=key: f[1].pop(key) for key in
     ('record', 'what', 'real_tokens', 'block_tokens', 'calls') + E.FOLD_MS])
def test_bad_fold_records_are_rejected_never_relabelled(tmp_path, change):
    _, folds, _, _ = evidence()
    change(folds)
    path = tmp_path / 'fold.jsonl'
    dump_rows(path, folds)
    raw = path.read_bytes()
    with pytest.raises(E.EvidenceError):
        E.validate_fold(path, geometry(), 5, 'block')
    assert path.read_bytes() == raw


@pytest.mark.parametrize('change', [
    lambda t: t.replace('resident after 1.0 s', ''),
    lambda t: t + 'resident after invalid s\n',
    lambda t: t.replace('prefill 5 tokens: 100 ms (20 ms/token)', ''),
    lambda t: t.replace('prefill 5 tokens:', 'prefill 6 tokens:'),
    lambda t: t.replace('prefill 5 tokens: 100', 'prefill 5 tokens: nan'),
    lambda t: t + 'prefill invalid\n',
    lambda t: t.replace('token 2\n', ''), lambda t: t.replace('token 2\n', 'token -1\n'),
    lambda t: t.replace('token 2\n', f'token {C.REAL_VOCAB}\n'),
    lambda t: t.replace('token 2\n', 'token 1.5\n'),
    lambda t: t.replace('step @5 ', 'step @4 '),
    lambda t: t.replace('step @6 ', 'step @5 '),
    lambda t: t.replace('[npu]', '[host]'),
    lambda t: t.replace('lm_head 2.0', 'lm_head 0.0'),
    lambda t: t.replace('20.0 ms (part0', '21.0 ms (part0'),
    lambda t: t.replace('part0 4.0', 'part0 5.0'),
    lambda t: t.replace('route 0.00', 'route 0.20'),
    lambda t: t.replace('part1 6.0', 'part1 7.0'),
    lambda t: t.replace('dispatches 8.0', 'dispatches 9.0'),
    lambda t: t.replace('embed 0.00', 'embed 0.20'),
    lambda t: t.replace('patch 0.00', 'patch 0.20'),
    lambda t: t.replace('step @5 [npu]: 20.0', 'step @5 [npu]: inf'),
    lambda t: t.replace('decode 5 tokens:', 'decode 4 tokens:'),
    lambda t: t.replace('20 ms/token (50.00 tok/s)', '10 ms/token (100.00 tok/s)'),
    lambda t: t.replace('20 ms/token (50.00 tok/s)', '20 ms/token (100.00 tok/s)'),
    lambda t: t.replace('decode 5 tokens: 20 ms/token (50.00 tok/s)', ''),
    lambda t: t + 'decode garbage\n',
    lambda t: t + '  step @10 [npu]: 20.0 ms (part0 4.0, route 0.00, part1 6.0, '
                     'lm_head 2.0; embed 0.00, patch 0.00, dispatches 8.0)\n',
])
def test_decode_log_contradictions(tmp_path, change):
    rows, _, text, _ = evidence('decode', 5, 6)
    with pytest.raises(E.EvidenceError):
        validate(tmp_path, rows, change(text), 'decode', 5, 6)


@pytest.mark.parametrize('change', [
    lambda t: t.replace('gemm-block [0,4)', 'gemm-prompt [0,4)'),
    lambda t: t.replace('gemm-block [4,8)', 'gemm-block [5,8)'),
    lambda t: t.replace('t_real=1', 't_real=2'),
    lambda t: t.replace('20.0 ms (GEMM', '22.0 ms (GEMM'),
    lambda t: t.replace('GEMM 4.0', 'GEMM 5.0'),
    lambda t: t.replace('lm_head 2.0', 'lm_head 0.0'),
    lambda t: t.replace('prefill 5 tokens: 40', 'prefill 5 tokens: 80'),
    lambda t: t.replace('prefill 5 tokens: 40 ms (8 ms/token)', ''),
    lambda t: t.replace('t_real=1: 20.0', 't_real=1: nan'),
])
def test_prefill_log_contradictions(tmp_path, change):
    rows, _, text, _ = evidence()
    with pytest.raises(E.EvidenceError):
        validate(tmp_path, rows, change(text))


def test_decode_prompt_profiles_do_not_count_as_generated(tmp_path):
    rows, _, text, _ = evidence('decode', 5, 6)
    for row in rows[1:]:
        row['lmhead_ms'] = 0
    with pytest.raises(E.EvidenceError, match='LM head'):
        validate(tmp_path, rows, text, 'decode', 5, 6)


def test_phase_order_and_unknown_phase_rejected(tmp_path):
    rows, _, text, _ = evidence('decode', 5, 6)
    lines = text.splitlines()
    lines[4], lines[5] = lines[5], lines[4]
    with pytest.raises(E.EvidenceError, match='phase order'):
        validate(tmp_path, rows, '\n'.join(lines) + '\n', 'decode', 5, 6)
    with pytest.raises(E.EvidenceError, match='phase/token/route'):
        validate(tmp_path, rows, text, 'unknown', 5, 6)


def test_lock_separate_handle_and_process_contention_and_release(tmp_path):
    fcntl = posix_lock()
    path = tmp_path / 'lock'
    script = ('import fcntl,sys\n'
              'with open(sys.argv[1], "a") as f:\n'
              ' try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
              ' except BlockingIOError: sys.exit(23)\n')
    with pytest.raises(RuntimeError, match='deliberate'):
        with C.lock_exclusive(path):
            with path.open('a') as other:
                with pytest.raises(OSError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child = subprocess.run([sys.executable, '-c', script, str(path)], timeout=5)
            assert child.returncode == 23
            raise RuntimeError('deliberate')
    child = subprocess.run([sys.executable, '-c', script, str(path)], timeout=5)
    assert child.returncode == 0
    with C.lock_exclusive(path):
        pass


def test_absent_fcntl_is_explicit_not_an_import_failure(tmp_path, monkeypatch):
    original = builtins.__import__

    def without_fcntl(name, *args, **kwargs):
        if name == 'fcntl':
            raise ImportError('unavailable')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_fcntl)
    with pytest.raises(E.EvidenceError, match='lock unavailable'):
        with C.lock_exclusive(tmp_path / 'lock'):
            pytest.fail('must not run unlocked')


@pytest.fixture
def assets(tmp_path):
    cli = tmp_path / 'cli'
    cli.write_text('fake executable, never launched by unit tests\n')
    cli.chmod(0o755)
    source = tmp_path / 'core.cpp'
    source.write_text('attested source\n')
    build = tmp_path / 'CMakeCache.txt'
    build.write_text('attested build configuration\n')
    identity = tmp_path / 'build-identity.json'
    dump(identity, dict(version=1, executable_sha256=E.sha256(cli), source_revision='archived-revision',
                        build_command=['cmake', '--build', 'archived-build'],
                        source_files={str(source): E.sha256(source)},
                        build_files={str(build): E.sha256(build)}))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.q4nx').write_bytes(b'fake weights')
    dump(model / 'config.json', {'model': 'fixture'})
    kernels = tmp_path / 'kernels'
    kernels.mkdir()
    (kernels / 'final.xclbin').write_bytes(b'fake kernel')
    (kernels / 'insts.bin').write_bytes(b'fake instructions')
    info = geometry()
    manifest = dict(family=info['family'], spec_hash=info['spec_hash'],
                    spec=dict(num_layers=2, layer_types=info['kinds'], real_vocab=C.REAL_VOCAB),
                    contexts={'ctx': 'final.xclbin'},
                    kernels={'run': dict(context='ctx', insts='insts.bin')},
                    layer_types={kind: {'gemm_block': {'t': 4,
                        'program': [{'op': 'run'} for _ in range(2)],
                        'ffn_program': [{'op': 'run'} for _ in range(2)]}}
                        for kind in info['kinds']})
    dump(kernels / 'manifest.json', manifest)
    runtime_dir = tmp_path / 'runtime'
    runtime_dir.mkdir()
    for name in E.RUNTIME_NAMES:
        (runtime_dir / (name + '.1')).write_bytes(b'fake runtime for unit tests')
    runtime = E.snapshot(runtime_dir.iterdir())
    return dict(cli=cli, identity=identity, source=source, build=build, model=model,
                kernels=kernels, runtime=runtime, root=tmp_path)


def fake_worker(assets, mutate=None):
    def worker(cmd, env, log, **kwargs):
        tokens = len(Path(cmd[cmd.index('--ids-file') + 1]).read_text().strip().split(','))
        max_tokens = int(cmd[cmd.index('--max-tokens') + 1])
        phase = 'decode' if '--decode-route' in cmd else 'prefill'
        route = 'block' if '--block-major' in cmd or phase == 'decode' else 'layer-major'
        rows, folds, text, _ = evidence(phase, tokens, max_tokens, route)
        dump_rows(Path(env['OFLM_OPEN_PROFILE']), rows)
        if 'OFLM_OPEN_FOLD_PROFILE' in env:
            dump_rows(Path(env['OFLM_OPEN_FOLD_PROFILE']), folds)
        Path(log).write_text(text)
        runtime = copy.deepcopy(assets['runtime'])
        result = dict(wall_s=1, cpu_observations=[{'utime_s': .1}],
                      runtime=dict(before=runtime, after=runtime, observed_pid=123,
                                   mapping_samples=[dict(at_s=.1, paths=list(runtime))]))
        if mutate:
            mutate(assets, cmd, env, result)
        return result
    return worker


def capture(assets, monkeypatch, *, mutate=None, **overrides):
    posix_lock()
    monkeypatch.setattr(C, 'fixture', lambda n: list(range(2, n + 2)))
    monkeypatch.setattr(C, 'capture_process', fake_worker(assets, mutate))
    args = dict(cli=assets['cli'], phase='prefill', tokens=5, max_tokens=1,
                out=assets['root'] / 'capture', ctx=32, pmode='none', profile_steps='0',
                build_identity=assets['identity'], model=assets['model'], kernels=assets['kernels'],
                lock=assets['root'] / 'lock', layers=2)
    args.update(overrides)
    return C.run(**args)


@pytest.mark.parametrize('phase,route,fold', [('prefill', 'block', False),
                                           ('prefill', 'block', True),
                                           ('prefill', 'layer-major', True),
                                           ('decode', 'block', False)])
def test_capture_owns_evidence_and_roundtrip_audit(assets, monkeypatch, phase, route, fold):
    kwargs = dict(phase=phase, route=route, fold_instrumentation=fold)
    if phase == 'decode':
        kwargs.update(max_tokens=6, profile_steps='decode:5')
    rec = capture(assets, monkeypatch, **kwargs)
    assert rec['version'] == 2 and rec['cpu_wait'] == 'NOT MEASURED'
    assert rec['manifest_block_tokens'] == 4
    assert rec['physical_block_tokens'] == (4 if phase == 'prefill' else None)
    assert rec['cpu_observations'] == [{'utime_s': .1}]
    assert rec['before'] == rec['after']
    assert str(assets['cli']) in rec['before']
    assert str(assets['source']) in rec['before']
    assert str(assets['build']) in rec['before']
    assert str(assets['kernels'] / 'insts.bin') in rec['before']
    assert str(assets['model'] / 'model.q4nx') in rec['before']
    assert rec['argv'][0] == str(assets['cli'])
    assert 'run.log' in rec['raw_hashes'] and 'invocation.json' in rec['raw_hashes']
    audit = E.audit_capture(assets['root'] / 'capture')
    assert audit['valid'], audit['errors']


@pytest.mark.parametrize('overrides', [
    {'phase': None}, {'phase': ''}, {'phase': 'invalid'}, {'tokens': 0}, {'tokens': -1},
    {'tokens': True}, {'tokens': C.POOL + 1}, {'tokens': 1.5}, {'max_tokens': 0},
    {'phase': 'decode', 'max_tokens': 5}, {'ctx': 4}, {'profile_steps': None},
    {'profile_steps': '3'}, {'phase': 'decode', 'max_tokens': 6, 'profile_steps': '3'},
    {'phase': 'decode', 'max_tokens': 6, 'profile_steps': 'decode:4'},
    {'phase': 'decode', 'max_tokens': 6, 'profile_steps': 'decode:5', 'fold_instrumentation': True},
    {'build_identity': None}, {'timeout': float('nan')}, {'timeout': 0},
    {'extra_env': {'OFLM_OPEN_FOLD_PROFILE': '/unexpected'}},
    {'extra_env': {'OFLM_OPEN_SUBMIT_AHEAD': '1'}}, {'pmode': 'unknown'},
])
def test_invalid_capture_inputs_fail_before_launch(assets, monkeypatch, overrides):
    with pytest.raises(E.EvidenceError):
        capture(assets, monkeypatch, **overrides)
    assert not (assets['root'] / 'capture' / 'capture.json').exists()


@pytest.mark.parametrize('field', ['version', 'executable_sha256', 'source_revision',
                                   'build_command', 'source_files', 'build_files'])
def test_each_build_identity_field_required(assets, monkeypatch, field):
    identity = E.read_json(assets['identity'])
    del identity[field]
    dump(assets['identity'], identity)
    with pytest.raises(E.EvidenceError):
        capture(assets, monkeypatch)


@pytest.mark.parametrize('field,value', [('executable_sha256', '0' * 64),
                                        ('source_revision', 'HEAD'), ('source_revision', 'unknown'),
                                        ('source_files', {}), ('build_files', {}),
                                        ('build_command', 'cmake --build .')])
def test_untrustworthy_build_identity_rejected(assets, monkeypatch, field, value):
    identity = E.read_json(assets['identity'])
    identity[field] = value
    dump(assets['identity'], identity)
    with pytest.raises(E.EvidenceError):
        capture(assets, monkeypatch)


@pytest.mark.parametrize('artifact', ['cli', 'identity', 'source', 'build', 'model/model.q4nx',
                                     'kernels/manifest.json', 'kernels/final.xclbin', 'kernels/insts.bin'])
def test_missing_required_artifacts_rejected(assets, monkeypatch, artifact):
    parts = artifact.split('/')
    path = assets[parts[0]].joinpath(*parts[1:])
    path.unlink()
    with pytest.raises((E.EvidenceError, OSError)):
        capture(assets, monkeypatch)


def test_inherited_oflm_contamination_is_removed_and_recorded(assets, monkeypatch):
    for key in ('OFLM_OPEN_FOLD_PROFILE', 'OFLM_OPEN_PROFILE_STEPS', 'OFLM_OPEN_SUBMIT_AHEAD',
                'OFLM_OPEN_SPIN_US', 'OFLM_OPEN_STAGE_MAJOR', 'OFLM_OPEN_LAYER_MAJOR',
                'OFLM_OPEN_PREFILL_ROLES', 'OFLM_OPEN_PREFILL_ROLE_MODE', 'OFLM_FUTURE_KNOB'):
        monkeypatch.setenv(key, 'contamination')
    rec = capture(assets, monkeypatch)
    assert rec['sanitized_inherited_oflm']['OFLM_OPEN_FOLD_PROFILE'] == 'contamination'
    assert rec['environment']['OFLM_OPEN_PROFILE_STEPS'] == '0'
    assert rec['environment']['OFLM_OPEN_LAYER_MAJOR'] == '0'
    assert 'OFLM_OPEN_FOLD_PROFILE' not in rec['environment']
    assert 'OFLM_OPEN_SUBMIT_AHEAD' not in rec['environment']
    assert 'OFLM_FUTURE_KNOB' not in rec['environment']


@pytest.mark.parametrize('name', E.RUNTIME_NAMES)
def test_missing_each_live_runtime_is_rejected(assets, monkeypatch, name):
    def mutate(assets, cmd, env, result):
        for group in ('before', 'after'):
            result['runtime'][group] = {p: v for p, v in result['runtime'][group].items()
                                        if not Path(p).name.startswith(name + '.')}
    with pytest.raises(E.EvidenceError, match='missing live runtime'):
        capture(assets, monkeypatch, mutate=mutate)


def test_empty_runtime_and_fabricated_mapping_inventory_rejected(assets):
    with pytest.raises(E.EvidenceError, match='missing/changed live runtime'):
        E.validate_runtime({})
    live = dict(before=assets['runtime'], after=assets['runtime'], observed_pid=123,
                mapping_samples=[dict(at_s=.1, paths=['/unrelated'])])
    with pytest.raises(E.EvidenceError, match='inventory contradicts'):
        E.validate_runtime(live)


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(observed_pid=True), lambda r: r.update(mapping_samples=[]),
    lambda r: r.update(mapping_samples=['not an object']),
    lambda r: r.update(mapping_samples=[dict(at_s=float('nan'), paths=[])]),
    lambda r: r.update(mapping_samples=[dict(at_s=.1, paths=['relative/path'])]),
    lambda r: r.update(before=[], after=[]),
])
def test_invalid_runtime_records(assets, mutation):
    live = dict(before=assets['runtime'], after=assets['runtime'], observed_pid=123,
                mapping_samples=[dict(at_s=.1, paths=list(assets['runtime']))])
    mutation(live)
    with pytest.raises(E.EvidenceError):
        E.validate_runtime(live)


@pytest.mark.parametrize('mutation', [
    lambda m: m.update(spec=[]), lambda m: m.update(spec={}),
    lambda m: m['spec'].update(num_layers=1), lambda m: m['spec'].update(layer_types=[]),
    lambda m: m.update(layer_types=[]), lambda m: m.update(layer_types={}),
    lambda m: m['layer_types']['full_attention'].update(gemm_block={'t': 8}),
    lambda m: m['layer_types']['full_attention'].update(gemm_block={'t': []}),
    lambda m: m.update(family=''), lambda m: m.update(spec_hash=''),
    lambda m: m.update(contexts={}), lambda m: m.update(kernels={}),
    lambda m: m['kernels']['run'].update(insts='../outside.bin'),
    lambda m: m['kernels']['run'].update(insts='/absolute.bin'),
    lambda m: m['kernels']['run'].pop('insts'),
])
def test_bad_manifest_and_kernel_inventory(assets, monkeypatch, mutation):
    path = assets['kernels'] / 'manifest.json'
    manifest = E.read_json(path)
    mutation(manifest)
    dump(path, manifest)
    with pytest.raises(E.EvidenceError):
        capture(assets, monkeypatch)


@pytest.mark.parametrize('artifact', ['cli', 'source', 'build', 'identity', 'model/model.q4nx',
                                     'kernels/manifest.json', 'kernels/final.xclbin', 'kernels/insts.bin'])
def test_artifact_changes_during_capture_rejected_and_lock_released(assets, monkeypatch, artifact):
    def mutate(assets, cmd, env, result):
        parts = artifact.split('/')
        path = assets[parts[0]].joinpath(*parts[1:])
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(E.EvidenceError, match='changed during capture'):
        capture(assets, monkeypatch, mutate=mutate)
    assert not (assets['root'] / 'capture' / 'capture.json').exists()
    assert (assets['root'] / 'capture' / 'invocation.json').exists()
    with C.lock_exclusive(assets['root'] / 'lock'):
        pass


@pytest.mark.parametrize('artifact', ['cli', 'source', 'build', 'identity'])
def test_changes_between_attestation_and_lock_are_rejected(assets, monkeypatch, artifact):
    from contextlib import contextmanager
    original_lock = C.lock_exclusive

    @contextmanager
    def racing_lock(path):
        with original_lock(path) as handle:
            target = assets[artifact]
            if artifact == 'identity':
                identity = E.read_json(target)
                identity['source_revision'] = 'different-build'
                dump(target, identity)
            else:
                target.write_bytes(target.read_bytes() + b' ')
            yield handle

    monkeypatch.setattr(C, 'lock_exclusive', racing_lock)
    with pytest.raises(E.EvidenceError, match='(snapshotted|before launch)'):
        capture(assets, monkeypatch)
    assert not (assets['root'] / 'capture').exists()


def test_runtime_change_during_capture_rejected(assets, monkeypatch):
    def mutate(assets, cmd, env, result):
        Path(next(iter(assets['runtime']))).write_bytes(b'replaced runtime')
    with pytest.raises(E.EvidenceError, match='runtime artifact changed'):
        capture(assets, monkeypatch, mutate=mutate)


@pytest.mark.parametrize('name', ['ids.txt', 'invocation.json'])
def test_invocation_and_token_inputs_cannot_change_during_capture(assets, monkeypatch, name):
    def mutate(assets, cmd, env, result):
        path = assets['root'] / 'capture' / name
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(E.EvidenceError, match='inputs changed during capture'):
        capture(assets, monkeypatch, mutate=mutate)


def test_added_artifacts_cannot_escape_provenance_snapshot(assets, monkeypatch):
    def mutate(assets, cmd, env, result):
        (assets['kernels'] / 'new-insts.bin').write_bytes(b'new instructions')
    with pytest.raises(E.EvidenceError, match='changed during capture'):
        capture(assets, monkeypatch, mutate=mutate)


def test_symlinked_kernel_directories_are_in_snapshot_and_tampering_rejected(assets, monkeypatch):
    external = assets['root'] / 'shared-kernels'
    external.mkdir()
    (external / 'insts.bin').write_bytes(b'shared instructions')
    link = assets['kernels'] / 'linked'
    try:
        link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('directory symlinks are unavailable')
    manifest_path = assets['kernels'] / 'manifest.json'
    manifest = E.read_json(manifest_path)
    manifest['kernels']['run']['insts'] = 'linked/insts.bin'
    dump(manifest_path, manifest)
    rec = capture(assets, monkeypatch)
    assert str(link / 'insts.bin') in rec['before']
    (external / 'insts.bin').write_bytes(b'replaced shared instructions')
    assert not E.audit_capture(assets['root'] / 'capture')['valid']


def test_cyclic_directory_symlink_is_rejected(tmp_path):
    directory = tmp_path / 'cycle'
    directory.mkdir()
    try:
        (directory / 'self').symlink_to(directory, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('directory symlinks are unavailable')
    with pytest.raises(E.EvidenceError, match='cyclic'):
        E.tree_files(directory)


@pytest.mark.parametrize('artifact', ['cli', 'source', 'build', 'identity', 'model/model.q4nx',
                                     'kernels/manifest.json', 'kernels/final.xclbin', 'kernels/insts.bin',
                                     'capture/run.log', 'capture/layers.jsonl', 'capture/ids.txt',
                                     'capture/invocation.json'])
def test_audit_rejects_provenance_or_raw_tampering(assets, monkeypatch, artifact):
    capture(assets, monkeypatch)
    parts = artifact.split('/')
    path = ((assets['root'] / 'capture').joinpath(*parts[1:]) if parts[0] == 'capture'
            else assets[parts[0]].joinpath(*parts[1:]))
    path.write_bytes(path.read_bytes() + b'tampered')
    audit = E.audit_capture(assets['root'] / 'capture')
    assert not audit['valid'] and audit['errors']


def test_audit_runtime_replacement_rejected(assets, monkeypatch):
    capture(assets, monkeypatch)
    Path(next(iter(assets['runtime']))).write_bytes(b'replaced runtime')
    audit = E.audit_capture(assets['root'] / 'capture')
    assert not audit['valid'] and any('runtime artifact changed' in e for e in audit['errors'])


@pytest.mark.parametrize('field', ['build_identity', 'before', 'raw_hashes', 'argv', 'environment',
                                   'input_before', 'input_after'])
def test_audit_missing_capture_provenance_rejected(assets, monkeypatch, field):
    capture(assets, monkeypatch)
    path = assets['root'] / 'capture' / 'capture.json'
    rec = E.read_json(path)
    del rec[field]
    dump(path, rec)
    assert not E.audit_capture(path.parent)['valid']


@pytest.mark.parametrize('field,value', [('phase', 'unknown'), ('physical_block_tokens', 99),
                                       ('fold_instrumentation', 'false'), ('max_tokens', 2),
                                       ('manifest_path', '/unrecorded/manifest.json')])
def test_audit_contradictory_capture_metadata(assets, monkeypatch, field, value):
    capture(assets, monkeypatch)
    path = assets['root'] / 'capture' / 'capture.json'
    record = E.read_json(path)
    record[field] = value
    dump(path, record)
    assert not E.audit_capture(path.parent)['valid']


def test_legacy_captures_are_independent_read_only_inventories(assets):
    captures = []
    for index, phase in enumerate(('prefill', 'decode')):
        directory = assets['root'] / f'legacy-{index}'
        directory.mkdir()
        binary = assets['root'] / f'old-cli-{index}'
        binary.write_bytes(f'actual old binary {index}'.encode())
        rec = dict(version=1, phase=phase, cli=dict(path=str(binary), sha256=E.sha256(binary)),
                   real_prompt_tokens=256 if index == 0 else 32, generated_tokens=index * 7,
                   command=f'{binary} --original-command {index}',
                   runtime={'paths': [f'/original/runtime-{index}'], 'libraries': {}})
        dump(directory / 'capture.json', rec)
        (directory / 'run.log').write_text(f'original log {index}\n')
        if index == 0:
            dump_rows(directory / 'fold.jsonl', [dict(record='header', version=1),
                dict(record='fold', what='', real_tokens=0, block_tokens=0, calls=256,
                     split_ms=1, copy_ms=2, accumulate_ms=3, total_ms=6)])
        captures.append((directory, rec))
    for directory, rec in captures:
        before = {p: p.read_bytes() for p in directory.iterdir()}
        result = E.audit_capture(directory)
        assert not result['valid'] and 'missing build identity' in result['errors']
        assert result['binary']['sha256'] == rec['cli']['sha256']
        assert result['binary_current_sha256'] == rec['cli']['sha256']
        assert result['phase'] == rec['phase'] and result['command'] == rec['command']
        assert result['runtime'] == rec['runtime']
        assert result['raw_hashes']['run.log'] == E.sha256(directory / 'run.log')
        assert {p: p.read_bytes() for p in directory.iterdir()} == before
        if rec['phase'] == 'prefill':
            assert any('cannot repair' in error for error in result['errors'])
    assert captures[0][1]['cli']['sha256'] != captures[1][1]['cli']['sha256']


def test_older_layer_profile_format_preserves_each_command_and_prompt_phase(assets):
    directory = assets['root'] / 'old-profile'
    directory.mkdir()
    rows, _, _, _ = evidence('decode', 5, 6)
    for row in rows[1:]:
        row['lmhead_ms'] = 0
    dump_rows(directory / 'profile-sequential.jsonl', rows)
    dump(directory / 'result.json', dict(tool='prefill-block-layer-profile', version=1,
         cli=str(assets['cli']), prompt_tokens=5,
         provenance={str(assets['cli']): {'sha256': E.sha256(assets['cli'])}},
         runs={'sequential': dict(command='actual old sequential invocation',
                                 profile='old/location/profile-sequential.jsonl', generated_ids=[2, 3])}))
    before = {p: p.read_bytes() for p in directory.iterdir()}
    audit = E.audit_capture(directory)
    assert not audit['valid'] and audit['phase'] is None
    assert audit['binary']['sha256'] == E.sha256(assets['cli'])
    entry = audit['runs'][0]
    assert entry['command'] == 'actual old sequential invocation'
    assert entry['runtime'] is None
    assert len(entry['profile_requests']) == 5
    assert any('not generated-decode evidence' in error for error in entry['errors'])
    assert {p: p.read_bytes() for p in directory.iterdir()} == before


def python_process(tmp_path, code, timeout=5):
    procfs()
    exe = Path(sys.executable).resolve()
    cmd = [str(exe), '-c', code]
    return C.capture_process(cmd, os.environ.copy(), tmp_path / 'run.log', timeout=timeout,
                             executable=E.fingerprint(exe))


def test_output_beyond_pipe_capacity_does_not_deadlock(tmp_path):
    # No XRT mappings: this process must finish and retain all output, then fail provenance.
    with pytest.raises(E.EvidenceError, match='live runtime provenance'):
        python_process(tmp_path, 'import os; os.write(1, b"x" * (2 * 1024 * 1024))')
    assert (tmp_path / 'run.log').stat().st_size == 2 * 1024 * 1024


def test_timeout_kills_process_group_and_reaps_child(tmp_path, monkeypatch):
    procfs()
    real_popen = subprocess.Popen
    processes = []

    def spawn(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(C.subprocess, 'Popen', spawn)
    code = ('import os,time\n'
            'print("started", flush=True)\n'
            'if os.fork() == 0: time.sleep(20)\n'
            'time.sleep(20)\n')
    with pytest.raises(subprocess.TimeoutExpired):
        python_process(tmp_path, code, timeout=.3)
    assert processes[0].returncode is not None
    assert processes[0].poll() is not None
    assert 'started' in (tmp_path / 'run.log').read_text()
    # Children may remain zombies until init reaps them; none may still execute.
    pgid = processes[0].pid
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
            assert int(fields[2]) != pgid or fields[0] == 'Z'
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            pytest.skip('process-group inspection unavailable under procfs permissions')


def test_observer_exception_also_kills_and_reaps(tmp_path, monkeypatch):
    procfs()
    children = []
    real_popen = subprocess.Popen

    def spawn(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        children.append(proc)
        return proc

    def broken(pid):
        raise E.EvidenceError('observer failed')

    monkeypatch.setattr(C.subprocess, 'Popen', spawn)
    monkeypatch.setattr(C, 'libs_from_maps', broken)
    with pytest.raises(E.EvidenceError, match='observer failed'):
        python_process(tmp_path, 'import time; time.sleep(20)')
    assert children[0].returncode is not None


def test_fake_process_live_mappings_are_sampled_and_hashed(tmp_path):
    procfs()
    paths = []
    for name in E.RUNTIME_NAMES:
        path = tmp_path / (name + '.fixture')
        path.write_bytes(b'fake mapped runtime\n')
        paths.append(str(path))
    code = ('import mmap,time\n'
            f'files=[open(p,"rb") for p in {paths!r}]\n'
            'maps=[mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) for f in files]\n'
            'print("mapped", flush=True)\n'
            'time.sleep(.2)\n')
    result = python_process(tmp_path, code)
    assert set(result['runtime']['before']) == set(paths)
    assert result['runtime']['observed_pid'] != os.getpid()
    assert result['runtime']['before'] == result['runtime']['after']


def test_live_mapping_replaced_path_is_not_hashed_as_loaded_binary(tmp_path):
    procfs()
    path = tmp_path / 'libxrt_core.so.fixture'
    path.write_bytes(b'original mmap bytes')
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'replacement bytes')
    code = ('import mmap,os,time\n'
            f'f=open({str(path)!r},"rb")\n'
            'm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)\n'
            f'os.replace({str(replacement)!r},{str(path)!r})\n'
            'time.sleep(.2)\n')
    with pytest.raises(E.EvidenceError, match='mapping (was deleted|path was replaced)'):
        python_process(tmp_path, code)


def test_nonzero_process_keeps_raw_output(tmp_path):
    with pytest.raises(E.EvidenceError, match='rc=9'):
        python_process(tmp_path, 'import sys; print("failure detail"); sys.exit(9)')
    assert 'failure detail' in (tmp_path / 'run.log').read_text()


def test_timeout_releases_capture_lock_and_never_emits_success(assets, monkeypatch):
    posix_lock()

    def worker(cmd, env, log, **kwargs):
        Path(log).write_text('partial failure output\n')
        raise subprocess.TimeoutExpired(cmd, .1)

    monkeypatch.setattr(C, 'fixture', lambda n: list(range(2, n + 2)))
    monkeypatch.setattr(C, 'capture_process', worker)
    with pytest.raises(subprocess.TimeoutExpired):
        C.run(assets['cli'], 'prefill', 5, 1, assets['root'] / 'failed', 32, 'none', '0',
              build_identity=assets['identity'], model=assets['model'], kernels=assets['kernels'],
              lock=assets['root'] / 'lock', layers=2)
    assert (assets['root'] / 'failed' / 'run.log').read_text() == 'partial failure output\n'
    assert not (assets['root'] / 'failed' / 'capture.json').exists()
    with C.lock_exclusive(assets['root'] / 'lock'):
        pass


def test_compiled_fake_cli_end_to_end(tmp_path, assets, monkeypatch):
    procfs()
    posix_lock()
    cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
    if not cc:
        pytest.skip('no C compiler for standalone fake CLI')
    rows, folds, text, _ = evidence()
    profile_text = ''.join(json.dumps(row) + '\n' for row in rows)
    runtime_paths = list(assets['runtime'])
    source = tmp_path / 'fake.c'
    source.write_text('#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n'
                      '#include <fcntl.h>\n#include <sys/mman.h>\n'
                      'int main(void) {\n' + ''.join(
                          f'int f{i}=open({json.dumps(p)},O_RDONLY); '
                          f'if (mmap(0,1,PROT_READ,MAP_PRIVATE,f{i},0)==MAP_FAILED) return 5;\n'
                          for i, p in enumerate(runtime_paths)) +
                      'FILE *f=fopen(getenv("OFLM_OPEN_PROFILE"),"w");\n'
                      f'fputs({json.dumps(profile_text)},f); fclose(f);\n'
                      f'fputs({json.dumps(text)},stdout); fflush(stdout);\n'
                      'usleep(200000); return 0; }\n')
    cli = tmp_path / 'fake-cli'
    subprocess.run([cc, str(source), '-o', str(cli)], check=True, capture_output=True, timeout=15)
    identity = E.read_json(assets['identity'])
    identity.update(executable_sha256=E.sha256(cli), source_files={str(source): E.sha256(source)},
                    build_command=[cc, str(source), '-o', str(cli)])
    dump(assets['identity'], identity)
    monkeypatch.setattr(C, 'fixture', lambda n: list(range(2, n + 2)))
    rec = C.run(cli, 'prefill', 5, 1, tmp_path / 'compiled-capture', 32, 'none', '0',
                build_identity=assets['identity'], model=assets['model'], kernels=assets['kernels'],
                lock=tmp_path / 'lock', layers=2, timeout=5)
    assert rec['generated_ids'] == [2]
    audit = E.audit_capture(tmp_path / 'compiled-capture')
    assert audit['valid'], audit['errors']
