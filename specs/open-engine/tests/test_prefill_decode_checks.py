"""Both decode gates share dispatch/readback, but keep distinct acceptance rules."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'utilities/prefill-block/decode_transition.py'


def load():
    spec = importlib.util.spec_from_file_location('decode_checks', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A subprocess CLI double writes real dumps; it is not hardware evidence."""
    m = load()
    cli = tmp_path / 'cli'
    cli.write_text('''#!/usr/bin/env python3
import json, os, struct, sys
from pathlib import Path
args = sys.argv[1:]
def arg(name): return args[args.index(name) + 1]
prefix = arg('--dump-logits')
n = len(Path(arg('--ids-file')).read_text().strip().split(','))
decode = int(arg('--max-tokens'))
block = '--gemm-block' in args
case = os.environ.get('PREFILL_TEST_CASE', '') if block else ''
assert os.environ['OFLM_OPEN_PREFILL_PRECISION'] == 'paired'
assert 'OFLM_OPEN_PREFILL_ROLES' not in os.environ
assert 'OFLM_OPEN_PREFILL_ROLE_MODE' not in os.environ
if block: assert os.environ['OFLM_OPEN_LAYER_MAJOR'] == '0'
def write(path, values): Path(path).write_bytes(struct.pack('<' + 'f'*len(values), *values))
for suffix in [f'p{n-1}'] + [f't{k}' for k in range(decode)]:
    if case == 'missing_all' or (case == 'missing_decode' and suffix == 't0'): continue
    values = [1., 1.00001, .3, -.2]
    if case == 'flip': values[:2] = [1.00001, 1.]
    if case == 'wrong': values = [-v for v in values]
    if case == 'nan': values[0] = float('nan')
    if case == 'empty': values = []
    write(prefix + '_' + suffix + '.bin', values)
for i, flag in enumerate(args):
    if flag != '--dump-state': continue
    layer, offset, size, path = args[i+1].split(':', 3)
    if case == 'missing_state': continue
    count = int(size)//4
    if case == 'short_state': count -= 1
    write(path, [1.]*count)
for k in range(decode):
    if case == 'no_tokens' or (case == 'short_tokens' and k == decode-1): continue
    print('token', 0 if case == 'flip' else 1)
Path(prefix + '.args.json').write_text(json.dumps(args))
''')
    cli.chmod(0o755)
    monkeypatch.setattr(m.O, 'CLI', cli)
    monkeypatch.setattr(m.O, 'LOCK', tmp_path / 'npu.lock')
    monkeypatch.setattr(m.O, 'layer_kinds', lambda: ['linear', 'full'])
    for name in ('CONV_BYTES', 'S_OFF', 'S_BYTES', 'KV_ROW_BYTES'):
        monkeypatch.setattr(m.O, name, 16)
    monkeypatch.setenv('OFLM_OPEN_PREFILL_PRECISION', 'weight_high')
    monkeypatch.setenv('OFLM_OPEN_PREFILL_ROLES', 'all')
    monkeypatch.setenv('OFLM_OPEN_PREFILL_ROLE_MODE', 'three_term')
    monkeypatch.setenv('OFLM_OPEN_LAYER_MAJOR', '1')
    return m


def execute(m, tmp_path, mode, *, position=0):
    out = tmp_path / 'result'
    rc = m.main(['--check', mode, '--prompt', '3', '--decode', '2', '--layers', '2',
                 '--position', str(position), '--out', str(out)])
    return rc, json.loads((out / 'result.json').read_text())


@pytest.mark.parametrize('mode,tool', [('state', 'prefill-block-decode-transition'),
                                      ('logits', 'prefill-block-decode16-logits')])
def test_both_modes_dispatch_and_read_every_dump(harness, tmp_path, mode, tool):
    rc, r = execute(harness, tmp_path, mode, position=17)
    assert rc == 0 and r['verdict'] == 'PASS' and r['tool'] == tool
    assert r['model_acceptance'] is False
    assert r['chunk'] == 256
    assert {t['tensor'] for t in r['tensors']} >= {
        'logits@prompt_last', 'logits@decode0', 'logits@decode1'}
    for side in ('sequential', 'block'):
        args = json.loads(Path(r[side]['prefix'] + '.args.json').read_text())
        assert ('--gemm-block' in args) == (side == 'block')
        assert args[args.index('--at-position')+1] == '17'
        assert args.count('--dump-state') == (3 if mode == 'state' else 0)
    if mode == 'state':
        assert {'L0.conv', 'L0.S', 'L1.kv', 'greedy_tokens'} <= {t['tensor'] for t in r['tensors']}
    else:
        assert r['summary']['compared'] == 3 and r['summary']['tokens_compared'] == 2


@pytest.mark.parametrize('mode,expected', [('state', 1), ('logits', 0)])
def test_argmax_difference_only_fails_the_strict_state_gate(harness, tmp_path, monkeypatch, mode, expected):
    monkeypatch.setenv('PREFILL_TEST_CASE', 'flip')
    rc, r = execute(harness, tmp_path, mode)
    assert rc == expected
    assert all(t['verdict'] == 'PASS' for t in r['tensors'] if t['tensor'].startswith('logits@'))
    if mode == 'logits':
        assert r['summary']['tokens_agreeing'] == 0


@pytest.mark.parametrize('mode', ['state', 'logits'])
@pytest.mark.parametrize('case', ['missing_all', 'missing_decode', 'wrong', 'nan', 'empty'])
def test_bad_or_missing_logits_fail_and_still_write_a_report(harness, tmp_path, monkeypatch, mode, case):
    monkeypatch.setenv('PREFILL_TEST_CASE', case)
    rc, r = execute(harness, tmp_path, mode)
    assert rc == 1 and r['verdict'] == 'FAIL'
    assert any(t['verdict'] == 'FAIL' for t in r['tensors'])


@pytest.mark.parametrize('case', ['missing_state', 'short_state', 'no_tokens', 'short_tokens'])
def test_state_mode_requires_complete_state_and_token_coverage(harness, tmp_path, monkeypatch, case):
    monkeypatch.setenv('PREFILL_TEST_CASE', case)
    rc, r = execute(harness, tmp_path, 'state')
    assert rc == 1 and r['verdict'] == 'FAIL'


@pytest.mark.parametrize('args', [['--prompt', '0'], ['--decode', '0'], ['--layers', '0'],
                                 ['--position', '-1'], ['--chunk', '32']])
def test_invalid_geometry_fails_before_dispatch(harness, tmp_path, args):
    out = tmp_path / 'bad'
    with pytest.raises(SystemExit):
        harness.main(['--out', str(out), *args])
    assert not out.exists()


def test_existing_output_is_never_overwritten(harness, tmp_path):
    out = tmp_path / 'saved'; out.mkdir()
    saved = out / 'result.json'; saved.write_text('retained')
    with pytest.raises(SystemExit):
        harness.main(['--out', str(out)])
    assert saved.read_text() == 'retained'


def test_cli_help_exposes_both_gates_without_model_access():
    proc = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True)
    assert proc.returncode == 0 and '--check {state,logits}' in proc.stdout


@pytest.mark.parametrize('mode,defaults', [('state', (40, 2, 8)), ('logits', (700, 16, 64))])
def test_mode_defaults_and_explicit_overrides(mode, defaults):
    m = load()
    a = m.arguments(['--check', mode])
    assert (a.prompt, a.decode, a.layers) == defaults
    a = m.arguments(['--check', mode, '--prompt', '257', '--decode', '3', '--layers', '4'])
    assert (a.prompt, a.decode, a.layers) == (257, 3, 4)


@pytest.mark.parametrize('args', [['--layers', '65'], ['--position', '4095', '--prompt', '2'],
                                 ['--prompt', '4096', '--decode', '1']])
def test_context_or_layer_overflow_is_rejected_before_creating_output(harness, tmp_path, args):
    out = tmp_path / 'overflow'
    with pytest.raises(SystemExit):
        harness.main(['--out', str(out), *args])
    assert not out.exists()


def test_cli_failure_preserves_the_log_without_claiming_success(harness, tmp_path):
    harness.O.CLI.write_text('#!/bin/sh\necho native-failure >&2\nexit 7\n')
    out = tmp_path / 'result'
    with pytest.raises(SystemExit, match='rc 7'):
        harness.main(['--check', 'logits', '--out', str(out)])
    assert 'native-failure' in (out / 'seq/out_seq.log').read_text()
    assert not (out / 'result.json').exists()
