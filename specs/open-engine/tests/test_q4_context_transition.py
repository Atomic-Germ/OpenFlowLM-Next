"""Transition recording is fail-closed and testable without XRT, an NPU or sudo."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location('context_transition', ROOT / 'utilities/q4-gemm/context_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured(tool, count, target=16):
    events = [dict(op='create', ok=True, held=i + 1) for i in range(count)]
    if count < target:
        events.append(dict(op='create', ok=False, held=count, error='NOAVAIL'))
    return dict(steps=tool.capacity_steps(['ln'], target, 'capacity'),
                result=dict(complete=True, returncode=0, events=events,
                            libraries_after={'coreutil': '/usr/lib/libxrt_coreutil.so.2.25.00'}))


@pytest.mark.parametrize('count,expected', [(16, 'NO_DECREASE'), (8, 'DECREASE_OBSERVED')])
def test_transition_requires_valid_paired_capacity(tool, count, expected):
    assert tool.transition_verdict(16, measured(tool, 16), measured(tool, count)) == expected
    assert tool.transition_verdict(16, measured(tool, 8), measured(tool, count)) == 'INVALID_BASELINE'


@pytest.mark.parametrize('mutation', [
    lambda r: r['result'].__setitem__('complete', False),
    lambda r: r['result'].__setitem__('returncode', 1),
    lambda r: r['result']['libraries_after'].__setitem__('coreutil', '/opt/libxrt_coreutil.so.2.26.0'),
    lambda r: r['result'].__setitem__('events', []),
    lambda r: r['result']['events'].pop(),
    lambda r: r['result']['events'][3].__setitem__('held', 2),
    lambda r: r.__setitem__('lifecycle', {'termination': 'watchdog timeout'}),
])
def test_invalid_probes_cannot_prove_capacity_or_trigger(tool, mutation):
    row = measured(tool, 16)
    mutation(row)
    assert tool.capacity_value(row, 'system') is None
    assert tool.transition_verdict(16, measured(tool, 16), row) == 'INCONCLUSIVE'


def test_post_churn_peak_does_not_hide_final_capacity_loss(tool):
    result = {'events': [dict(op='create', tag='churn-0', ok=True, held=16),
                         dict(op='create', tag='after-churn-0', ok=True, held=8)]}
    assert tool.churn_leaked(result, 16) is True


def plan_fixture(tool, tmp_path, monkeypatch):
    kernels = tmp_path / 'kernels'
    kernels.mkdir()
    artifact = kernels / 'final.xclbin'
    artifact.write_bytes(b'unchanged')
    manifest = kernels / 'manifest.json'
    manifest.write_text(json.dumps({'contexts': {'ln': 'final.xclbin'}}))
    monkeypatch.setattr(tool, 'ROOT', tmp_path)
    monkeypatch.setattr(tool, 'machine_state', lambda: {'boot_id': 'fixture-boot', 'boot_age_s': 60})
    monkeypatch.setattr(tool, 'pyxrt_python', lambda *a: sys.executable)
    monkeypatch.setattr(tool, 'xclbin_width', lambda *a: 8)
    monkeypatch.setattr(tool, 'journal_cursor', lambda: 'cursor')
    monkeypatch.setattr(tool, 'journal_window', lambda *a: {'available': True, 'stdout': ''})
    plan = dict(expected_capacity=16, inputs={str(p): tool.digest(p) for p in (manifest, artifact)},
                actions=[dict(kind='create-destroy', name='first', count=1),
                         dict(kind='create-destroy', name='must-not-run', count=1)])
    return kernels, plan


def test_aged_boot_refuses_before_probe_or_action(tool, tmp_path, monkeypatch):
    pytest.importorskip('fcntl', reason='transition driver requires POSIX locking')
    kernels, plan = plan_fixture(tool, tmp_path, monkeypatch)
    monkeypatch.setattr(tool, 'machine_state', lambda: {'boot_id': 'old', 'boot_age_s': 100000})
    monkeypatch.setattr(tool, 'run_arm', lambda *a, **k: pytest.fail('must not touch NPU on aged boot'))
    result = tool.record_transition(kernels, tmp_path/'result', plan, None)
    assert result['status'] == 'BLOCKED_FRESH_BOOT'
    assert not result['capacity'] and not result['actions']
    assert json.loads((tmp_path/'result/result.json').read_text())['status'] == 'BLOCKED_FRESH_BOOT'


@pytest.mark.parametrize('values,status', [([16, 8], 'STOPPED_PROBE_CONTROL'),
                                          ([16, 16, 16, 8], 'STOPPED_DECREASE_OBSERVED')])
def test_first_decrease_stops_remaining_sequence(tool, tmp_path, monkeypatch, values, status):
    pytest.importorskip('fcntl', reason='transition driver requires POSIX locking')
    kernels, plan = plan_fixture(tool, tmp_path, monkeypatch)
    observations = iter(values)
    actions = []

    def run_arm(assets, name, experiment, steps, *args, **kwargs):
        if experiment == 'capacity-control':
            return measured(tool, next(observations))
        actions.append(name)
        return measured(tool, 1, 1)

    monkeypatch.setattr(tool, 'run_arm', run_arm)
    result = tool.record_transition(kernels, tmp_path/'result', plan, None)
    assert result['status'] == status
    assert len(actions) == (1 if status == 'STOPPED_DECREASE_OBSERVED' else 0)
    assert all('must-not-run' not in name for name in actions)
    if actions:
        assert result['trigger'] == 'first'
        assert 'exact operation unproven' in result['attribution']


def test_changed_pinned_input_stops_before_npu_access(tool, tmp_path, monkeypatch):
    pytest.importorskip('fcntl', reason='transition driver requires POSIX locking')
    kernels, plan = plan_fixture(tool, tmp_path, monkeypatch)
    (kernels/'final.xclbin').write_bytes(b'tampered')
    monkeypatch.setattr(tool, 'run_arm', lambda *a, **k: pytest.fail('must not touch NPU on identity failure'))
    assert tool.record_transition(kernels, tmp_path/'result', plan, None)['status'] == 'BLOCKED_INPUT_IDENTITY'


def test_real_child_lifecycle_and_raw_streams(tool, tmp_path):
    row = tool.process_record([sys.executable, '-c', 'print("child"); raise SystemExit(3)'],
                             dict(__import__('os').environ), tmp_path/'child', timeout=10)
    assert row['pid'] > 0 and row['returncode'] == 3
    assert row['finished_monotonic_ns'] >= row['started_monotonic_ns']
    assert (tmp_path/'child/stdout.log').read_text().strip() == 'child'
    assert json.loads((tmp_path/'child/lifecycle.json').read_text())['returncode'] == 3


def test_worker_records_explicit_release_and_stops_after_first_create_failure(tool, tmp_path):
    # A Python fake drives the actual worker; no source scraping or NPU import.
    (tmp_path/'pyxrt.py').write_text('''
alive = 0
class device:
    def __init__(self, n): pass
    def register_xclbin(self, x): return x
def xclbin(p): return p
class hw_context:
    def __init__(self, d, u):
        global alive
        self.created = False
        if alive == 2: raise RuntimeError('NOAVAIL')
        alive += 1
        self.created = True
    def __del__(self):
        global alive
        if self.created: alive -= 1
''')
    cfg = tmp_path/'cfg.json'
    cfg.write_text(json.dumps(dict(assets=[dict(name='ln', path='fixture')],
                                  steps=tool.capacity_steps(['ln'], 8, 'create'), stop_on_failure=True)))
    output = tmp_path/'result.json'
    proc = subprocess.run([sys.executable, '-c', tool.WORKER, str(cfg), str(output)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    row = json.loads(output.read_text())
    assert len([e for e in row['events'] if e['op'] == 'create']) == 3
    assert row['events'][-1]['op'] == 'release-all'
    assert row['events'][-1]['held'] == 0
    assert row['pid'] > 0 and row['complete'] is True


def test_sequence_preparation_preserves_campaign_order_and_existing_identities(tool, tmp_path, monkeypatch):
    kernels, _ = plan_fixture(tool, tmp_path, monkeypatch)
    ids = tmp_path/'ids.txt'
    ids.write_text('1,2,3')
    binary = tmp_path/'cli'
    binary.write_bytes(b'fixture')
    argv = [str(binary), '--ids-file', str(ids)]
    campaign = tmp_path/'campaign.json'
    campaign.write_text(json.dumps(dict(provenance={str(binary): {'sha256': 'f'*64}},
                                       rows=[dict(arm='after', command=argv), dict(arm='before', command=argv)])))
    plan = tool.prepare_sequence(kernels, campaign)
    assert [a['name'] for a in plan['actions']] == ['process-00-after', 'process-01-before']
    assert plan['actions'][0]['argv'] == argv
    assert plan['inputs'][str(binary)] == 'f'*64  # Do not retag a changed binary as historical.
    assert plan['inputs'][str(ids)] == tool.digest(ids)


def test_transition_command_exits_nonzero_on_prerequisite_block(tool, tmp_path, monkeypatch):
    _, plan = plan_fixture(tool, tmp_path, monkeypatch)
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(tool, 'record_transition', lambda *a: {'status': 'BLOCKED_FRESH_BOOT', 'trigger': None})
    assert tool.main(['--experiment', 'transition', '--kernels', str(tmp_path), '--sequence', str(path),
                      '--out', str(tmp_path/'out')]) == 1


def test_journal_capture_preserves_raw_data_and_does_not_reuse_previous_arm(tool, monkeypatch):
    commands = []

    def capture(argv, env=None):
        commands.append(argv)
        return dict(returncode=0, stdout='PM: suspend entry\n', stderr='')

    monkeypatch.setattr(tool, 'capture', capture)
    result = tool.journal_window('exact-cursor')
    assert '--after-cursor=exact-cursor' in commands[0]
    assert result['stdout'] == 'PM: suspend entry\n'
    assert result['errors'] == []
    assert result['available'] is True
    assert tool.journal_window(None)['available'] is False
