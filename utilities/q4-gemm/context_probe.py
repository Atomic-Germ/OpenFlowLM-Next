"""Probe XDNA2 context capacity/reclaim and record controlled process transitions.

Two entry points:

* the legacy probe (default) creates each manifest context alone and all of them
  together through the kernel harness;
* ``--experiment diagnose`` runs the capacity / destroy-reclaim / churn-leak /
  partition-width matrix in one pyxrt process per arm, capturing the kernel log
  and ``xrt-smi examine --report aie-partitions --verbose`` around each arm.
* ``--experiment transition --sequence plan.json`` brackets one foreground
  action at a time, after a fresh-boot, probe-only baseline. It stops at the
  first decrease, failed action or invalid measurement. It never performs a
  driver reload, power-state write, reboot or deliberate abnormal termination.

The diagnostic never changes firmware, driver parameters, precision or recipes.
It reports what was measured; it does not infer a firmware maximum from a
failure alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'Models/qwen38-27b/.npu.lock'

# Which libraries each arm links. The paths are resolved back from /proc/self/maps
# in the worker, so the recorded version is the one actually loaded, not the one
# a package manager claims.
RUNTIMES = {
    'opt': {'LD_LIBRARY_PATH': '/opt/xilinx/xrt/lib', 'PYTHONPATH': '/opt/xilinx/xrt/python'},
    'system': {'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu', 'PYTHONPATH': '/opt/xilinx/xrt/python'},
}

# amdxdna's aie2_msg_status values (drivers/accel/amdxdna/aie2_msg_priv.h).
STATUS_NAMES = {
    '0x2000001': 'AIE2_STATUS_MGMT_ERT_SELF_TEST_FAILURE',
    '0x2000002': 'AIE2_STATUS_MGMT_ERT_HASH_MISMATCH',
    '0x2000003': 'AIE2_STATUS_MGMT_ERT_NOAVAIL',
    '0x2000004': 'AIE2_STATUS_MGMT_ERT_INVALID_PARAM',
    '0x2000005': 'AIE2_STATUS_MGMT_ERT_ENTER_SUSPEND_FAILURE',
    '0x2000006': 'AIE2_STATUS_MGMT_ERT_BUSY',
    '0x2000007': 'AIE2_STATUS_MGMT_ERT_APPLICATION_ACTIVE',
    '0x2000008': 'AIE2_STATUS_MGMT_ERT_DRAM_BUFFER_SIZE_INVALID',
}

# One pyxrt arm. Contexts are created and destroyed explicitly in one process, so
# destroy/reclaim and churn are observable; the harness only has create.
WORKER = r'''
import gc, json, os, subprocess, sys, time

def first_map(pattern):
    try:
        with open('/proc/self/maps') as f:
            for line in f:
                if pattern in line:
                    return line.split()[-1]
    except OSError:
        pass
    return None

def main():
    cfg = json.load(open(sys.argv[1]))
    out = {'events': [], 'held': 0, 'complete': False, 'pyxrt': None, 'libraries': {},
           'pid': os.getpid(), 'started_ns': time.time_ns(),
           'started_monotonic_ns': time.monotonic_ns()}
    def save():
        with open(sys.argv[2] + '.tmp', 'w') as stream:
            json.dump(out, stream, indent=2)
        os.replace(sys.argv[2] + '.tmp', sys.argv[2])
    def event(row):
        row.update(wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
        command = cfg.get('telemetry_command')
        if command and row.get('snapshot'):
            try:
                report = subprocess.run(command, capture_output=True, text=True, timeout=15)
                row['partitions'] = {'command': command, 'returncode': report.returncode,
                                     'output': report.stdout + report.stderr}
            except (OSError, subprocess.TimeoutExpired) as exc:
                row['partitions'] = {'error': str(exc)}
        out['events'].append(row)
        save()
    try:
        import pyxrt
    except Exception as exc:
        out['error'] = 'pyxrt import: %s: %s' % (type(exc).__name__, exc)
        save()
        return 1
    out['pyxrt'] = pyxrt.__file__
    out['libraries'] = {'coreutil': first_map('libxrt_coreutil.so.2'),
                        'driver': first_map('libxrt_driver_xdna.so.2')}
    held = []
    try:
        dev = pyxrt.device(0)
        uuids = {}
        for asset in cfg['assets']:
            uuids[asset['name']] = dev.register_xclbin(pyxrt.xclbin(asset['path']))
        for step in cfg['steps']:
            tag = step.get('tag')
            if step['op'] == 'create':
                try:
                    held.append(pyxrt.hw_context(dev, uuids[step['asset']]))
                    event({'op': 'create', 'asset': step['asset'], 'tag': tag,
                           'ok': True, 'held': len(held), 'snapshot': step.get('snapshot', False)})
                except Exception as exc:
                    event({'op': 'create', 'asset': step['asset'], 'tag': tag,
                           'ok': False, 'held': len(held), 'snapshot': True,
                           'error': '%s: %s' % (type(exc).__name__, exc)})
                    if cfg.get('stop_on_failure'):
                        break
            elif step['op'] == 'destroy':
                try:
                    held.pop()
                    gc.collect()
                    event({'op': 'destroy', 'tag': tag, 'ok': True, 'held': len(held),
                           'snapshot': step.get('snapshot', False)})
                except Exception as exc:
                    event({'op': 'destroy', 'tag': tag, 'ok': False, 'held': len(held),
                           'error': '%s: %s' % (type(exc).__name__, exc)})
        out['complete'] = True
    except Exception as exc:
        out['error'] = '%s: %s' % (type(exc).__name__, exc)
    finally:
        out['held'] = len(held)
        out['libraries_after'] = {'coreutil': first_map('libxrt_coreutil.so.2'),
                                  'driver': first_map('libxrt_driver_xdna.so')}
        if cfg.get('release', 'explicit') == 'explicit':
            held.clear()
            gc.collect()
            event({'op': 'release-all', 'ok': True, 'held': 0, 'snapshot': True})
        out['finished_ns'] = time.time_ns()
        save()
    return 0 if out['complete'] else 1

sys.exit(main())
'''


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def boot_age_seconds() -> float | None:
    try:
        return float(Path('/proc/uptime').read_text().split()[0])
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- pure planning
def capacity_steps(assets: list[str], count: int, tag: str) -> list[dict]:
    """Create ``count`` contexts, cycling through ``assets`` in order."""
    if not assets or count <= 0:
        raise ValueError('capacity needs assets and a positive count')
    return [{'op': 'create', 'asset': assets[i % len(assets)], 'tag': f'{tag}-{i}'}
            for i in range(count)]


def reclaim_steps(asset: str, hold: int) -> list[dict]:
    """Hold ``hold`` contexts, destroy one, then try to create a replacement."""
    if hold <= 0:
        raise ValueError('reclaim needs a positive hold count')
    steps = capacity_steps([asset], hold, 'hold')
    steps.append({'op': 'destroy', 'tag': 'freed'})
    steps.append({'op': 'create', 'asset': asset, 'tag': 'replacement'})
    return steps


def churn_steps(asset: str, cycles: int, probe: int) -> list[dict]:
    """Create/destroy one context ``cycles`` times, then measure capacity again."""
    if cycles <= 0 or probe <= 0:
        raise ValueError('churn needs positive cycles and probe counts')
    steps = []
    for i in range(cycles):
        steps.append({'op': 'create', 'asset': asset, 'tag': f'churn-{i}'})
        steps.append({'op': 'destroy', 'tag': f'churn-{i}'})
    steps.extend(capacity_steps([asset], probe, 'after-churn'))
    return steps


def max_held(result: dict) -> int:
    return max((event.get('held', 0) for event in result.get('events', []) if event.get('ok')), default=0)


def first_create_failure(result: dict) -> dict | None:
    for event in result.get('events', []):
        if event.get('op') == 'create' and not event.get('ok'):
            return event
    return None


def reclaim_worked(result: dict) -> bool | None:
    events = result.get('events', [])
    freed = any(e.get('op') == 'destroy' and e.get('ok') for e in events)
    replacement = [e for e in events if e.get('tag') == 'replacement']
    if not replacement:
        return None
    return freed and replacement[-1].get('ok', False)


def churn_leaked(result: dict, baseline: int) -> bool | None:
    after = [e for e in result.get('events', []) if str(e.get('tag', '')).startswith('after-churn')]
    if not after:
        return None
    # A capacity loss is not itself proof of a leak. Ignore the earlier churn
    # peak: only the final capacity phase is comparable to the baseline.
    return max((e.get('held', 0) for e in after if e.get('ok')), default=0) < baseline


# ------------------------------------------------------------- log/report capture
KERNEL_ERROR = re.compile(r'amdxdna .*\[drm\] \*ERROR\*.*')
MGMT_STATUS = re.compile(r'command opcode (?P<opcode>0x[0-9a-fA-F]+) failed, '
                         r'status (?P<status>0x[0-9a-fA-F]+)')


def parse_kernel_log(text: str) -> list[dict]:
    """Keep amdxdna error lines, decoding management status codes when present."""
    rows = []
    for line in text.splitlines():
        if not KERNEL_ERROR.search(line):
            continue
        status = MGMT_STATUS.search(line)
        code = status.group('status') if status else None
        rows.append({'line': line.strip(),
                     'opcode': status.group('opcode') if status else None,
                     'status': code, 'status_name': STATUS_NAMES.get((code or '').lower())})
    return rows


def kernel_log_since(epoch: int) -> dict:
    journal = shutil.which('journalctl')
    if not journal:
        return {'available': False, 'reason': 'journalctl not found', 'lines': []}
    proc = subprocess.run([journal, '-k', '--since', f'@{epoch}', '--no-pager', '-o', 'short-iso'],
                          capture_output=True, text=True)
    if proc.returncode:
        return {'available': False, 'reason': proc.stderr.strip()[:400], 'lines': []}
    return {'available': True, 'lines': parse_kernel_log(proc.stdout)}


def capture(command: list[str], env: dict | None = None) -> dict:
    """Read-only diagnostic, preserving failures rather than treating them as silence."""
    started = time.time_ns()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=20)
        return dict(command=command, started_ns=started, finished_ns=time.time_ns(),
                    returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(command=command, started_ns=started, finished_ns=time.time_ns(), error=str(exc))


def journal_cursor() -> str | None:
    row = capture(['journalctl', '-k', '-b', '-n', '1', '--show-cursor', '--no-pager'])
    if row.get('returncode') != 0:
        return None
    match = re.search(r'^-- cursor: (.+)$', row['stdout'], re.MULTILINE)
    return match[1] if match else None


def journal_window(cursor: str | None) -> dict:
    if cursor is None:
        return {'available': False, 'reason': 'kernel journal cursor unavailable; absence of errors is unknown'}
    row = capture(['journalctl', '-k', '-b', '--after-cursor=' + cursor, '--no-pager', '-o', 'json'])
    row['available'] = row.get('returncode') == 0
    # Keep every entry, including PM, resets and successful allocations if the
    # operator enabled existing diagnostics. Only the error summary is filtered.
    row['errors'] = parse_kernel_log(row.get('stdout', ''))
    row['lines'] = row['errors']
    return row


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def machine_state() -> dict:
    device = Path('/sys/bus/pci/devices/0000:67:00.1')
    module = Path('/sys/module/amdxdna')
    return dict(wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                boot_id=read_optional(Path('/proc/sys/kernel/random/boot_id')),
                boot_age_s=boot_age_seconds(),
                driver_version=read_optional(module / 'version'),
                driver_srcversion=read_optional(module / 'srcversion'),
                power={name: read_optional(device / name) for name in
                       ('power_state', 'power/control', 'power/runtime_status',
                        'power/runtime_active_time', 'power/runtime_suspended_time',
                        'power/autosuspend_delay_ms')})


def process_record(command: list[str], env: dict, folder: Path, timeout: float = 1200) -> dict:
    """One foreground child, fully reaped, with raw streams and lifecycle evidence."""
    folder.mkdir()
    row = dict(command=command, cwd=str(ROOT), started_ns=time.time_ns(),
               started_monotonic_ns=time.monotonic_ns(), runtime_overrides={
                   key: env.get(key) for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'LD_PRELOAD')},
               settings={key: env.get(key) for key in (
                   'OMP_WAIT_POLICY', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                   'OFLM_OPEN_PREFILL_PRECISION', 'OFLM_OPEN_LAYER_MAJOR', 'OFLM_OPEN_OMP_PASSIVE',
                   'XRT_INI_PATH', 'XRT_ROOT', 'XILINX_XRT')}, libraries=[])
    with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
        try:
            proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stdout, stderr=stderr)
        except OSError as exc:
            row.update(error=str(exc), finished_ns=time.time_ns())
            (folder / 'lifecycle.json').write_text(json.dumps(row, indent=2) + '\n')
            return row
        row['pid'] = proc.pid
        deadline = time.monotonic() + timeout
        libraries = set()
        while proc.poll() is None:
            maps = read_optional(Path(f'/proc/{proc.pid}/maps')) or ''
            libraries.update(line.split()[-1] for line in maps.splitlines()
                             if '/' in line and ('libxrt_' in line or 'pyxrt' in line))
            if time.monotonic() >= deadline:
                row['termination'] = 'watchdog timeout (not graceful exit)'
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(0.02)
        row.update(returncode=proc.wait(), libraries=sorted(libraries), finished_ns=time.time_ns(),
                   finished_monotonic_ns=time.monotonic_ns())
    (folder / 'lifecycle.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


def partitions_report() -> dict:
    exe = shutil.which('xrt-smi')
    if not exe:
        return {'available': False, 'reason': 'xrt-smi not found', 'output': ''}
    proc = subprocess.run([exe, 'examine', '--report', 'aie-partitions', '--verbose'],
                          capture_output=True, text=True)
    return {'available': proc.returncode == 0, 'returncode': proc.returncode,
            'output': proc.stdout + proc.stderr}


def xclbin_width(path: Path) -> int | None:
    """Column width from AIE_PARTITION, via xclbinutil; None if unavailable."""
    exe = shutil.which('xclbinutil')
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'partition.json'
        proc = subprocess.run([exe, '--quiet', '--dump-section', f'AIE_PARTITION:JSON:{out}',
                               '--input', str(path), '--force'], capture_output=True)
        if proc.returncode or not out.exists():
            return None
        try:
            return int(json.loads(out.read_text())['aie_partition']['partition']['column_width'])
        except (KeyError, TypeError, ValueError):
            return None


# ---------------------------------------------------------------------- runtime
def pyxrt_python(runtime: str, override: str | None = None) -> str | None:
    """First interpreter that imports this runtime's pyxrt, with the arm's env."""
    env = dict(os.environ)
    env.pop('LD_PRELOAD', None)
    env.update(RUNTIMES[runtime])
    for candidate in [override] if override else [sys.executable, '/usr/bin/python3.14', '/usr/bin/python3']:
        if not candidate or not Path(candidate).exists():
            continue
        proc = subprocess.run([candidate, '-c', 'import pyxrt'], env=env, capture_output=True)
        if proc.returncode == 0:
            return candidate
    return None


def run_arm(assets: list[dict], name: str, experiment: str, steps: list[dict], runtime: str,
            python: str, out: Path, *, stop_on_failure: bool = False,
            observe: bool = False, release: str = 'explicit') -> dict:
    cfg, res = out / f'{name}.cfg.json', out / f'{name}.json'
    telemetry = (['/usr/bin/xrt-smi', 'examine', '--report', 'aie-partitions', '--verbose']
                 if observe else None)
    cfg.write_text(json.dumps(dict(assets=assets, steps=steps, stop_on_failure=stop_on_failure,
                                   telemetry_command=telemetry, release=release), indent=2) + '\n')
    env = dict(os.environ)
    env.update(RUNTIMES[runtime])
    before = machine_state()
    cursor = journal_cursor()
    lifecycle = process_record([python, '-c', WORKER, str(cfg), str(res)], env, out / f'{name}-process', 180)
    result = json.loads(res.read_text()) if res.exists() else {'complete': False, 'error': 'worker result missing'}
    result['returncode'] = lifecycle.get('returncode')
    return dict(name=name, experiment=experiment, runtime=runtime,
                assets=[s['asset'] for s in steps if s['op'] == 'create'], steps=steps, result=result,
                lifecycle=lifecycle, before=before, after=machine_state(),
                kernel=journal_window(cursor),
                max_held=max_held(result), first_failure=first_create_failure(result))


def capacity_value(arm: dict, runtime: str | None = None) -> int | None:
    """Capacity is unknown for failed setup, partial records or the wrong runtime."""
    result = arm.get('result', {})
    if (result.get('complete') is not True or result.get('returncode') != 0 or
            arm.get('lifecycle', {}).get('termination')):
        return None
    if runtime:
        core = result.get('libraries_after', {}).get('coreutil') or ''
        version = '2.25.' if runtime == 'system' else '2.26.'
        if not Path(core).name.startswith('libxrt_coreutil.so.' + version):
            return None
        plugins = [p for p in arm.get('lifecycle', {}).get('libraries', []) if 'driver_xdna' in p]
        if any(version not in Path(p).name for p in plugins):
            return None
    events = [e for e in result.get('events', []) if e.get('op') == 'create']
    if not events:
        return None
    held = 0
    failed = False
    for event in events:
        if event.get('ok'):
            if failed:
                return None
            held += 1
        else:
            failed = True
        if event.get('held') != held:
            return None
    planned = sum(s['op'] == 'create' for s in arm['steps'])
    return held if failed or len(events) == planned else None


def transition_verdict(baseline: int, before: dict, after: dict) -> str:
    old, new = capacity_value(before, 'system'), capacity_value(after, 'system')
    if old is None or new is None:
        return 'INCONCLUSIVE'
    if old != baseline:
        return 'INVALID_BASELINE'
    if new < old:
        return 'DECREASE_OBSERVED'
    return 'NO_DECREASE'


def check_sequence(plan: dict) -> None:
    """The recorder runs only named, hash-pinned foreground actions; no shell."""
    if plan.get('expected_capacity') != 16:
        raise ValueError('transition baseline must require 16 available contexts')
    if not isinstance(plan.get('inputs'), dict) or not plan['inputs']:
        raise ValueError('sequence needs SHA-256-pinned inputs')
    for path, identity in plan['inputs'].items():
        if not Path(path).is_absolute() or not re.fullmatch('[0-9a-f]{64}', identity):
            raise ValueError('input paths must be absolute with SHA-256 identities')
    if not isinstance(plan.get('actions'), list):
        raise ValueError('sequence needs an actions list')
    if (not isinstance(plan.get('env', {}), dict) or
            any(not isinstance(v, str) for v in plan.get('env', {}).values())):
        raise ValueError('sequence environment values must be strings')
    if not isinstance(plan.get('unset_env', []), list) or any(not isinstance(k, str) for k in plan.get('unset_env', [])):
        raise ValueError('unset_env must be a string list')
    names = set()
    for action in plan['actions']:
        name = action.get('name', '')
        if not re.fullmatch('[a-zA-Z0-9_-]+', name) or name in names:
            raise ValueError('action names must be unique safe filenames')
        names.add(name)
        kind = action.get('kind')
        if kind not in ('create-destroy', 'process-exit', 'process', 'idle', 'observer-control'):
            raise ValueError('unsupported action: no reset, reload, PM write or abnormal termination mode')
        if kind == 'process':
            command = action.get('argv')
            if (not isinstance(command, list) or not command or
                    any(not isinstance(arg, str) for arg in command) or command[0] not in plan['inputs']):
                raise ValueError('foreground executable must be pinned; argv must be a string list')
            if not 0 < action.get('timeout_s', 1200) <= 3600:
                raise ValueError('process watchdog must be 1..3600 seconds')
        if kind == 'idle' and not 0 <= action.get('seconds', -1) <= 600:
            raise ValueError('idle duration must be 0..600 seconds')
        if kind in ('create-destroy', 'process-exit') and not 1 <= action.get('count', 1) <= 16:
            raise ValueError('creation-only action count must be 1..16')


def prepare_sequence(kernels: Path, campaign: Path) -> dict:
    """Freeze the retained campaign's actual process order, not a shell loop."""
    data = json.loads(campaign.read_text())
    inputs = {path: row['sha256'] for path, row in data['provenance'].items()}
    manifest = kernels / 'manifest.json'
    contexts = json.loads(manifest.read_text())['contexts']
    for path in (campaign, manifest, kernels / contexts['ln']):
        inputs[str(path.resolve())] = digest(path)
    actions = []
    for index, row in enumerate(data['rows']):
        command = row['command']
        ids = Path(command[command.index('--ids-file') + 1])
        inputs[str(ids)] = digest(ids)
        actions.append(dict(kind='process', name=f'process-{index:02d}-{row["arm"]}',
                            argv=command, timeout_s=1800))
    plan = dict(expected_capacity=16, inputs=inputs, actions=actions,
                source_campaign=str(campaign),
                note='Original argv/order; reproduction deliberately pins installed XRT 2.25 for every process',
                env={'OPENBLAS_NUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE',
                     'OFLM_OPEN_OMP_PASSIVE': '1', 'OFLM_OPEN_PREFILL_PRECISION': 'paired',
                     'OFLM_OPEN_LAYER_MAJOR': '0'},
                unset_env=['LD_PRELOAD'])
    check_sequence(plan)
    return plan


def record_transition(kernels: Path, out: Path, plan: dict, python_override: str | None,
                      baseline_only: bool = False) -> dict:
    """Bracket each action with non-saturating, first-failure-stop capacity probes.

    Fresh-boot Phase A is fail-closed. Baseline-only observes the present boot;
    it never executes actions or calls that a reproduction of the transition.
    """
    import fcntl
    check_sequence(plan)
    out.mkdir(parents=True, exist_ok=False)
    start = machine_state()
    record = dict(status='PRECHECK', started=start, plan=plan, capacity=[], actions=[],
                  trigger=None, probe_side_effects='NOT MEASURED',
                  attribution='No transition identified', recovery='NOT TESTED; requires operator authorization')

    def save():
        record['finished'] = machine_state()
        temp = out / 'result.tmp'
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(out / 'result.json')

    lock = ROOT / 'Models/qwen38-27b/.npu.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        start = machine_state()
        record['started'] = start
        save()
        age = start['boot_age_s']
        if not baseline_only and (age is None or age > 900):
            record['status'] = 'BLOCKED_FRESH_BOOT'
            save()
            return record
        for path, identity in plan['inputs'].items():
            if digest(Path(path)) != identity:
                record.update(status='BLOCKED_INPUT_IDENTITY', changed_input=path)
                save()
                return record
        record['probe_sha256'] = digest(Path(__file__))
        manifest = kernels / 'manifest.json'
        contexts = json.loads(manifest.read_text())['contexts']
        asset_path = (kernels / contexts['ln']).resolve()
        if str(manifest.resolve()) not in plan['inputs'] or str(asset_path) not in plan['inputs']:
            raise ValueError('manifest and capacity artifact must be pinned')
        assets = [dict(name='ln', path=str(asset_path))]
        record['artifact_width'] = xclbin_width(asset_path)
        python = pyxrt_python('system', python_override)
        if not python:
            record['status'] = 'BLOCKED_PYXRT'
            save()
            return record
        expected = plan['expected_capacity']

        def measure(tag):
            steps = capacity_steps(['ln'], expected, tag)
            steps[-1]['snapshot'] = True
            arm = run_arm(assets, tag, 'capacity-control', steps, 'system', python, out,
                          stop_on_failure=True, observe=True)
            record['capacity'].append(arm)
            save()
            return arm

        # Three identical no-work controls detect obvious self-interference.
        # Agreement does NOT prove that the probe has no observer effects.
        previous = measure('baseline')
        if capacity_value(previous, 'system') != expected:
            record['status'] = 'BLOCKED_CAPACITY_BASELINE'
            save()
            return record
        for index in range(2):
            current = measure(f'probe-control-{index}')
            verdict = transition_verdict(expected, previous, current)
            if verdict != 'NO_DECREASE':
                record.update(status='STOPPED_PROBE_CONTROL', trigger=f'capacity probe {index}',
                              probe_side_effects=verdict)
                save()
                return record
            previous = current
        record['probe_side_effects'] = 'NO_DECREASE across three probes; observer effects not excluded'
        if baseline_only:
            record['status'] = 'BASELINE_ONLY'
            save()
            return record
        env = dict(os.environ, **RUNTIMES['system'])
        for name in plan.get('unset_env', []):
            env.pop(name, None)
        env.update(plan.get('env', {}))
        # Runtime bindings cannot be overridden by a benchmark action.
        env.update(RUNTIMES['system'])
        for index, action in enumerate(plan['actions']):
            if machine_state()['boot_id'] != start['boot_id']:
                record['status'] = 'STOPPED_BOOT_CHANGED'
                break
            changed = [p for p, h in plan['inputs'].items() if digest(Path(p)) != h]
            if changed:
                record.update(status='STOPPED_INPUT_CHANGED', changed_inputs=changed)
                break
            cursor = journal_cursor()
            row = dict(action=action, before=machine_state(), before_capacity=capacity_value(previous, 'system'))
            record['actions'].append(row)
            save()
            tag = f'action-{index}-{action["name"]}'
            kind = action['kind']
            if kind in ('create-destroy', 'process-exit'):
                steps = capacity_steps(['ln'], action.get('count', 1), 'create-only')
                steps[-1]['snapshot'] = True
                row['process'] = run_arm(assets, tag, kind, steps, 'system', python, out,
                                         stop_on_failure=True, observe=True,
                                         release='explicit' if kind == 'create-destroy' else 'process-exit')
                ok = capacity_value(row['process'], 'system') == action.get('count', 1)
            elif kind == 'process':
                row['process'] = process_record(action['argv'], env, out / tag,
                                                 action.get('timeout_s', 1200))
                ok = row['process'].get('returncode') == 0 and not row['process'].get('termination')
                libraries = row['process'].get('libraries', [])
                row['runtime_verified'] = (any('libxrt_coreutil.so.2.25.' in p for p in libraries) and
                                          not any('libxrt_' in p and '2.26.' in p for p in libraries))
                ok = ok and row['runtime_verified']
            elif kind == 'idle':
                time.sleep(action['seconds'])
                ok = True
            else:
                row['observer'] = capture(['/usr/bin/xrt-smi', 'examine', '--report',
                                            'aie-partitions', '--verbose'], env)
                ok = row['observer'].get('returncode') == 0
            row.update(after=machine_state(), kernel=journal_window(cursor))
            save()
            current = measure(f'after-{index}')
            row['comparison'] = transition_verdict(expected, previous, current)
            row['after_capacity'] = capacity_value(current, 'system')
            if row['comparison'] != 'NO_DECREASE':
                record.update(status='STOPPED_' + row['comparison'], trigger=action['name'],
                              attribution='Change bracketed by action AND following probe; exact operation unproven')
                break
            if not ok:
                record['status'] = 'STOPPED_ACTION_FAILED'
                break
            previous = current
        else:
            record['status'] = 'COMPLETED_NO_DECREASE'
        save()
    return record


# ---------------------------------------------------------------- classification
def classify(arms: list[dict], boot_age: float | None, narrow: dict | None) -> dict:
    """Summarise what the arms support, without promoting a failure to a proof.

    Capacity arms are grouped by the number of distinct assets they cycle and the
    number of contexts they request: comparing 11-of-11 distinct against 16-of-1
    single would confound the asset set with the target count.
    """
    conclusions, remaining, per_runtime = [], [], {}
    for runtime in sorted({a['runtime'] for a in arms}):
        rows = [a for a in arms if a['runtime'] == runtime]
        groups: dict[tuple, list[int]] = {}
        for a in rows:
            if not a['experiment'].startswith('capacity'):
                continue
            value = capacity_value(a)
            if value is None:
                conclusions.append(f'{runtime}: {a.get("name", a["experiment"])} is incomplete; capacity unknown.')
                continue
            target = sum(1 for s in a['steps'] if s['op'] == 'create')
            key = (len(set(a['assets'])), target, a.get('width'))
            groups.setdefault(key, []).append(value)
        reclaim = [reclaim_worked(a['result']) for a in rows if a['experiment'] == 'reclaim']
        reclaim_result = next((r for r in reclaim if r is not None), None)
        saturated = {k: v for k, v in groups.items() if any(n < k[1] for n in v)}
        churn_rows = [a for a in rows if a['experiment'] == 'churn']
        baseline = max((n for v in groups.values() for n in v), default=0)
        leak_result = next((churn_leaked(a['result'], baseline) for a in churn_rows
                            if churn_leaked(a['result'], baseline) is not None), None)
        statuses = sorted({line['status'] for a in rows for line in a['kernel'].get('lines', [])
                           if line.get('status')})
        per_runtime[runtime] = dict(capacity_by_assets_and_target={
                                        f'{k[0]} assets, target {k[1]}, width {k[2]}': v
                                        for k, v in groups.items()},
                                    capacity_returned_after_destroy=reclaim_result,
                                    leak_after_churn=leak_result, firmware_statuses=statuses,
                                    loaded_coreutil=next((a['result'].get('libraries', {}).get('coreutil')
                                                          for a in rows if a['result'].get('libraries')), None))
        if not groups:
            conclusions.append(f'{runtime}: no capacity arm ran.')
        elif saturated:
            parts = [f'{k[0]} distinct asset(s), target {k[1]}: max held {v}' for k, v in saturated.items()]
            ceiling = {max(v) for v in saturated.values()}
            consistent = (len(ceiling) == 1 and
                          all(n == k[1] - 1 for k, v in saturated.items() for n in v))
            if consistent:
                limit = ceiling.pop()
                conclusions.append(f'{runtime}: every over-target arm stopped at exactly {limit} contexts '
                                   f'in the comparable groups. Captured status values: {statuses}. '
                                   'This is an observation, not proof of a firmware maximum, '
                                   'driver-advertised limit, leak or specific exhausted resource.')
            else:
                conclusions.append(f'{runtime}: capacity was reached before the target ({" ; ".join(parts)}). '
                                   'A fixed maximum is not established across asset sets.')
        else:
            parts = [f'{k[0]} distinct asset(s), target {k[1]}: max held {v}' for k, v in groups.items()]
            conclusions.append(f'{runtime}: every capacity arm held its full target, so no cutoff was '
                               f'reached ({" ; ".join(parts)}).')
        if reclaim_result is True:
            conclusions.append(f'{runtime}: destroying one context allowed a replacement, so capacity is returned.')
        elif reclaim_result is False:
            conclusions.append(f'{runtime}: destroying one context did not allow a replacement.')
        if leak_result is True:
            conclusions.append(f'{runtime}: lower post-churn capacity observed; leak versus stale state is unresolved.')
        elif leak_result is False:
            conclusions.append(f'{runtime}: no post-churn capacity reduction observed in this arm; leaks not excluded.')
    width_arms = [a for a in arms if a['experiment'] == 'capacity-narrow']
    if not width_arms:
        remaining.append('partition-width dependence: no validated narrower-partition xclbin was available; '
                         'rerun with --narrow-xclbin after proving width < device columns.')
    if boot_age is None or boot_age > 900:
        remaining.append('fresh-boot capacity: this run started %.0f s after boot; repeat arm 1 after a reboot.'
                         % (boot_age or -1))
    remaining.append('transition attribution: bracket each process from a fresh-boot baseline, including '
                     'probe-only controls and runtime-PM observations. A failed management reply has no '
                     'documented exhausted-resource reason beyond its status; use existing logs/traces.')
    remaining.append('contention: repeat the capacity arm while another process holds a context.')
    return dict(per_runtime=per_runtime, conclusions=conclusions, remaining_experiments=remaining,
                firmware_max_proven=False,
                narrow_artifact=narrow, fresh_boot=boot_age is not None and boot_age <= 900)


def matrix(arms: list[dict]) -> str:
    lines = ['| Experiment | Runtime | Asset width | Hold target | Max held | First failure | FW status | Capacity returned |',
             '|---|---|---:|---:|---:|---|---|---|']
    for a in arms:
        failure = a.get('first_failure') or {}
        statuses = sorted({l['status'] for l in a['kernel'].get('lines', []) if l.get('status')})
        target = sum(1 for s in a['steps'] if s['op'] == 'create')
        returned = reclaim_worked(a['result']) if a['experiment'] == 'reclaim' else None
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            a['experiment'], a['runtime'], a.get('width', 'mixed'), target, a['max_held'],
            failure.get('error', '—'), ', '.join(statuses) or '—',
            {True: 'yes', False: 'no', None: 'n/a'}[returned]))
    return '\n'.join(lines)


# ----------------------------------------------------------------- legacy probe
def probe(kernels: Path, harness: Path, out: Path) -> dict:
    import fcntl
    manifest = kernels / 'manifest.json'
    contexts = json.loads(manifest.read_text())['contexts']
    out.mkdir(parents=True, exist_ok=False)
    paths = [manifest, harness, *(kernels / p for p in contexts.values())]
    hashes = {}
    for path in paths:
        with path.open('rb') as stream:
            hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(inputs=hashes, individual=[], combined=None,
                  scope='context creation only; no kernel execution or model acceptance')
    lock_path = ROOT / 'Models/qwen38-27b/.npu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cases = [(f'context-{i}', [(name, path)]) for i, (name, path) in enumerate(contexts.items())]
        cases.append(('combined', list(contexts.items())))
        for tag, entries in cases:
            cfg = out / f'{tag}.cfg'
            cfg.write_text('device\n' + ''.join(f'xclbin {name} {kernels / path}\n' for name, path in entries))
            command = [str(harness), str(cfg)]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
            log = proc.stdout + proc.stderr
            (out / f'{tag}.log').write_text(log)
            row = dict(command=command, contexts=[name for name, _ in entries],
                       returncode=proc.returncode, output=log)
            if tag == 'combined':
                result['combined'] = row
            else:
                result['individual'].append(row)
            (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


# ------------------------------------------------------------------- diagnostic
def diagnose(kernels: Path, out: Path, runtimes: list[str], count: int, repeats: int,
             churn: int, narrow: Path | None, python_override: str | None) -> dict:
    import fcntl
    manifest = kernels / 'manifest.json'
    contexts = json.loads(manifest.read_text())['contexts']
    out.mkdir(parents=True, exist_ok=False)
    assets = [{'name': name, 'path': str(kernels / rel)} for name, rel in contexts.items()]
    widths = {a['name']: xclbin_width(Path(a['path'])) for a in assets}
    narrow_record = None
    if narrow:
        width = xclbin_width(narrow)
        narrow_record = {'path': str(narrow), 'sha256': digest(narrow), 'width': width}
        if width is None or width < 1 or width >= (max(w for w in widths.values() if w) if any(widths.values()) else 0):
            narrow_record['rejected'] = 'width is not narrower than the device partition'
    lock_path = ROOT / 'Models/qwen38-27b/.npu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started_age = boot_age_seconds()
    arms = []
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for runtime in runtimes:
            python = pyxrt_python(runtime, python_override)
            if not python:
                arms.append({'name': f'{runtime}-unavailable', 'experiment': 'runtime', 'runtime': runtime,
                             'assets': [], 'steps': [], 'result': {'error': 'pyxrt unavailable'},
                             'kernel': {'lines': []}, 'partitions': {}, 'max_held': 0, 'first_failure': None})
                continue
            full = assets[0]['name']
            distinct = [a['name'] for a in assets]
            plan = [('capacity-full', capacity_steps(distinct, count, 'full'), widths[full])]
            plan += [(f'capacity-repeat-{i}', capacity_steps([full], count, f'repeat{i}'), widths[full])
                     for i in range(repeats)]
            plan += [('capacity-identical', capacity_steps([full], count, 'same'), widths[full]),
                     ('reclaim', reclaim_steps(full, 8), widths[full]),
                     ('churn', churn_steps(full, churn, count), widths[full])]
            if narrow_record and 'rejected' not in narrow_record:
                name = 'narrow'
                plan.append(('capacity-narrow', capacity_steps([name], count, 'narrow'), narrow_record['width']))
                arm_assets = assets + [{'name': name, 'path': str(narrow)}]
            else:
                arm_assets = assets
            for experiment, steps, width in plan:
                arm = run_arm(arm_assets, f'{runtime}-{experiment}', experiment, steps, runtime, python, out)
                arm['width'] = width
                arms.append(arm)
                (out / 'result.json').write_text(json.dumps(
                    dict(boot_age_s=started_age, assets=assets, widths=widths, narrow=narrow_record,
                         arms=arms), indent=2) + '\n')
    summary = classify(arms, started_age, narrow_record)
    record = dict(boot_age_s=started_age, assets=assets, widths=widths, narrow=narrow_record,
                  arms=arms, summary=summary)
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    (out / 'matrix.md').write_text(matrix(arms) + '\n\n' + '\n'.join('- ' + c for c in summary['conclusions']) +
                                   '\n\nRemaining experiments:\n' +
                                   '\n'.join('- ' + r for r in summary['remaining_experiments']) + '\n')
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels', type=lambda v: Path(v).resolve(), required=True)
    parser.add_argument('--harness', type=lambda v: Path(v).resolve())
    parser.add_argument('--out', type=lambda v: Path(v).resolve(), required=True)
    parser.add_argument('--experiment', choices=('legacy', 'diagnose', 'transition', 'prepare'), default='legacy')
    parser.add_argument('--runtime', action='append', choices=tuple(RUNTIMES), dest='runtimes')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--churn', type=int, default=64)
    parser.add_argument('--narrow-xclbin', type=lambda v: Path(v).resolve())
    parser.add_argument('--python', help='interpreter that imports pyxrt')
    parser.add_argument('--sequence', type=Path, help='hash-pinned JSON foreground process sequence')
    parser.add_argument('--campaign', type=lambda v: Path(v).resolve(), help='retained benchmark result.json to replay')
    parser.add_argument('--baseline-only', action='store_true',
                        help='observe current capacity without actions; not a fresh-boot transition experiment')
    args = parser.parse_args(argv)
    if args.experiment == 'prepare':
        if not args.campaign:
            parser.error('--campaign is required for sequence preparation')
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / 'plan.json').write_text(json.dumps(prepare_sequence(args.kernels, args.campaign), indent=2) + '\n')
        print(args.out / 'plan.json')
    elif args.experiment == 'transition':
        if not args.sequence:
            parser.error('--sequence is required for transition recording')
        record = record_transition(args.kernels, args.out, json.loads(args.sequence.read_text()),
                                   args.python, args.baseline_only)
        print(json.dumps({'status': record['status'], 'trigger': record['trigger']}))
        return 0 if record['status'] in ('COMPLETED_NO_DECREASE', 'BASELINE_ONLY') else 1
    elif args.experiment == 'legacy':
        if not args.harness:
            parser.error('--harness is required for --experiment legacy')
        result = probe(args.kernels, args.harness, args.out)
        failed = [row['contexts'] for row in result['individual'] + [result['combined']]
                  if row['returncode'] != 0]
        if failed:
            print('context creation failed for: ' + ', '.join('/'.join(row) for row in failed), file=sys.stderr)
            return 1
    else:
        record = diagnose(args.kernels, args.out, args.runtimes or ['opt', 'system'], args.count,
                          args.repeats, args.churn, args.narrow_xclbin, args.python)
        print((args.out / 'matrix.md').read_text())
    return 0


if __name__ == '__main__':
    sys.exit(main())
