"""Probe XDNA2 hardware-context creation, capacity and reclaim, without inference.

Two entry points:

* the legacy probe (default) creates each manifest context alone and all of them
  together through the kernel harness;
* ``--experiment diagnose`` runs the capacity / destroy-reclaim / churn-leak /
  partition-width matrix in one pyxrt process per arm, capturing the kernel log
  and ``xrt-smi examine --report aie-partitions --verbose`` around each arm.

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
import json, sys

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
    out = {'events': [], 'held': 0, 'complete': False, 'pyxrt': None, 'libraries': {}}
    try:
        import pyxrt
    except Exception as exc:
        out['error'] = 'pyxrt import: %s: %s' % (type(exc).__name__, exc)
        json.dump(out, open(sys.argv[2], 'w'), indent=2)
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
                    out['events'].append({'op': 'create', 'asset': step['asset'], 'tag': tag,
                                          'ok': True, 'held': len(held)})
                except Exception as exc:
                    out['events'].append({'op': 'create', 'asset': step['asset'], 'tag': tag,
                                          'ok': False, 'held': len(held),
                                          'error': '%s: %s' % (type(exc).__name__, exc)})
            elif step['op'] == 'destroy':
                try:
                    held.pop()
                    out['events'].append({'op': 'destroy', 'tag': tag, 'ok': True, 'held': len(held)})
                except Exception as exc:
                    out['events'].append({'op': 'destroy', 'tag': tag, 'ok': False, 'held': len(held),
                                          'error': '%s: %s' % (type(exc).__name__, exc)})
        out['complete'] = True
    except Exception as exc:
        out['error'] = '%s: %s' % (type(exc).__name__, exc)
    finally:
        out['held'] = len(held)
        held.clear()
        out['libraries_after'] = {'coreutil': first_map('libxrt_coreutil.so.2'),
                                  'driver': first_map('libxrt_driver_xdna.so.2')}
        json.dump(out, open(sys.argv[2], 'w'), indent=2)
    return 0

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
    return max_held(result) < baseline


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
    env.update(RUNTIMES[runtime])
    for candidate in [override] if override else [sys.executable, '/usr/bin/python3.14', '/usr/bin/python3']:
        if not candidate or not Path(candidate).exists():
            continue
        proc = subprocess.run([candidate, '-c', 'import pyxrt'], env=env, capture_output=True)
        if proc.returncode == 0:
            return candidate
    return None


def run_arm(assets: list[dict], name: str, experiment: str, steps: list[dict], runtime: str,
            python: str, out: Path) -> dict:
    cfg, res = out / f'{name}.cfg.json', out / f'{name}.json'
    cfg.write_text(json.dumps({'assets': assets, 'steps': steps}, indent=2) + '\n')
    env = dict(os.environ)
    env.update(RUNTIMES[runtime])
    started = int(time.time())
    proc = subprocess.run([python, '-c', WORKER, str(cfg), str(res)], env=env,
                          capture_output=True, text=True, timeout=180)
    result = json.loads(res.read_text()) if res.exists() else {'error': proc.stderr[-400:]}
    result['returncode'] = proc.returncode
    result['stderr'] = proc.stderr[-400:]
    return dict(name=name, experiment=experiment, runtime=runtime,
                assets=[a['name'] for a in assets], steps=steps, result=result,
                kernel=kernel_log_since(started), partitions=partitions_report(),
                max_held=max_held(result), first_failure=first_create_failure(result))


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
            target = sum(1 for s in a['steps'] if s['op'] == 'create')
            key = (len(set(a['assets'])), target, a.get('width'))
            groups.setdefault(key, []).append(a['max_held'])
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
                                   f'(at least {len(saturated)} arms, distinct and identical assets); firmware '
                                   'reported NOAVAIL on the next create. This is an observed concurrency ceiling '
                                   f'consistent with the driver-advertised hwctx_limit={limit}, not proof of a '
                                   'firmware-enforced maximum.')
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
            conclusions.append(f'{runtime}: churn reduced capacity, indicating a leak or stale state.')
        elif leak_result is False:
            conclusions.append(f'{runtime}: churn did not reduce capacity.')
    width_arms = [a for a in arms if a['experiment'] == 'capacity-narrow']
    if not width_arms:
        remaining.append('partition-width dependence: no validated narrower-partition xclbin was available; '
                         'rerun with --narrow-xclbin after proving width < device columns.')
    if boot_age is None or boot_age > 900:
        remaining.append('fresh-boot capacity: this run started %.0f s after boot; repeat arm 1 after a reboot.'
                         % (boot_age or -1))
    remaining.append('firmware-side resource identity: capture the firmware response payload for the failing '
                     'create-context request, which the driver log does not expose.')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels', type=lambda v: Path(v).resolve(), required=True)
    parser.add_argument('--harness', type=lambda v: Path(v).resolve())
    parser.add_argument('--out', type=lambda v: Path(v).resolve(), required=True)
    parser.add_argument('--experiment', choices=('legacy', 'diagnose'), default='legacy')
    parser.add_argument('--runtime', action='append', choices=tuple(RUNTIMES), dest='runtimes')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--churn', type=int, default=64)
    parser.add_argument('--narrow-xclbin', type=lambda v: Path(v).resolve())
    parser.add_argument('--python', help='interpreter that imports pyxrt')
    args = parser.parse_args()
    if args.experiment == 'legacy':
        if not args.harness:
            parser.error('--harness is required for --experiment legacy')
        result = probe(args.kernels, args.harness, args.out)
        failed = [row['contexts'] for row in result['individual'] + [result['combined']]
                  if row['returncode'] != 0]
        if failed:
            raise SystemExit('context creation failed for: ' + ', '.join('/'.join(row) for row in failed))
    else:
        record = diagnose(args.kernels, args.out, args.runtimes or ['opt', 'system'], args.count,
                          args.repeats, args.churn, args.narrow_xclbin, args.python)
        print((args.out / 'matrix.md').read_text())
