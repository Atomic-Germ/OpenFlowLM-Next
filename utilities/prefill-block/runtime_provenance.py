"""Record the XRT runtime a process actually loads, while it is alive.

`utilities/env.sh` puts `/opt/xilinx/xrt/lib` first, so the tree shell and the
distro XRT can disagree about which `libxrt_*` wins. The link-time `NEEDED`
entry (`libxrt_coreutil.so.2`) does not name a version, and the xdna plugin is
opened at run time, so neither `ldd` nor a package list answers the question.
`/proc/<pid>/maps` does: it lists the resolved paths the loader mapped.

    python runtime_provenance.py --out run/provenance -- <command...>
    python runtime_provenance.py --out run/provenance --env LD_LIBRARY_PATH=... -- <cmd>

The child is started once, sampled while it runs, and fully reaped. This is a
diagnostic: it reports paths, hashes and version strings; it asserts nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RUNTIME = re.compile(r'\S*libxrt_[A-Za-z0-9_.+-]*\.so\S*|\S*pyxrt\S*\.so\S*')
VERSION = re.compile(r'(\d+\.\d+\.\d+)')


def runtime_paths(maps_text: str) -> list[str]:
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and RUNTIME.fullmatch(fields[5]):
            paths.add(fields[5])
    return sorted(paths)


def process_paths(pid: int) -> list[str]:
    """Include inference children when the direct child is a timing wrapper."""
    pending, visited, paths = [pid], set(), set()
    while pending:
        child = pending.pop()
        if child in visited:
            continue
        visited.add(child)
        try:
            paths.update(runtime_paths(Path(f'/proc/{child}/maps').read_text()))
            pending.extend(map(int, Path(f'/proc/{child}/task/{child}/children').read_text().split()))
        except (OSError, ValueError):
            pass
    return sorted(paths)


def run_sampled(command: list[str], env: dict, timeout: float):
    """Collect output without blocking pipes and reap the process tree on timeout."""
    with tempfile.TemporaryFile(mode='w+') as stdout, tempfile.TemporaryFile(mode='w+') as stderr:
        proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                start_new_session=os.name == 'posix')
        seen, deadline = set(), time.monotonic() + timeout
        try:
            while proc.poll() is None:
                seen.update(process_paths(proc.pid))
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(0.1)
            proc.wait()
        finally:
            if proc.poll() is None:
                if os.name == 'posix':
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
            proc.wait()
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(command, proc.returncode,
                                           stdout.read(), stderr.read()), sorted(seen)


def describe(paths: list[str]) -> dict:
    out = {}
    for seen in paths:
        entry = dict(seen=seen)
        try:
            real = Path(seen).resolve(strict=True)
            with real.open('rb') as stream:
                entry.update(resolved=str(real), bytes=real.stat().st_size,
                             sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
            found = VERSION.findall(real.name)
            entry['version'] = found[-1] if found else None
        except OSError as exc:
            entry['error'] = str(exc)
        out[seen] = entry
    return out


def runtime_identity(libraries: dict) -> dict | None:
    """Missing core/plugin or an unhashable mapping is not verified provenance."""
    required = ('libxrt_coreutil.so.', 'libxrt_driver_xdna.so.')
    if not libraries or any(sum(Path(p).name.startswith(prefix) for p in libraries) != 1
                            for prefix in required):
        return None
    if any(v.get('error') or not re.fullmatch('[0-9a-f]{64}', v.get('sha256', ''))
           for v in libraries.values()):
        return None
    return {p: v['sha256'] for p, v in libraries.items()}


def capture(command: list[str], out: Path, env: dict, sample_s: float) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    record = dict(command=command, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  platform=platform.platform(),
                  environment={k: env.get(k) for k in (
                      'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'XILINX_XRT',
                      'XRT_ROOT', 'OFLM_OPEN_KERNELS_DIR', 'OFLM_OPEN_PREFILL_PRECISION',
                      'OFLM_OPEN_OMP_PASSIVE', 'OFLM_OPEN_LAYER_MAJOR')},
                  executable=str(Path(command[0]).resolve()))
    try:
        child, paths = run_sampled(command, env, sample_s)
        record['returncode'] = child.returncode
        (out / 'stdout.log').write_text(child.stdout)
        (out / 'stderr.log').write_text(child.stderr)
        record['libraries'] = describe(paths)
    except subprocess.TimeoutExpired:
        record.update(returncode=None, error='process timeout; process tree killed and reaped',
                      libraries={})
    record['versions'] = sorted({v['version'] for v in record['libraries'].values() if v.get('version')})
    record['paths'] = sorted({v.get('resolved') for v in record['libraries'].values() if v.get('resolved')})
    record['runtime_verified'] = runtime_identity(record['libraries']) is not None
    (out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=1800,
                    help='child timeout in seconds')
    ap.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
                    help='override one variable for the child')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        ap.error('a command is required after --')
    if not 0 < args.seconds < float('inf'):
        ap.error('--seconds must be positive and finite')
    env = dict(os.environ)
    for item in args.env:
        key, separator, value = item.partition('=')
        if not separator or not key:
            ap.error('--env requires KEY=VALUE')
        env[key] = value
    record = capture(command, args.out.resolve(), env, args.seconds)
    print(json.dumps(dict(returncode=record['returncode'], versions=record['versions'],
                          paths=record['paths']), indent=1))
    return 0 if record['returncode'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
