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
import subprocess
import sys
import time
from pathlib import Path

RUNTIME = re.compile(r'\S*libxrt_[A-Za-z0-9_.+-]*\.so\S*|\S*pyxrt\S*\.so\S*')
VERSION = re.compile(r'(\d+\.\d+\.\d+)')


def runtime_paths(maps_text: str) -> list[str]:
    paths = []
    for line in maps_text.splitlines():
        match = re.search(r'(\S+)\s*$', line)
        if match and RUNTIME.fullmatch(match.group(1)):
            paths.append(match.group(1))
    return sorted(set(paths))


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


def capture(command: list[str], out: Path, env: dict, sample_s: float) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    record = dict(command=command, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  platform=platform.platform(),
                  environment={k: env.get(k) for k in (
                      'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'XILINX_XRT',
                      'XRT_ROOT', 'OFLM_OPEN_KERNELS_DIR', 'OFLM_OPEN_PREFILL_PRECISION',
                      'OFLM_OPEN_OMP_PASSIVE', 'OFLM_OPEN_LAYER_MAJOR')},
                  samples=[])
    with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
        proc = subprocess.Popen(command, cwd=str(Path.cwd()), env=env, stdout=stdout, stderr=stderr)
        record['pid'] = proc.pid
        seen: set[str] = set()
        deadline = time.monotonic() + sample_s
        while proc.poll() is None and time.monotonic() < deadline:
            try:
                text = Path(f'/proc/{proc.pid}/maps').read_text()
            except OSError:
                break
            fresh = set(runtime_paths(text))
            if fresh - seen:
                seen |= fresh
                record['samples'].append(dict(monotonic_s=round(time.monotonic() - deadline + sample_s, 3),
                                              paths=sorted(fresh)))
            time.sleep(0.02)
        if proc.poll() is None:
            record['termination'] = 'sampling window closed; child left running and reaped below'
            proc.terminate()
        record['returncode'] = proc.wait()
    (out / 'stdout.log').read_text()
    record['libraries'] = describe(sorted(seen))
    record['versions'] = sorted({v['version'] for v in record['libraries'].values() if v.get('version')})
    record['paths'] = sorted({v.get('resolved') for v in record['libraries'].values() if v.get('resolved')})
    (out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=1800,
                    help='how long to keep sampling a live child')
    ap.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
                    help='override one variable for the child')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        ap.error('a command is required after --')
    env = dict(os.environ)
    for item in args.env:
        key, _, value = item.partition('=')
        env[key] = value
    record = capture(command, args.out.resolve(), env, args.seconds)
    print(json.dumps(dict(returncode=record['returncode'], versions=record['versions'],
                          paths=record['paths']), indent=1))
    return 0 if record['returncode'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
