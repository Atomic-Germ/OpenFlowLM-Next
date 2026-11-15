"""Probe manifest context creation independently and together, without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('kernels', 'harness', 'out'):
        parser.add_argument('--' + name, type=lambda v: Path(v).resolve(), required=True)
    args = parser.parse_args()
    result = probe(args.kernels, args.harness, args.out)
    failed = [row['contexts'] for row in result['individual'] + [result['combined']]
              if row['returncode'] != 0]
    if failed:
        raise SystemExit('context creation failed for: ' + ', '.join('/'.join(row) for row in failed))
