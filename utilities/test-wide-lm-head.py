#!/usr/bin/env python3
"""Synthetic full-vocabulary Q8 LM-head hardware acceptance using production packing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'open_kernels'))
sys.path.insert(0, str(ROOT / 'utilities'))
from recipes import pack
from wide_deltanet_reference import metric

_spec = importlib.util.spec_from_file_location('lm_head_oracle', ROOT / 'open_kernels/designs/lm_head_q8/make_test.py')
ORACLE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ORACLE)
GUARD = bytes([0xA5]) * 64


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(out, k, n, cores):
    if k <= 0 or k % 256 or n <= 0 or n % 128 or not 1 <= cores <= n // 128:
        raise ValueError('K must tile 256, N must tile 128, and every core needs a band')
    rng = np.random.default_rng(2963)
    # Bound working memory even for the full 1.35 GB head. Each block is a
    # whole number of 128-row bands; packing therefore composes exactly.
    class Model:
        def raw(self, name): return raw
    with (out / 'weights.bin').open('wb') as f:
        for row in range(0, n, 1024):
            rows = min(1024, n-row)
            chunks = rows // 32 * (k // 256)
            raw = np.empty((chunks, 8704), np.uint8)
            scales = rng.uniform(.001, .02, (chunks, 256)).astype(bfloat16)
            raw[:, :512] = scales.view(np.uint8).reshape(chunks, 512)
            raw[:, 512:] = rng.integers(-128, 128, (chunks, 8192), dtype=np.int8).view(np.uint8)
            pool = np.empty(raw.size, np.uint8)
            pack.apply_op(dict(op='lmhead_q8', tensor='lm_head.weight', dst=0,
                               in_dim=k, chunk_bytes=8704), Model(), 0, pool)
            f.write(pool.tobytes())
    inputs = [(f'random{i}', rng.normal(0, .6, k).astype(bfloat16)) for i in range(2)]
    inputs += [('ones', np.ones(k, bfloat16)), ('zero', np.zeros(k, bfloat16))]
    for col in sorted({0, 255, 256, 2047, 2048, 4095, 4096, k-1}):
        if col >= k:
            continue
        x = np.zeros(k, bfloat16)
        x[col] = 1
        inputs.append((f'impulse{col}', x))
    inputs.append(('repeat_first', inputs[0][1]))
    print(f'Packed N={n}, K={k}; computing {len(inputs)} independent references', flush=True)
    weights = np.memmap(out / 'weights.bin', dtype=np.uint8, mode='r')
    refs = ORACLE.references(weights, np.stack([x for _, x in inputs]), n)
    (out / 'poison.bin').write_bytes(np.full(n, np.nan, np.float32).tobytes() + GUARD)
    cfg = ['device', 'xclbin lm final.xclbin', 'kernelx lm lm insts.bin',
           f'buf w {weights.size} weights.bin', f'buf x {2*k}', f'buf y {4*n+len(GUARD)}']
    for i, (_, x) in enumerate(inputs):
        x.tofile(out / f'x{i}.bin')
        refs[i].tofile(out / f'ref{i}.bin')
        cfg += [f'load x x{i}.bin', 'load y poison.bin', 'run lm w x y',
                f'dump y got{i}.bin {4*n+len(GUARD)}']
        (out / f'got{i}.bin').unlink(missing_ok=True)
    (out / 'lm.cfg').write_text('\n'.join(cfg) + '\n')
    files = ['final.xclbin', 'insts.bin', 'weights.bin', 'lm.cfg', 'poison.bin']
    files += [f'{prefix}{i}.bin' for i in range(len(inputs)) for prefix in ('x', 'ref')]
    meta = dict(k=k, n=n, cores=cores, seed=2963, cases=[name for name, _ in inputs],
                sha256={f: digest(out / f) for f in files})
    (out / 'lm-fixture.json').write_text(json.dumps(meta, indent=2) + '\n')
    (out / 'lm-results.json').unlink(missing_ok=True)
    print(f'Prepared {len(inputs)} inputs, pool={weights.size} bytes', flush=True)


def compare(out):
    meta = json.loads((out / 'lm-fixture.json').read_text())
    for f, sha in meta['sha256'].items():
        if digest(out / f) != sha:
            raise ValueError(f'{f} changed since fixture generation')
    n, cores = meta['n'], meta['cores']
    bands, remainder = divmod(n // 128, cores)
    counts = [128 * (bands + int(c < remainder)) for c in range(cores)]
    offsets = np.cumsum([0] + counts)
    results = []
    for i, name in enumerate(meta['cases']):
        raw = (out / f'got{i}.bin').read_bytes()
        if len(raw) != 4*n+len(GUARD) or raw[4*n:] != GUARD:
            raise ValueError(f'{name}: wrong output size or damaged canary')
        got = np.frombuffer(raw[:4*n], np.float32)
        ref = np.fromfile(out / f'ref{i}.bin', np.float32)
        whole = metric(got, ref, .9999999)
        by_core = [metric(got[a:b], ref[a:b], .9999999) for a, b in zip(offsets[:-1], offsets[1:])]
        ok = whole['passed'] and all(c['passed'] for c in by_core)
        results.append(dict(case=name, passed=ok, whole=whole, cores=by_core))
        print(f"{'PASS' if ok else 'FAIL'} {name}: {whole}")
    repeat = (out / 'got0.bin').read_bytes() == (out / f'got{len(results)-1}.bin').read_bytes()
    passed = repeat and all(c['passed'] for c in results)
    (out / 'lm-results.json').write_text(json.dumps(dict(passed=passed, repeat_exact=repeat,
                                                        checks=results), indent=2) + '\n')
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'compare'))
    p.add_argument('--build-dir', required=True, type=Path)
    p.add_argument('--width', type=int, default=5120)
    p.add_argument('--rows', type=int, default=248320)
    p.add_argument('--cores', type=int, default=8)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.build_dir, args.width, args.rows, args.cores)
        return 0
    return compare(args.build_dir)


if __name__ == '__main__':
    raise SystemExit(main())
