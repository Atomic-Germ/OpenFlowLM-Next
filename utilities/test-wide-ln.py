#!/usr/bin/env python3
"""Prepare/compare standalone residual RMSNorm hardware fixtures (no CPU fallback)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

GUARD = bytes([0xA5]) * 64


def metrics(got, ref):
    """The existing ln/compare.py bf16 gate, with explicit zero/finite handling."""
    g, r = got.astype(np.float64), ref.astype(np.float64)
    if g.shape != r.shape or not np.isfinite(g).all() or not np.isfinite(r).all():
        return dict(passed=False, reason='shape or nonfinite')
    scale = float(np.max(np.abs(r)))
    error = float(np.max(np.abs(g - r)))
    rel = error / scale if scale else (0.0 if error == 0 else None)
    gn, rn = np.linalg.norm(g), np.linalg.norm(r)
    cos = float((g / gn) @ (r / rn)) if gn and rn else float(gn == rn)
    ndiff = int(np.count_nonzero(got.view(np.uint16) != ref.view(np.uint16)))
    passed = rel is not None and rel < 8e-3 and cos > .999999 and ndiff < len(r) // 20
    return dict(passed=passed, max_relative=rel, cosine=cos, mismatches=ndiff)


def prepare(out, n, eps):
    if n <= 0 or n % 64 or not np.isfinite(eps) or eps <= 0:
        raise ValueError('width must be positive/divisible by 64; epsilon must be finite/positive')
    rng = np.random.default_rng(2962)
    w = (1 + rng.normal(0, .1, n)).astype(bfloat16)
    cases = [(f'random{i}', rng.normal(0, .5, n).astype(np.float32),
              rng.normal(0, .5, n).astype(np.float32)) for i in range(3)]
    x = cases[0][1]
    zero = np.zeros(n, np.float32)
    cases += [('entry', x, zero), ('cancel', x, -x),
              ('unequal_halves', np.concatenate([np.full(n//2, .125), np.full(n//2, 8)]).astype(np.float32), zero),
              ('ones', np.ones(n, np.float32), zero),
              ('tiny', x * np.float32(1e-5), zero)]
    for index in (n//2-1, n//2, n-1):
        impulse = zero.copy()
        impulse[index] = 1
        cases.append((f'impulse{index}', impulse, zero))
    cases.append(('repeat_first', cases[0][1], cases[0][2]))
    w.tofile(out / 'w.bin')
    (out / 'poison_y.bin').write_bytes(np.full(n, np.nan, np.float32).tobytes() + GUARD)
    (out / 'poison_xn.bin').write_bytes(np.full(n, np.nan, bfloat16).tobytes() + GUARD)
    cfg = ['device', 'xclbin ln final.xclbin', 'kernelx ln ln insts.bin',
           f'buf x {4*n}', f'buf add {4*n}', f'buf w {2*n} w.bin',
           f'buf y {4*n+len(GUARD)}', f'buf xn {2*n+len(GUARD)}']
    for i, (_, x, add) in enumerate(cases):
        x.tofile(out / f'x{i}.bin')
        add.tofile(out / f'add{i}.bin')
        y = x.astype(np.float64) + add.astype(np.float64)
        xn = (y / np.sqrt(np.mean(y*y) + eps) * w.astype(np.float64)).astype(np.float32).astype(bfloat16)
        y.astype(np.float32).tofile(out / f'ref_y{i}.bin')
        xn.tofile(out / f'ref_xn{i}.bin')
        cfg += [f'load x x{i}.bin', f'load add add{i}.bin', 'load y poison_y.bin',
                'load xn poison_xn.bin', 'run ln x add w y xn',
                f'dump y got_y{i}.bin {4*n+len(GUARD)}', f'dump xn got_xn{i}.bin {2*n+len(GUARD)}']
        for name in ('y', 'xn'):
            (out / f'got_{name}{i}.bin').unlink(missing_ok=True)
    (out / 'ln.cfg').write_text('\n'.join(cfg) + '\n')
    files = ['final.xclbin', 'insts.bin', 'ln.cfg', 'w.bin', 'poison_y.bin', 'poison_xn.bin']
    files += [f'{prefix}{i}.bin' for i in range(len(cases)) for prefix in ('x', 'add', 'ref_y', 'ref_xn')]
    meta = dict(n=n, eps=eps, seed=2962, cases=[c[0] for c in cases],
                sha256={f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files})
    (out / 'ln-fixture.json').write_text(json.dumps(meta, indent=2) + '\n')
    (out / 'ln-results.json').unlink(missing_ok=True)
    print(f'Prepared {len(cases)} inputs, N={n}, eps={eps}')


def compare(out):
    meta = json.loads((out / 'ln-fixture.json').read_text())
    for f, digest in meta['sha256'].items():
        if hashlib.sha256((out / f).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{f} changed since fixture generation')
    n, results = meta['n'], []
    for i, name in enumerate(meta['cases']):
        got = {}
        for tensor, dtype in (('y', np.float32), ('xn', bfloat16)):
            raw = (out / f'got_{tensor}{i}.bin').read_bytes()
            size = n * np.dtype(dtype).itemsize
            if len(raw) != size + len(GUARD) or raw[size:] != GUARD:
                raise ValueError(f'{name}/{tensor}: wrong output size or damaged canary')
            got[tensor] = np.frombuffer(raw[:size], dtype)
        ry = np.fromfile(out / f'ref_y{i}.bin', np.float32).astype(np.float64)
        gy = got['y'].astype(np.float64)
        finite = bool(np.isfinite(gy).all())
        error = float(np.max(np.abs(gy - ry))) if finite else None
        scale = float(np.max(np.abs(ry)))
        rel = error / scale if finite and scale else (0.0 if error == 0 else None)
        my = dict(passed=rel is not None and rel < 1e-6, max_relative=rel,
                  exact=bool(np.array_equal(gy, ry)))
        mx = metrics(got['xn'], np.fromfile(out / f'ref_xn{i}.bin', bfloat16))
        result = dict(case=name, passed=my['passed'] and mx['passed'], y=my, xn=mx)
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name}: y={my}, xn={mx}")
    # Repeated input after unrelated calls must produce bit-identical outputs.
    repeat = all((out / f'got_{t}0.bin').read_bytes() ==
                 (out / f'got_{t}{len(results)-1}.bin').read_bytes() for t in ('y', 'xn'))
    passed = repeat and all(r['passed'] for r in results)
    (out / 'ln-results.json').write_text(json.dumps(dict(passed=passed, repeat_exact=repeat,
                                                        checks=results), indent=2) + '\n')
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'compare'))
    p.add_argument('--build-dir', required=True, type=Path)
    p.add_argument('--width', type=int, default=5120)
    p.add_argument('--eps', type=float, default=1e-6)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.build_dir, args.width, args.eps)
        return 0
    return compare(args.build_dir)


if __name__ == '__main__':
    raise SystemExit(main())
