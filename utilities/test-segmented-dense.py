#!/usr/bin/env python3
"""Synthetic acceptance of segmented Q4 down and full dense FFN on actual geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'open_kernels'))
from recipes.spec import ModelSpec
from recipes import qwen35 as Q
from q4_1_pack import pack_q4_1_pool, random_q4_1_blocks, chunk_geometry, dequant_chunk
from wide_deltanet_reference import metric

GUARD = bytes([0xA5]) * 64


def reference(pool, inputs, n, k, partials=False):
    """Independent float64 math directly from original pool chunk coordinates."""
    _, _, rows, cols = chunk_geometry(n, k, 2)
    nseg = (k + 8191) // 8192 if partials else 1
    sums = np.zeros((len(inputs), nseg, n), np.float64)
    x = inputs.astype(bfloat16).astype(np.float64)
    for c, (r, col) in enumerate(zip(rows, cols)):
        w = dequant_chunk(pool[c * 5120:(c + 1) * 5120]).astype(np.float64)
        sums[:, col // 8192 if partials else 0, r:r + 32] += x[:, col:col + 256] @ w.T
    return sums.cumsum(axis=1).astype(np.float32)


def prepare(out):
    spec = ModelSpec.from_dict(json.loads((out / 'probe-spec.json').read_text()))
    tool = json.loads((out / 'probe-toolchain.json').read_text())
    l, f = Q.layout(spec), Q.ffn_geometry(spec)
    full = tool['scope'] == 'ffn'
    trace = full and tool.get('trace', False)
    extra_buffer = tool.get('buffer_args', 3 if trace else 2) == 3
    snapshots = len(f.DOWN_SEGMENTS) if not full and not tool['final_only'] else 1
    act_bytes = max(l.A_BYTES, l.A_OUT2 + spec.hidden * 4 * len(f.DOWN_SEGMENTS))
    rng = np.random.default_rng(38417)
    k, n = spec.intermediate, spec.hidden
    def weights(rows, cols):
        return pack_q4_1_pool(random_q4_1_blocks(rows, cols, rng, scale=.002), 2)
    poolfile = out / 'pool.bin'
    with poolfile.open('wb') as stream:
        stream.truncate(l.POOL_BYTES)
    pool = np.memmap(poolfile, mode='r+', dtype=np.uint8)
    down = weights(n, k)
    pool[l.POOL_FFN_DOWN:l.POOL_FFN_DOWN + len(down)] = down
    if full:
        inputs = rng.normal(0, .6, (2, n)).astype(bfloat16)
        up, gate = weights(k, n), weights(k, n)
        pool[l.POOL_FFN_UP:l.POOL_FFN_UP + len(up)] = up
        pool[l.POOL_FFN_GATE:l.POOL_FFN_GATE + len(gate)] = gate
        # Keep the original two inputs and weights unchanged, then broaden the
        # suite using the RNG state AFTER weight generation.
        extra = list(rng.normal(0, .6, (4, n)).astype(bfloat16))
        extra += [np.zeros(n, bfloat16), np.ones(n, bfloat16), -np.ones(n, bfloat16)]
        for index in (2047, 2048, n - 1):
            x = np.zeros(n, bfloat16)
            x[index] = 1
            extra.append(x)
        inputs = np.concatenate([inputs, np.array(extra), inputs[:1]])
        u, g = reference(up, inputs, k, n)[:, 0], reference(gate, inputs, k, n)[:, 0]
        h = (u.astype(np.float64) * g / (1 + np.exp(-g.astype(np.float64)))).astype(np.float32)
        refs = reference(down, h, n, k)[:, -1:]
        for i in range(len(inputs)):
            if trace:
                u[i].tofile(out / f'uref{i}.bin')
                g[i].tofile(out / f'gref{i}.bin')
            h[i].tofile(out / f'href{i}.bin')
    else:
        inputs = list(rng.normal(0, .6, (2, k)).astype(np.float32))
        inputs += [np.ones(k, np.float32), np.zeros(k, np.float32)]
        for index in (0, 8191, 8192, 16383, 16384, k - 1):
            if index >= k:
                continue
            x = np.zeros(k, np.float32)
            x[index] = 1
            inputs.append(x)
        inputs.append(inputs[0].copy())  # deterministic repeat after other inputs
        inputs = np.array(inputs)
        refs = reference(down, inputs, n, k, partials=True)
        if snapshots == 1:
            refs = refs[:, -1:]
    pool.flush()
    del pool
    cfg = ['device', 'xclbin p final.xclbin', 'kernelx p p insts.bin',
           f'buf w {l.POOL_BYTES} pool.bin', f'buf a {act_bytes + len(GUARD)}']
    trace_bytes = (2 * k if trace else 1) * 4
    if extra_buffer:
        (out / 'trace-poison.bin').write_bytes(np.full(trace_bytes // 4, np.nan, np.float32).tobytes() + GUARD)
        cfg += [f'buf t {trace_bytes + len(GUARD)}']
    for i, x in enumerate(inputs):
        raw = bytearray(np.full(act_bytes // 4, np.nan, np.float32).tobytes() + GUARD)
        pos = l.A_XM if full else l.A_H
        data = x.tobytes()
        raw[pos:pos + len(data)] = data
        (out / f'act{i}.bin').write_bytes(raw)
        refs[i].tofile(out / f'ref{i}.bin')
        cfg += [f'load a act{i}.bin']
        if extra_buffer:
            cfg += ['load t trace-poison.bin']
        cfg += ['run p w a' + (' t' if extra_buffer else ''), f'dump a got{i}.bin {len(raw)}']
        if trace:
            cfg += [f'dump t trace{i}.bin {trace_bytes + len(GUARD)}']
            (out / f'trace{i}.bin').unlink(missing_ok=True)
        (out / f'got{i}.bin').unlink(missing_ok=True)
    (out / 'segmented.cfg').write_text('\n'.join(cfg) + '\n')
    files = ['final.xclbin', 'insts.bin', 'pool.bin', 'segmented.cfg']
    files += [f'{prefix}{i}.bin' for i in range(len(inputs)) for prefix in ('act', 'ref')]
    if full:
        files += [f'href{i}.bin' for i in range(len(inputs))]
    if trace:
        files += [f'{prefix}{i}.bin' for i in range(len(inputs)) for prefix in ('uref', 'gref')]
    meta = dict(k=k, n=n, inputs=len(inputs), full=full, trace=trace, repeated_last=True, snapshots=snapshots,
                act_bytes=act_bytes, out_offset=l.A_OUT2, h_offset=l.A_H, seed=38417,
                sha256={name: digest(out / name) for name in files})
    (out / 'segmented-fixture.json').write_text(json.dumps(meta, indent=2) + '\n')
    (out / 'segmented-results.json').unlink(missing_ok=True)
    print(f'Prepared {len(inputs)} inputs; K={k}, N={n}, full={full}, snapshots={snapshots}', flush=True)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(out):
    meta = json.loads((out / 'segmented-fixture.json').read_text())
    for name, value in meta['sha256'].items():
        if digest(out / name) != value:
            raise ValueError(f'{name} changed since fixture generation')
    results, diagnostics = [], []
    if meta['full']:
        s = ModelSpec.from_dict(json.loads((out / 'probe-spec.json').read_text()))
        l = Q.layout(s)
        pool = np.memmap(out / 'pool.bin', mode='r', dtype=np.uint8)
        down = pool[l.POOL_FFN_DOWN:l.POOL_FFN_DOWN + meta['n'] * meta['k'] // 8192 * 5120]
    for i in range(meta['inputs']):
        raw = (out / f'got{i}.bin').read_bytes()
        if len(raw) != meta['act_bytes'] + len(GUARD) or raw[-len(GUARD):] != GUARD:
            raise ValueError(f'input {i}: wrong length or damaged canary')
        got = np.frombuffer(raw, np.float32, count=meta['snapshots'] * meta['n'], offset=meta['out_offset']).reshape(meta['snapshots'], meta['n'])
        ref = np.fromfile(out / f'ref{i}.bin', np.float32).reshape(got.shape)
        for j, (g, r) in enumerate(zip(got, ref)):
            m = metric(g, r, .9999999)
            results.append(dict(input=i, snapshot=j, **m))
        if meta['full']:
            h = np.frombuffer(raw, np.float32, count=meta['k'], offset=meta['h_offset'])
            href = np.fromfile(out / f'href{i}.bin', np.float32)
            results.append(dict(input=i, tensor='h', **metric(h, href, .9999999)))
            if meta.get('trace'):
                traw = (out / f'trace{i}.bin').read_bytes()
                if len(traw) != meta['k'] * 8 + len(GUARD) or traw[-len(GUARD):] != GUARD:
                    raise ValueError('invalid up/gate trace length or canary')
                t = np.frombuffer(traw[:-len(GUARD)], np.float32).reshape(-1, 2, 64)
                u, g = t[:, 0, :].ravel(), t[:, 1, :].ravel()
                for name, value in [('u', u), ('g', g)]:
                    results.append(dict(input=i, tensor=name, **metric(value, np.fromfile(out / f'{name}ref{i}.bin', np.float32), .9999999)))
                local_h = (u.astype(np.float64) * g / (1 + np.exp(-g.astype(np.float64)))).astype(np.float32)
                diagnostics.append(dict(input=i, h_from_device_up_gate=metric(h, local_h, .9999999)))
            # Localize errors without replacing the end-to-end acceptance above.
            local = reference(down, h[None, :], meta['n'], meta['k'])[0, 0]
            rounded, rounded_ref = h.astype(bfloat16), href.astype(bfloat16)
            different = np.flatnonzero(rounded != rounded_ref)
            diagnostics.append(dict(input=i, down_from_device_h=metric(got[0], local, .9999999),
                                    bf16_h_differences=len(different),
                                    first_differences=[dict(index=int(j), got=float(h[j]), ref=float(href[j]),
                                                            got_bf16=float(rounded[j]), ref_bf16=float(rounded_ref[j]))
                                                       for j in different[:8]]))
    if not meta['full'] or meta.get('repeated_last'):
        first, last = [(out / f'got{i}.bin').read_bytes() for i in (0, meta['inputs'] - 1)]
        start, size = meta['out_offset'], meta['snapshots'] * meta['n'] * 4
        if first[start:start + size] != last[start:start + size]:
            raise ValueError('repeated input is not deterministic')
    ok = all(r['passed'] for r in results)
    result = dict(passed=ok, checks=results, diagnostics=diagnostics)
    (out / 'segmented-results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'compare'))
    p.add_argument('--build-dir', type=Path, required=True)
    args = p.parse_args()
    return prepare(args.build_dir) if args.stage == 'prepare' else compare(args.build_dir)


if __name__ == '__main__':
    sys.exit(main())
