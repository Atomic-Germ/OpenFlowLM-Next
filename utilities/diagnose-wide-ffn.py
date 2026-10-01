#!/usr/bin/env python3
"""Offline FFN diagnosis from immutable full-model weights and device captures.

Conditional references localize errors only; the full-model oracle is untouched.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'open_kernels'))
from wide_slice_reference import projection, silu, H, F
from wide_attention_reference import metric
from recipes.spec import LINEAR
from recipes.spec import ModelSpec
from recipes import qwen35

GUARD = bytes([0xA5])*64


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(path, size, dtype, *, logical=None, offset=0):
    raw = Path(path).read_bytes()
    if len(raw) != size+64 or raw[-64:] != GUARD:
        raise ValueError(f'{path}: size/canary')
    data = np.frombuffer(raw[:size], dtype, offset=offset)
    if logical is not None: data = data[:logical]
    if not np.isfinite(data.astype(np.float32)).all():
        raise ValueError(f'{path}: nonfinite logical data')
    return data


def measure(got, ref):
    result = metric(got, ref, .005)
    result['maxabs'] = float(np.max(np.abs(got.astype(np.float64)-ref)))
    result['different'] = int(np.count_nonzero(got != ref))
    return result


def analyze(project, xm, h, fo):
    u, g = project('up', xm), project('gate', xm)
    expected_h = (u.astype(np.float64)*silu(g)).astype(np.float32)
    expected_fo = project('down', expected_h)
    local_fo = project('down', h)
    rounded, expected = h.astype(bfloat16), expected_h.astype(bfloat16)
    ids = np.flatnonzero(rounded != expected)
    result = dict(diagnostic_only=True, h=measure(h, expected_h),
                  ffn=measure(fo, expected_fo), conditional_down=measure(fo, local_fo),
                  h_bf16_differences=len(ids),
                  first_h_differences=[dict(index=int(i), got=float(h[i]), ref=float(expected_h[i]),
                                           got_bf16=float(rounded[i]), ref_bf16=float(expected[i]))
                                       for i in ids[:32]])
    return result, dict(up=u, gate=g, h=expected_h, fo=expected_fo, down_from_device_h=local_fo)


def trace_arrays(raw, width):
    if raw.size != 2*width or width % 64:
        raise ValueError('trace size must contain up/gate pairs for each 64-row band')
    bands = raw.reshape(-1, 2, 64)
    return bands[:,0,:].ravel(), bands[:,1,:].ravel()


def pad_activation(raw, size):
    if len(raw)<64 or raw[-64:]!=GUARD or size<len(raw)-64 or size%4:
        raise ValueError('incompatible activation size/canary')
    return raw[:-64]+np.full((size-len(raw)+64)//4,np.nan,np.float32).tobytes()+GUARD


def prepare_trace(source, tag, kernel, out):
    """Replay one real FFN frame, without altering the model's fixtures."""
    source, kernel = source.resolve(), kernel.resolve()
    spec = ModelSpec.from_dict(json.loads((kernel/'probe-spec.json').read_text()))
    fixture = json.loads((kernel/'segmented-fixture.json').read_text())
    if (spec.hidden, spec.intermediate, fixture['full'], fixture['trace']) != (H,F,True,True):
        raise ValueError('requires validated H5120/FF17408 full FFN trace kernel')
    if not json.loads((kernel/'segmented-results.json').read_text())['passed']:
        raise ValueError('trace primitive gate failed')
    for name, digest in fixture['sha256'].items():
        if sha(kernel/name) != digest: raise ValueError(f'kernel fixture changed: {name}')
    diagnose(source, tag)
    meta = json.loads((source/'slice-fixture.json').read_text())
    case = next(c for c in meta['cases'] if c['tag']==tag)
    size = meta['outputs']['d' if case['kind']==LINEAR else 'a']['act']
    original = (source/f'{tag}-got-act.bin').read_bytes()
    if len(original)!=size+64: raise ValueError('source activation size')
    size = fixture['act_bytes']
    padded = pad_activation(original,size)
    out.mkdir(parents=True, exist_ok=False)
    files = {'pool.bin': source/f"layer{case['layer']}/pool.bin",
             'reference.npz': source/f'{tag}-ffn-diagnostic-ref.npz',
             'source-fo.bin': source/f'{tag}-got-fo.bin',
             'final.xclbin': kernel/'final.xclbin', 'insts.bin': kernel/'insts.bin'}
    for name, target in files.items():
        if name=='reference.npz': (out/name).write_bytes(target.read_bytes())
        else: (out/name).symlink_to(target)
    (out/'act.bin').write_bytes(padded)
    layout = qwen35.layout(spec)
    (out/'trace-poison.bin').write_bytes(np.full(F*2,np.nan,np.float32).tobytes()+GUARD)
    lines = ['device','xclbin p final.xclbin','kernelx p p insts.bin',
             f'buf w {files["pool.bin"].stat().st_size} pool.bin',
             f'buf a {size+64}',f'buf t {F*8+64}']
    for i in range(2):
        lines += ['load a act.bin','load t trace-poison.bin','run p w a t',
                  f'dump a got{i}.bin {size+64}',f'dump t trace{i}.bin {F*8+64}']
    (out/'trace.cfg').write_text('\n'.join(lines)+'\n')
    record = dict(diagnostic_only=True, tag=tag, act_bytes=size, h_offset=layout.A_H,
                  out_offset=layout.A_OUT2, source=str(source),
                  sha256={name:sha(out/name) for name in [*files,'act.bin','trace.cfg','trace-poison.bin']})
    (out/'trace-fixture.json').write_text(json.dumps(record,indent=2)+'\n')
    print(out/'trace.cfg', flush=True)


def compare_trace(out):
    meta = json.loads((out/'trace-fixture.json').read_text())
    for name, digest in meta['sha256'].items():
        if sha(out/name)!=digest: raise ValueError(f'trace fixture changed: {name}')
    ref = np.load(out/'reference.npz')
    u,g = trace_arrays(capture(out/'trace0.bin',F*8,np.float32),F)
    raw = (out/'got0.bin').read_bytes()
    if len(raw)!=meta['act_bytes']+64 or raw[-64:]!=GUARD: raise ValueError('act size/canary')
    h = capture(out/'got0.bin',meta['act_bytes'],np.float32,logical=F,offset=meta['h_offset'])
    fo = capture(out/'got0.bin',meta['act_bytes'],np.float32,logical=H,offset=meta['out_offset'])
    source = (out/'act.bin').read_bytes()
    original_h = np.frombuffer(source,np.float32,count=F,offset=meta['h_offset'])
    original_fo = capture(out/'source-fo.bin',H*4,np.float32)
    local_h = (u.astype(np.float64)*silu(g)).astype(np.float32)
    ids = np.flatnonzero(h.astype(bfloat16)!=ref['h'].astype(bfloat16))
    result = dict(diagnostic_only=True, tag=meta['tag'], up=measure(u,ref['up']),
                  h_bf16_differences=len(ids),
                  gate=measure(g,ref['gate']), activation_from_device_up_gate=measure(h,local_h),
                  matches_recorded_h=bool(np.array_equal(h,original_h)),
                  matches_recorded_fo=bool(np.array_equal(fo,original_fo)),
                  repeat_exact=all((out/f'{n}0.bin').read_bytes()==(out/f'{n}1.bin').read_bytes() for n in ('got','trace')),
                  h_bf16_from_projection=int(np.count_nonzero(local_h.astype(bfloat16)!=ref['h'].astype(bfloat16))),
                  h_bf16_from_activation=int(np.count_nonzero(h.astype(bfloat16)!=local_h.astype(bfloat16))),
                  differences=[dict(index=int(i),up=float(u[i]),up_ref=float(ref['up'][i]),
                                    gate=float(g[i]),gate_ref=float(ref['gate'][i]),h=float(h[i]),
                                    h_ref=float(ref['h'][i]),h_local=float(local_h[i])) for i in ids])
    (out/'trace-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


def diagnose(out, tag):
    meta = json.loads((out/'slice-fixture.json').read_text())
    case = next(c for c in meta['cases'] if c['tag'] == tag)
    spec = importlib.util.spec_from_file_location('full_model', ROOT/'utilities/test-wide-full-model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, layout = module.S.A.geometry(meta['rows'])
    names = [f"layer{case['layer']}/{f}" for f in ('pool.bin','const.bin')]+[f'{tag}-ref.npz']
    for name in names:
        if sha(out/name) != meta['fixtures'][name]: raise ValueError(f'fixture changed: {name}')
    for name, digest in meta['kernels'].items():
        if sha(name) != digest: raise ValueError(f'kernel changed: {name}')
    pool = np.memmap(out/names[0], mode='r', dtype=np.uint8)
    const = np.fromfile(out/names[1], np.uint8)
    sizes = meta['outputs']['d' if case['kind']==LINEAR else 'a']
    xm = capture(out/f'{tag}-got-xm.bin', sizes['xm'], bfloat16, logical=H)
    fo = capture(out/f'{tag}-got-fo.bin', sizes['fo'], np.float32)
    # The activation arena contains intentionally uninitialized padding.
    act_path = out/f'{tag}-got-act.bin'
    raw = act_path.read_bytes()
    if len(raw)!=sizes['act']+64 or raw[-64:]!=GUARD: raise ValueError('act size/canary')
    h = np.frombuffer(raw, np.float32, count=F, offset=layout.A_H)
    if not np.isfinite(h).all(): raise ValueError('nonfinite h')
    project = lambda name, x: projection(pool, const, layout, case['kind'], name, x)
    result, refs = analyze(project, xm, h, fo)
    ref = np.load(out/f'{tag}-ref.npz')
    result.update(tag=tag, xm_bf16_differences=int(np.count_nonzero(xm!=ref['xm'].astype(bfloat16))),
                  device_fo_vs_independent=measure(fo, ref['fo']),
                  capture_sha256={p.name:sha(p) for p in (act_path,out/f'{tag}-got-xm.bin',out/f'{tag}-got-fo.bin')})
    np.savez(out/f'{tag}-ffn-diagnostic-ref.npz', **refs)
    (out/f'{tag}-ffn-diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tag', default='cold-0-layer1')
    p.add_argument('--trace-build', type=Path)
    p.add_argument('--trace-out', type=Path)
    p.add_argument('--compare-trace', action='store_true')
    p.add_argument('--require-exact-h-bf16', action='store_true')
    a = p.parse_args()
    if a.require_exact_h_bf16 and not a.compare_trace:
        p.error('--require-exact-h-bf16 requires --compare-trace')
    if a.compare_trace and a.trace_build:
        p.error('--compare-trace cannot prepare a trace')
    if a.compare_trace:
        result = compare_trace(a.out)
        if not result['repeat_exact'] or (a.require_exact_h_bf16 and result['h_bf16_differences']):
            sys.exit(1)
    elif a.trace_build:
        if a.trace_out is None: p.error('--trace-build requires --trace-out')
        prepare_trace(a.out,a.tag,a.trace_build,a.trace_out)
    else:
        diagnose(a.out, a.tag)
