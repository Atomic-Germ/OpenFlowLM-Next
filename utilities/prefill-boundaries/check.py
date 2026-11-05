"""Conditioned host prefill gates; no weights, NPU projections or model acceptance."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'open_kernels/model'))
sys.path.insert(0, str(ROOT / 'open_kernels'))
import replica_block as rb
from recipes.load import load_spec
from recipes.qwen35 import layout, recipe

SPEC = ROOT / 'open_kernels/recipes/specs/qwen35-27b.json'
FIELDS = ('delta', 'attn', 'conv', 'state', 'kv')


def schedule(prompt):
    if not isinstance(prompt, int) or prompt < 1:
        raise ValueError('positive prompt length required')
    return [256] * (prompt // 256) + ([prompt % 256] if prompt % 256 else []) + [1]


def metric(got, ref):
    got, ref = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape or not got.size or not np.isfinite(got).all() or not np.isfinite(ref).all():
        raise ValueError('complete finite arrays required')
    error = float(np.max(np.abs(got-ref)) / max(float(np.max(np.abs(ref))), 1e-12))
    return dict(maxrel=error, limit=1e-3, passed=error < 1e-3)


def kv_values(bits):
    planes = (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
    kh, kl, vh, vl = np.split(planes, 4, axis=1)
    return np.concatenate((kh+kl, vh+vl), axis=1)


def score_result(got, sequential, reference):
    identity = {k: got[k].shape == sequential[k].shape and got[k].dtype == sequential[k].dtype
                and got[k].tobytes() == sequential[k].tobytes() for k in FIELDS}
    numerical = {k: metric(got[k], reference[k]) for k in ('delta', 'attn', 'state')}
    numerical['kv'] = metric(kv_values(got['kv']), kv_values(reference['kv']))
    history = np.array_equal(got['conv'], reference['conv'])
    padding = bool(got['padding_ok'] and sequential['padding_ok'] and reference['padding_ok'])
    return dict(passed=all(identity.values()) and all(x['passed'] for x in numerical.values())
                and history and padding, identical=all(identity.values()), identity=identity,
                numerical=numerical, history_exact=history, padding_ok=padding)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(out):
    """Always rebuild; record compiler-emitted local dependencies, not a hand-picked list."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    flags = ['g++', '-std=c++17', '-O2', '-fPIC', '-mavx2', '-mfma', '-ffp-contract=off',
             '-fopenmp', '-I'+str(ROOT/'src'), '-I'+str(ROOT/'src/include')]
    commands, deps, objects = [], set(), []
    for src in (HERE/'bridge.cpp', ROOT/'src/open_qwen36/block_host.cpp'):
        obj = out/(src.stem+'.o')
        dep = out/(src.stem+'.d')
        cmd = flags + ['-MMD', '-MF', str(dep), '-c', str(src), '-o', str(obj)]
        subprocess.run(cmd, check=True)
        commands.append(cmd)
        objects.append(str(obj))
        deps.update(dep.read_text().replace('\\\n', '').split(':', 1)[1].split())
    lib = out/'libprefill_boundaries.so'
    cmd = ['g++', '-shared', '-fopenmp', *objects, '-o', str(lib)]
    subprocess.run(cmd, check=True)
    commands.append(cmd)
    record = dict(commands=commands, compiler=subprocess.check_output(['g++', '--version'], text=True),
                  sources={str(Path(p).relative_to(ROOT)): digest(p) for p in sorted(deps)},
                  binary_sha256=digest(lib))
    (out/'build.json').write_text(json.dumps(record, indent=2)+'\n')
    return lib


def native_library(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    ptr = lambda dtype: np.ctypeslib.ndpointer(dtype=dtype, flags='C_CONTIGUOUS')
    lib.prefill_delta.argtypes = [ptr(np.uintp)] + [ptr(np.float32)]*12
    lib.prefill_attention.argtypes = [ptr(np.uintp)] + [ptr(np.float32)]*6 + [ptr(np.float64), ptr(np.uint16), ptr(np.float32)]
    lib.prefill_delta.restype = lib.prefill_attention.restype = ctypes.c_int
    return lib


def geometry(small=False):
    if small:
        return dict(hid=32, kh=2, vh=6, dim=8, taps=4, lanes=8, rows=10,
                    nh=6, kvh=2, hd=16, rot=8, theta=1e7)
    s = load_spec(SPEC)
    if s.norm_eps != 1e-6:
        raise ValueError('bridge epsilon differs from spec')
    return dict(hid=s.hidden, kh=s.lin_key_heads, vh=s.lin_value_heads, dim=s.lin_key_dim,
                taps=s.conv_kernel, lanes=recipe(s).linear.AB_LANES, rows=layout(s).S_ROWS,
                nh=s.num_heads, kvh=s.num_kv_heads, hd=s.head_dim, rot=s.rotary_dim, theta=s.rope_theta)


def bits(x):
    return (rb.bf16_round(x).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def make_case(g, tokens, seed=38, prefix=3):
    rng = np.random.default_rng(seed)
    def f(shape, scale=1):
        return (rng.standard_normal(shape)*scale).astype(np.float32)
    vw, nch = g['vh']*g['dim'], (2*g['kh']+g['vh'])*g['dim']
    qw, kw = g['nh']*g['hd'], g['kvh']*g['hd']
    c = dict(qkv=f((tokens,nch)), z=f((tokens,vw)), xn=f((tokens,g['hid'])),
             convw=f((g['taps'],nch), .3), Wa=f((g['hid'],g['lanes']), .1/np.sqrt(g['hid'])),
             Wb=f((g['hid'],g['lanes']), .1/np.sqrt(g['hid'])),
             A=-np.exp(f((g['vh'],), .3)), dtb=f((g['vh'],), .1), nw=1+f((g['dim'],), .1),
             conv=f((g['taps']-1,nch), .3), state=np.zeros((g['vh'],g['rows'],g['dim']),np.float32),
             q=f((tokens,qw)), k=f((tokens,kw)), v=f((tokens,kw)), gate=f((tokens,qw)),
             qn=1+f((g['hd'],), .1), kn=1+f((g['hd'],), .1),
             freq=g['theta']**(-np.arange(g['rot']//2)/(g['rot']//2)), prefix=prefix)
    c['state'][:,:g['dim']] = f((g['vh'],g['dim'],g['dim']), .02)
    # Distinct nonzero low planes and future rows expose accidental cache reads/writes.
    c['kv'] = bits(np.full((prefix+tokens+5,4*kw), .125, np.float32))
    for off in (0,2*kw):
        value = f((prefix,kw))
        hi = rb.bf16_round(value)
        c['kv'][:prefix,off:off+kw] = bits(hi)
        c['kv'][:prefix,off+kw:off+2*kw] = bits(value-hi)
    return c


def untouched(cache, initial, prefix, end):
    return np.array_equal(cache[:prefix], initial[:prefix]) and np.array_equal(cache[end:], initial[end:])


def run_reference(c, g, tokens):
    # Promote all operands before arithmetic: independent FP64, never fed native state.
    a = {k: v.astype(np.float64) for k,v in c.items() if isinstance(v,np.ndarray)}
    a['kv'] = (c['kv'].astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    d = rb.deltanet_block(*(a[k][:tokens] for k in ('qkv','z','xn')),
        *(a[k] for k in ('convw','Wa','Wb','A','dtb','nw','conv','state')),
        key_heads=g['kh'], value_heads=g['vh'], head_dim=g['dim'], conv_f32=True)
    att = rb.attention_block(*(a[k][:tokens] for k in ('q','k','v','gate')),
        *(a[k] for k in ('qn','kn','freq','kv')), c['prefix'],
        nh=g['nh'], kvh=g['kvh'], hd=g['hd'], kv_pair=True)
    kv = bits(a['kv'])
    return dict(delta=d, attn=att, conv=a['conv'], state=a['state'], kv=kv,
                padding_ok=untouched(kv,c['kv'],c['prefix'],c['prefix']+tokens))


def run_native(lib, c, g, chunks, padded=True):
    if not chunks or min(chunks)<1 or max(chunks)>256 or sum(chunks)>len(c['qkv']):
        raise ValueError('invalid native chunks')
    conv, state, kv = (c[k].copy() for k in ('conv','state','kv'))
    ds, ats, start, valid = [], [], 0, True
    for real in chunks:
        T = 256 if padded else real
        def inp(k):
            x = np.full((T,c[k].shape[1]), np.nan, np.float32)
            x[:real] = c[k][start:start+real]
            return x
        d = np.full((T,g['vh']*g['dim']), np.nan, np.float32)
        at = np.full((T,g['nh']*g['hd']), np.nan, np.float32)
        dg = np.array([T,real,*(g[k] for k in ('hid','kh','vh','dim','taps','lanes','rows'))],np.uintp)
        ag = np.array([T,real,*(g[k] for k in ('nh','kvh','hd','rot')),c['prefix']+start],np.uintp)
        if lib.prefill_delta(dg, *(inp(k) for k in ('qkv','z','xn')),
                *(c[k] for k in ('convw','Wa','Wb','A','dtb','nw')), conv,state,d):
            raise RuntimeError('native DeltaNet failed')
        if lib.prefill_attention(ag, *(inp(k) for k in ('q','k','v','gate')),
                c['qn'],c['kn'],c['freq'],kv,at):
            raise RuntimeError('native attention failed')
        start += real
        valid &= bool(np.all(d[real:]==0) and np.all(at[real:]==0)
                      and np.all(state[:,g['dim']:]==0)
                      and untouched(kv,c['kv'],c['prefix'],c['prefix']+start))
        ds.append(d[:real].copy()); ats.append(at[:real].copy())
    return dict(delta=np.concatenate(ds),attn=np.concatenate(ats),conv=conv,state=state,kv=kv,padding_ok=valid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--small', action='store_true', help='small synthetic geometry for development')
    ap.add_argument('--seed', type=int, default=38)
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    report_path = out/'report.json'
    report_path.unlink(missing_ok=True)  # a failed rerun must never leave a stale success
    g = geometry(args.small)
    report = dict(passed=False, scope='synthetic conditioned CPU host stages only',
                  geometry=g, seed=args.seed, spec_sha256=digest(SPEC), cases=[])
    try:
        lib = native_library(build(out))
        for prompt in (1,2,255,256,257):
            c = make_case(g,prompt+1,args.seed)
            np.savez(out/f'inputs-{prompt}.npz', **c)
            reference = run_reference(c,g,prompt+1)
            sequential = run_native(lib,c,g,[1]*(prompt+1),padded=False)
            blocked = run_native(lib,c,g,schedule(prompt))
            score = score_result(blocked,sequential,reference)
            for name,result in (('reference',reference),('sequential',sequential),('blocked',blocked)):
                np.savez(out/f'{name}-{prompt}.npz',**result)
            report['cases'].append(dict(prompt=prompt,continuation=1,chunks=schedule(prompt),**score))
            print(f"prompt={prompt} passed={score['passed']} identity={score['identical']} "
                  f"errors={ {k:v['maxrel'] for k,v in score['numerical'].items()} }",flush=True)
        report['passed'] = all(c['passed'] for c in report['cases'])
    except Exception as e:
        report['error'] = str(e)
        raise
    finally:
        report['artifacts'] = {p.name:digest(p) for p in sorted(out.glob('*.npz'))}
        report['sources'] = {str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),HERE/'bridge.cpp',ROOT/'open_kernels/model/replica_block.py',SPEC)}
        if (out/'build.json').exists(): report['build_sha256'] = digest(out/'build.json')
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    # Set before loading libgomp; callers should also pin BLAS before Python starts.
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
    raise SystemExit(main())
