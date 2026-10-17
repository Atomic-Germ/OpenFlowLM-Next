"""Causal private full-lx replay: identity, DeltaNet, projections, FFN and residuals.

Incoming residuals are forced from production. Each variant carries its own
state from zero. This experiment is not full-model propagation or acceptance.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np
from consistent_lx import ROOT, HERE, digest, geometry, verify_build
from full_layer_replay import config, exact_capture, checked_files
from recurrence_replay import read_f32
from decode_bundle import load_bundle
from decode_reference import load_reference, sfx
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout, ffn_geometry
from recipes.spec import LINEAR
from q4nx import Q4NX, bf16_to_f32
import deltanet_state as ds
import deltanet_inputs as di
import ffn_front as front
import ffn_partials as down
import replica_qwen35 as replica


def separate_output(out, inputs):
    out = out.resolve()
    if any(out.is_relative_to(p.resolve()) or p.resolve().is_relative_to(out) for p in inputs):
        raise ValueError('separate private replay output required')


def verify_original_inputs(current, original):
    original = {str((ROOT/Path(p)).resolve()): h for p, h in original.items()}
    if not current or any(original.get(p) != h for p, h in current.items()):
        raise ValueError('model/pool differs from original production provenance')


def accepted(mode, row):
    if mode == 'control': return row['control_identical']
    if mode != 'consistent': raise ValueError('unknown replay mode')
    return (row['entry_identical'] and row['isolated_qkv_identical']
            and row['delta']['passed'] and row['delta']['recurrence']['head_diagnostics_passed']
            and all(row[k]['passed'] for k in ('residual', 'front', 'down', 'projections')))


def context(model, kernels, fixture, build):
    spec = spec_from_model_dir(model); meta = load_reference(fixture)
    bundle = load_bundle(kernels, manifest(spec, max_ctx=meta['max_ctx']))
    if (any(meta[k] != bundle[k] for k in ('build_key', 'spec_hash'))
            or meta['layers'] != spec.num_layers or meta['tokens'] < 2):
        raise ValueError('invalid full-depth production fixture')
    b = verify_build(build, spec, bundle, kernels)
    if ds.linear_dtypes(bundle) != ('float32', 'fp32'): raise ValueError('requires FP32 entry/history')
    return spec, meta, bundle, b, geometry(spec)


def prepare(model, kernels, fixture, pools, build, isolated, original, out):
    separate_output(out, [model, kernels, fixture, pools, build, isolated, original])
    spec, meta, bundle, b, g = context(model, kernels, fixture, build)
    layers = [l for l in range(meta['layers']) if spec.layer_types[l] == LINEAR]
    if not layers: raise ValueError('no linear layers')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('audit.json', 'score.json', 'progress.json'): (out/name).unlink(missing_ok=True)
    for l in layers:
        for t in range(meta['tokens']):
            for k in ('act', 'state', 'res'): (out/f'y_{k}{l}{sfx(t)}.bin').unlink(missing_ok=True)
    (out/'zero.bin').write_bytes(bytes(g['state_bytes']))
    (out/'zero_act.bin').write_bytes(bytes(g['act_bytes']))
    (out/'run.cfg').write_text(config(bundle, g, layers, meta['tokens'], fixture, pools, out, build))
    files = [build/n for n in ('build.json', 'final.xclbin', 'insts.bin')]
    files += [model/'config.json', fixture/'decode_reference.json', original, isolated/'results.json']
    files += [out/n for n in ('run.cfg', 'zero.bin', 'zero_act.bin')]
    files += list(HERE.glob('*.py'))
    files += [ROOT/'utilities/deltanet-replay'/n for n in ('full_layer_replay.py', 'full_layer_common.py', 'recurrence_common.py', 'recurrence_replay.py')]
    files += [ROOT/'utilities/qkv-precision/qkv_build.py']
    files += [ROOT/'open_kernels/model'/n for n in ('deltanet_inputs.py', 'deltanet_state.py', 'q4nx.py', 'ffn_front.py', 'ffn_partials.py', 'replica_qwen35.py')]
    original_files = [model/'model.q4nx'] + [pools/f'pool_L{l}.bin' for l in layers]
    original_hashes = {str(p.resolve()): digest(p) for p in original_files}
    verify_original_inputs(original_hashes, json.loads(original.read_text()))
    for l in layers:
        files += [fixture/f'consts_{l}.bin', isolated/f'L{l}/consistent/score.json']
        for t in range(meta['tokens']):
            inp = fixture/(f'y_res{l-1}{sfx(t)}.bin' if l else f'xres{t}.bin')
            read_f32(inp, (spec.hidden,))
            files += [inp, isolated/f'L{l}/consistent/qkv{t}.bin']
            files += [fixture/f'y_{k}{l}{sfx(t)}.bin' for k in ('act', 'state', 'res')]
    audit = dict(mode=b['mode'], layers=layers, tokens=meta['tokens'], build_key=bundle['build_key'],
                 **{n:str(p.resolve()) for n,p in dict(model=model,kernels=kernels,fixture=fixture,pools=pools,build=build,isolated=isolated,original=original).items()},
                 sha256={**original_hashes, **{str(p.resolve()):digest(p) for p in files}})
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('Prepared', len(layers)*meta['tokens'], 'causal full-layer steps', flush=True)


def verified(out):
    a = json.loads((out/'audit.json').read_text()); checked_files(a['sha256'])
    spec, meta, bundle, b, g = context(*(Path(a[n]) for n in ('model', 'kernels', 'fixture', 'build')))
    layers = [l for l in range(meta['layers']) if spec.layer_types[l] == LINEAR]
    if a['layers'] != layers or a['tokens'] != meta['tokens'] or a['mode'] != b['mode'] or a['build_key'] != bundle['build_key']:
        raise ValueError('changed replay coverage')
    expected = config(bundle, g, layers, meta['tokens'], Path(a['fixture']), Path(a['pools']), out, Path(a['build']))
    if (out/'run.cfg').read_text() != expected: raise ValueError('changed replay program')
    original_files = [Path(a['model'])/'model.q4nx'] + [Path(a['pools'])/f'pool_L{l}.bin' for l in layers]
    verify_original_inputs({str(p):a['sha256'][str(p)] for p in original_files}, json.loads(Path(a['original']).read_text()))
    return a, spec, meta, bundle, b, g


class CachedLayer:
    def __init__(self, model): self.model, self.cache = model, {}
    def matmul_w(self, *args):
        if args not in self.cache: self.cache[args] = self.model.matmul_w(*args)
        return self.cache[args]
    def bf16(self, *args): return self.model.bf16(*args)
    def f32(self, *args): return self.model.f32(*args)


def projections(data, offsets, spec, weights, layer, x, capture):
    """FP64 Z/out dots plus post-recurrence norm/gate, conditioned on NPU inputs."""
    pre = f'model.layers.{layer}.'; vw = spec.lin_value_width
    def f32(name, n):
        return ds.finite(np.frombuffer(data, '<f4', count=n, offset=getattr(offsets, name)))
    z = f32('A_Z', vw); out = f32('A_OUT', spec.hidden); res = f32('A_RES', spec.hidden)
    og = ds.finite(bf16_to_f32(np.frombuffer(data, '<u2', count=vw, offset=offsets.A_OG)))
    wz = weights.matmul_w(pre+'self_attn.gate_proj.weight', vw, spec.hidden).astype(np.float64)
    wo = weights.matmul_w(pre+'linear_attn.ssm_out_proj.weight', spec.hidden, vw).astype(np.float64)
    nw = weights.bf16(pre+'linear_attn.ssm_norm.weight').astype(np.float64)
    norm = weights.bf16(pre+'input_layernorm.weight').astype(np.float64)
    exact = x / np.sqrt(np.mean(x*x)+spec.norm_eps) * norm
    post = replica.rms(capture['output'], spec.norm_eps) * nw
    metrics = dict(z=front.metric(z, wz@exact), z_conditioned=front.metric(z, wz@capture['xn']),
                   post=front.metric(og, post.reshape(vw)*replica.silu(z)),
                   out=front.metric(out, wo@og), residual_add=front.metric(res, x+out))
    return dict(passed=all(v['passed'] for v in metrics.values()), metrics=metrics)


def score(out):
    (out/'score.json').unlink(missing_ok=True)
    a, spec, meta, bundle, b, g = verified(out)
    fixture = Path(a['fixture']); isolated = Path(a['isolated']); off = layout(spec, max_ctx=meta['max_ctx'])
    isolated_result = json.loads((isolated/'results.json').read_text())
    if not isolated_result['passed'] or isolated_result['layers'] != a['layers']:
        raise ValueError('unvalidated isolated sweep')
    model = Q4NX(Path(a['model'])/'model.q4nx')
    patterns = [re.compile(re.escape(op['tensor']).replace(r'\{l\}', r'\d+'))
                for d in bundle['layer_types'].values() for op in d['pack']['pool']+d['pack']['consts'] if op.get('op') == 'q8_perm']
    model.native_q8 = lambda n: any(p.fullmatch(n) for p in patterns)
    rows, hashes, ref_hashes = [], {}, {}
    for l in a['layers']:
        inputs, captures, base = [], [], []
        single = json.loads((isolated/f'L{l}/consistent/score.json').read_text())
        if not single['passed'] or single['mode'] != 'consistent' or single['layer'] != l or single['tokens'] != a['tokens']:
            raise ValueError('incomplete isolated QKV evidence')
        for t in range(a['tokens']):
            inputs.append(read_f32(fixture/(f'y_res{l-1}{sfx(t)}.bin' if l else f'xres{t}.bin'), (spec.hidden,)).astype(np.float64))
            for directory, seq in ((out,captures), (fixture,base)):
                act = (directory/f'y_act{l}{sfx(t)}.bin').read_bytes()
                c = ds.read_boundaries(act, (directory/f'y_state{l}{sfx(t)}.bin').read_bytes(), spec, off, conv_dtype='fp32')
                c['xn'] = di.read_entry(act, off, spec.hidden, norm_dtype='float32'); seq.append(c)
        dw, _ = di.load_weights(model, spec, l)
        delta = di.score_sequence(spec, dw, inputs, captures, conv_dtype='fp32'); del dw
        weights = CachedLayer(model)
        conv = np.zeros((spec.conv_kernel-1, spec.lin_qkv_dim)); state = np.zeros((spec.lin_value_heads,128,128))
        pre = f'model.layers.{l}.'
        for t, (x,c,orig,d) in enumerate(zip(inputs,captures,base,delta)):
            ref, conv, state, _, _ = replica.layer_decode(weights, spec, l, x, conv, state, None, None, t)
            ref_path = out/f'ref_res{l}{sfx(t)}.bin'; np.asarray(ref, '<f8').tofile(ref_path); ref_hashes[ref_path.name] = digest(ref_path)
            act = out/f'y_act{l}{sfx(t)}.bin'; residual = out/f'y_res{l}{sfx(t)}.bin'
            f = front.read_front(act, off, spec.hidden, spec.intermediate, False, norm_dtype='float32')
            fscore = front.compare_front(f['res'], weights.bf16(pre+'post_attention_layernorm.weight'),
                weights.matmul_w(pre+'mlp.up_proj.weight', spec.intermediate, spec.hidden),
                weights.matmul_w(pre+'mlp.gate_proj.weight', spec.intermediate, spec.hidden), f['xm'], f['h'], spec.norm_eps)
            r = down.read_regions(act, off, spec.hidden, spec.intermediate, False)
            closed = read_f32(residual, (spec.hidden,)).astype(np.float64)
            dscore = down.compare_partials(r['h'], weights.matmul_w(pre+'mlp.down_proj.weight', spec.hidden, spec.intermediate),
                r['res'], r['partials'], closed, tuple(ffn_geometry(spec).DOWN_SPLIT))
            exact = True
            for kind, size in [('act',g['act_bytes']),('state',g['state_bytes']),('res',spec.hidden*4)]:
                p = out/f'y_{kind}{l}{sfx(t)}.bin'; hashes[p.name] = digest(p)
                exact &= exact_capture(p, fixture/p.name, size)
            iq = isolated/f'L{l}/consistent/qkv{t}.bin'
            if digest(iq) != single['capture_sha256'][iq.name]: raise ValueError('changed isolated QKV capture')
            row = dict(layer=l, token=t, control_identical=exact,
                entry_identical=c['xn'].tobytes() == orig['xn'].tobytes(),
                isolated_qkv_identical=np.asarray(c['qkv'], '<f4').tobytes() == iq.read_bytes(),
                delta=d, residual=front.metric(closed, ref), front=fscore, down=dscore,
                projections=projections(act.read_bytes(), off, spec, weights, l, x, c))
            row['passed'] = bool(accepted(b['mode'], row)); rows.append(row)
        del weights
        (out/'progress.json').write_text(json.dumps(dict(completed_layers=len(rows)//a['tokens'], cases=len(rows), passed=sum(r['passed'] for r in rows)))+'\n')
        print('Layer', l, 'PASS' if all(r['passed'] for r in rows[-a['tokens']:]) else 'FAIL', flush=True)
    checked_files(a['sha256'])
    result = dict(mode=b['mode'], passed=all(r['passed'] for r in rows), layers=a['layers'], tokens=a['tokens'],
        production_build_key=a['build_key'], scope=__doc__.strip(),
        residual_oracle='existing replica_qwen35, mixed FP64/FP32, independent causal state from zero',
        evidence_sha256={str(out/'audit.json'):digest(out/'audit.json')}, capture_sha256=hashes,
        reference_sha256=ref_hashes, results=rows)
    (out/'score.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('PASS' if result['passed'] else 'FAIL', len(rows), 'full-layer cases', flush=True)
    return 0 if result['passed'] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('stage', choices=['prepare','score'])
    for n in ('model','kernels','fixture','pools','build','isolated','original','out'): ap.add_argument('--'+n, type=Path)
    a = ap.parse_args(); required = ['out'] if a.stage == 'score' else ['model','kernels','fixture','pools','build','isolated','original','out']
    for n in required:
        if getattr(a,n) is None: ap.error('missing --'+n)
        setattr(a,n,getattr(a,n).resolve())
    if a.stage == 'prepare': prepare(*(getattr(a,n) for n in required)); return 0
    return score(a.out)


if __name__ == '__main__': raise SystemExit(main())
