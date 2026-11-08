"""Localize L34 QKV table/epilogue error with independent causal record checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from qkv_reference import ROOT, activation_table, project, unpack_chunks

sys.path[:0] = [str(ROOT/'open_kernels'), str(ROOT/'utilities')]
import deltanet_projection as dp
from q4nx import q4_1_chunks_of
from ffn_front import metric


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(model_dir, kernel_dir, fixture, layer):
    spec = dp.spec_from_model_dir(model_dir)
    meta = dp.load_reference(fixture)
    bundle = dp.load_bundle(kernel_dir, dp.manifest(spec, max_ctx=meta['max_ctx']))
    if any(meta[k] != bundle[k] for k in ('build_key', 'spec_hash')):
        raise ValueError('fixture/export identity mismatch')
    if (spec.hidden != 5120 or spec.lin_qkv_dim != 10240 or spec.quant_of('linear') != 'q4_1'
            or not 0 <= layer < meta['layers'] <= spec.num_layers
            or spec.layer_types[layer] != dp.LINEAR or meta['tokens'] < 2):
        raise ValueError('requires covered multi-token H5120 Q4 linear layer')
    offsets = dp.layout(spec, max_ctx=meta['max_ctx'])
    norm_dtype, conv_dtype = dp.linear_dtypes(bundle)
    if norm_dtype != 'float32' or conv_dtype != 'fp32':
        raise ValueError('requires FP32 entry/history')
    model = dp.Q4NX(model_dir/'model.q4nx')
    weights, weight_hashes = dp.di.load_weights(model, spec, layer)
    chunks = q4_1_chunks_of(model, f'model.layers.{layer}.linear_attn.qkv_proj.weight')
    hashes = {}

    def read(name):
        data = (fixture/name).read_bytes(); hashes[name] = hashlib.sha256(data).hexdigest()
        return data

    zero = read(f'zstate_{dp.LINEAR}.bin')
    if len(zero) != offsets.STATE_BYTES or any(zero):
        raise ValueError('nonzero or incomplete initial state')
    inputs, captures = [], []
    for t in range(meta['tokens']):
        suffix = dp.sfx(t)
        data = read(f'y_res{layer-1}{suffix}.bin' if layer else f'xres{t}.bin')
        if len(data) != spec.hidden*4:
            raise ValueError('wrong residual size')
        inputs.append(dp.finite(np.frombuffer(data, '<f4')))
        act = read(f'y_act{layer}{suffix}.bin')
        c = dp.read_boundaries(act, read(f'y_state{layer}{suffix}.bin'), spec, offsets, conv_dtype=conv_dtype)
        c['xn'] = dp.di.read_entry(act, offsets, spec.hidden, norm_dtype=norm_dtype)
        captures.append(c)
    return spec, meta, weights, chunks, inputs, captures, dict(
        build_key=bundle['build_key'], spec_hash=bundle['spec_hash'],
        fixture_sha256=digest(fixture/'decode_reference.json'),
        capture_sha256=hashes, weight_sha256=weight_hashes,
        chunks_sha256=hashlib.sha256(chunks.tobytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('model-dir', 'kernel-dir', 'fixture', 'out'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--layer', type=int, default=34)
    a = p.parse_args()
    spec, meta, w, chunks, inputs, captures, provenance = load(a.model_dir, a.kernel_dir, a.fixture, a.layer)
    histories = {name: np.zeros((3, spec.lin_qkv_dim)) for name in
                 ('primary', 'exact_entry', 'original', 'quantized')}
    results = []
    for t, (x, c) in enumerate(zip(inputs, captures)):
        xn = c['xn']; exact = x/np.sqrt(np.mean(x*x)+spec.norm_eps)*w['norm']
        projections = dict(primary=w['qkv']@exact, exact_entry=w['qkv']@xn,
                           original=project(chunks, xn), quantized=project(chunks, xn, 'quantized'))
        records = {}
        for name, y in projections.items():
            records[name], histories[name] = dp.di.records_reference(
                spec, exact if name == 'primary' else xn, y, histories[name], w)
        results.append(dict(token=t,
            projection_vs_capture={n: metric(y, c['qkv']) for n, y in projections.items()},
            projection_vs_exact_entry={n: metric(y, projections['exact_entry']) for n, y in projections.items()},
            records_vs_primary={n: dp.di.record_metrics(r, records['primary']) for n, r in records.items()},
            observed_records_vs_primary=dp.di.record_metrics(c['records'], records['primary']),
            original_projection_byte_exact=bool(np.array_equal(projections['original'], c['qkv'])),
            original_records_vs_capture=dp.di.record_metrics(records['original'], c['records'])))
    report = dict(scope=__doc__, diagnostic_only=True, layer=a.layer, **provenance, results=results,
                  source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for r in results:
        print(r['token'], 'emulation/capture', r['projection_vs_capture']['original']['maxrel'],
              'K original/quantized',*[r['records_vs_primary'][n]['k']['maxrel'] for n in ('original', 'quantized')])


if __name__ == '__main__':
    main()
