"""Replay one QKV projection; score causal CPU records, not full NPU layers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from diagnose import load, digest, dp, metric
from qkv_reference import activation_table, f32_to_bf16, bf16_to_f32, ROOT
from qkv_build import verify_generated
from recipes.pack import std_perm


def table_bytes(x, mode):
    t = activation_table(x, mode)
    return (t['xi'].astype('<i2').tobytes()+t['shift'].astype('<i4').tobytes()
            +f32_to_bf16(t['hi']).astype('<u2').tobytes()+f32_to_bf16(t['lo']).astype('<u2').tobytes())


def validate_table(x, data, mode):
    expected = table_bytes(x, mode)
    if len(data) != len(expected):
        raise ValueError('invalid table size')
    k, nb = len(x), len(x)//32
    integer_exact = data[:2*k+4*nb] == expected[:2*k+4*nb]
    hi, lo = (bf16_to_f32(np.frombuffer(data, '<u2', count=nb, offset=2*k+off*nb)).astype(float)
              for off in (4, 6))
    # The AIE FP32 reduction can differ slightly from NumPy's FP32 additions.
    # Validate original sums against FP64 with a reduction + BF16 pair bound;
    # only the candidate's integer-derived sums promise exact table identity.
    values = (np.asarray(x, np.float64).reshape(-1, 32) if mode == 'original'
              else activation_table(x, mode)['reconstructed'].astype(np.float64))
    ref = values.sum(axis=1)
    bound = np.maximum(np.abs(values).sum(axis=1)*8*np.finfo(np.float32).eps
                       +abs(ref)*2**-15, np.finfo(np.float32).tiny)
    sum_ok = bool(np.isfinite(hi+lo).all() and np.all(abs(hi+lo-ref) <= bound))
    exact = data == expected
    return dict(passed=bool(integer_exact and sum_ok and (mode == 'original' or exact)),
                quantized_input_byte_exact=integer_exact, table_byte_exact=exact,
                sum_passed=sum_ok, max_sum_error=float(np.max(abs(hi+lo-ref))))


def acceptance(mode, byte_exact, table_valid, records, projection):
    primary = projection['passed'] and all(m['passed'] for m in records.values())
    return bool(table_valid and (byte_exact if mode == 'control' else primary)), bool(primary)


def verify_build(build):
    meta = json.loads((build/'build.json').read_text())
    if verify_generated(build/'generated', meta['mode']) != meta['generated_sha256']:
        raise ValueError('generated source identity mismatch')
    if digest(Path(__file__).with_name('qkv_build.py')) != meta['builder_sha256']:
        raise ValueError('builder changed since compilation')
    for name, h in meta['sha256'].items():
        if digest(build/name) != h:
            raise ValueError('binary identity mismatch')
    return meta


def execute(a):
    for p in (a.model_dir, a.kernel_dir, a.fixture, a.build, a.pools):
        if a.out.resolve() == p.resolve() or a.out.resolve().is_relative_to(p.resolve()):
            raise ValueError('output must be separate from inputs')
    build = verify_build(a.build)
    s, meta, w, chunks, inputs, captures, provenance = load(a.model_dir, a.kernel_dir, a.fixture, a.layer)
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    record = out/'inputs.json'
    source_hashes = {str(p.resolve()): digest(p) for p in Path(__file__).parent.glob('*.py')}
    if not a.score:
        record.unlink(missing_ok=True); (out/'score.json').unlink(missing_ok=True)
        packed = chunks[std_perm(len(chunks), s.hidden)].tobytes()
        # Compare the extracted source tensor with the actual retained production
        # pool region; neither ordering nor requantization is assumed equivalent.
        pool = a.pools/f'pool_L{a.layer}.bin'
        offsets = dp.layout(s, max_ctx=meta['max_ctx'])
        with pool.open('rb') as f:
            f.seek(offsets.POOL_QKV)
            if f.read(len(packed)) != packed:
                raise ValueError('QKV tensor/pool mismatch')
        (out/'weights.bin').write_bytes(packed)
        np.full(s.lin_qkv_dim, np.nan, '<f4').tofile(out/'poison.bin')
        lines = ['device', f'xclbin P {a.build.resolve()}/final.xclbin',
                 f'kernelx qkv P {a.build.resolve()}/insts.bin',
                 f'buf w {len(packed)} {out}/weights.bin', 'buf x 20480', 'buf y 40960', 'buf tab 11520']
        files = [pool, out/'weights.bin', out/'poison.bin', a.build/'build.json',
                 a.build/'final.xclbin', a.build/'insts.bin']
        for t, c in enumerate(captures):
            c['xn'].astype('<f4').tofile(out/f'x{t}.bin'); files.append(out/f'x{t}.bin')
            for name in (f'qkv{t}.bin', f'table{t}.bin'):
                (out/name).unlink(missing_ok=True)
            lines += [f'load x {out}/x{t}.bin', f'load y {out}/poison.bin', 'run qkv w x y tab',
                      f'dump y {out}/qkv{t}.bin 40960', f'dump tab {out}/table{t}.bin 11520']
        (out/'run.cfg').write_text('\n'.join(lines)+'\n'); files.append(out/'run.cfg')
        record.write_text(json.dumps(dict(mode=build['mode'], layer=a.layer, tokens=meta['tokens'],
            provenance=provenance, source_sha256=source_hashes,
            sha256={str(p.resolve()): digest(p) for p in files}), indent=2)+'\n')
        print('Prepared', build['mode'], meta['tokens'], 'projections; source/pool byte equality verified')
        return 0
    (out/'score.json').unlink(missing_ok=True)
    pinned = json.loads(record.read_text())
    if (pinned['mode'] != build['mode'] or pinned['layer'] != a.layer
            or pinned['tokens'] != meta['tokens'] or pinned['provenance'] != provenance
            or pinned['source_sha256'] != source_hashes):
        raise ValueError('experiment identity changed since preparation')
    for path, h in pinned['sha256'].items():
        if digest(path) != h:
            raise ValueError('input identity mismatch: '+path)
    histories = {n: np.zeros((3, s.lin_qkv_dim)) for n in ('primary', 'observed', 'candidate')}
    results, hashes = [], {}
    for t, (x, c) in enumerate(zip(inputs, captures)):
        data = (out/f'qkv{t}.bin').read_bytes(); table = (out/f'table{t}.bin').read_bytes()
        if len(data) != s.lin_qkv_dim*4 or len(table) != 11520:
            raise ValueError('incomplete output/table capture')
        y = dp.finite(np.frombuffer(data, '<f4'))
        exact = x/np.sqrt(np.mean(x*x)+s.norm_eps)*w['norm']
        ref = w['qkv']@exact
        records = {}
        for name, (norm, projection) in dict(primary=(exact, ref), observed=(c['xn'], c['qkv']),
                                            candidate=(c['xn'], y)).items():
            records[name], histories[name] = dp.di.records_reference(s, norm, projection, histories[name], w)
        fields = dp.di.record_metrics(records['candidate'], records['primary'])
        proj = metric(y, ref)
        exact_bytes = data == c['qkv'].astype('<f4').tobytes()
        table_score = validate_table(c['xn'], table, 'original' if build['mode'] == 'control' else 'quantized')
        passed, primary = acceptance(build['mode'], exact_bytes, table_score['passed'], fields, proj)
        results.append(dict(token=t, passed=passed, primary_passed=primary,
                            production_qkv_byte_exact=exact_bytes, table=table_score,
                            projection=proj, records_vs_primary=fields,
                            production_records_vs_primary=dp.di.record_metrics(c['records'], records['primary']),
                            production_projection_records_vs_primary=dp.di.record_metrics(records['observed'], records['primary'])))
        hashes.update({f'qkv{t}.bin': hashlib.sha256(data).hexdigest(), f'table{t}.bin': hashlib.sha256(table).hexdigest()})
    report = dict(scope=__doc__, mode=build['mode'], layer=a.layer, tokens=meta['tokens'],
                  passed=all(r['passed'] for r in results), primary_passed=all(r['primary_passed'] for r in results),
                  maxrel_bound=dp.di.MAXREL, conditioning='captured FP32 norm, independent CPU conv histories from zero',
                  inputs_sha256=digest(record), capture_sha256=hashes, results=results)
    (out/'score.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for r in results:
        print(r['token'], 'PASS' if r['passed'] else 'FAIL',
              'identity', r['production_qkv_byte_exact'], 'table', r['table']['passed'],
              'K', r['records_vs_primary']['k']['maxrel'])
    return 0 if report['passed'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('model-dir', 'kernel-dir', 'fixture', 'pools', 'build', 'out'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--layer', type=int, default=34); p.add_argument('--score', action='store_true')
    return execute(p.parse_args())


if __name__ == '__main__':
    sys.exit(main())
