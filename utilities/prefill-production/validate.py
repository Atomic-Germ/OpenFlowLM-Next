"""Audit shared-runtime all-layer traces and resident repeats against pinned data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

import importlib.util
s = importlib.util.spec_from_file_location('_production_builder', Path(__file__).with_name('build.py'))
builder = importlib.util.module_from_spec(s)
s.loader.exec_module(builder)
smoke = builder.load('_production_smoke', Path(__file__).with_name('smoke.py'))
runner, repeat_tool = smoke.capture.original, builder.repeat_tool
base, digest = smoke.base, smoke.digest


def identity(actual, expected, prefix, control_prefix, tokens, layers):
    names = runner.capture_names(prefix, tokens, layers)
    controls = runner.capture_names(control_prefix, tokens, layers)
    if set(actual) != set(names) or set(expected) != set(controls):
        raise ValueError('incomplete or extra identity inventory')
    checks = {a: actual[a] == expected[b] for a,b in zip(names, controls)}
    if not all(checks.values()):
        raise ValueError('capture identity failed: '+', '.join(k for k,v in checks.items() if not v))
    return checks


def aggregate(rows, repeat):
    if [r['route'] for r in rows] != list(runner.ROUTES):
        raise ValueError('incomplete or unordered routes')
    for row in rows:
        keys = ('identity', 'repeat_identity') if repeat else ('identity',)
        if not row['first']['passed'] or (repeat and not row['second']['passed']):
            raise ValueError('independent numerical gate failed')
        if any(not row[k] or not all(row[k].values()) for k in keys):
            raise ValueError('identity gate failed')
    return True


def sources():
    paths = [Path(__file__), Path(builder.__file__), Path(smoke.__file__), Path(runner.__file__),
             Path(repeat_tool.__file__), Path(base.__file__), Path(runner.trace.__file__),
             Path(builder.trace.__file__), Path(builder.trace.paired.__file__)]
    return {str(p.resolve()): digest(p) for p in paths}


def fixture(locations):
    smoke_dir, baseline, kernels = (Path(locations[k]) for k in ('smoke', 'baseline', 'kernels'))
    known = json.loads((smoke_dir/'report.json').read_text())
    if not known['complete'] or not known['passed'] or set(known['routes']) != {'hybrid','block_major'}:
        raise ValueError('requires a complete passing unpatched CLI smoke')
    # Reuse only pinned input data, not an exemption for historical binaries.
    base.verify(known['inputs'])
    for p in (baseline/'report.json', baseline/'audit.json', kernels/'manifest.json', kernels/'toolchain.json'):
        if known['inputs'].get(str(p)) != digest(p):
            raise ValueError('fixture differs from integrated smoke: '+str(p))
    original = json.loads((baseline/'report.json').read_text())
    audit = json.loads((baseline/'audit.json').read_text())
    if not (original['complete'] and original['passed'] and audit['complete'] and audit['passed']
            and audit['report_sha256'] == digest(baseline/'report.json')):
        raise ValueError('historical captures lack passing audit')
    prepared = Path(original['prepared'])
    meta = smoke.read_fixture(prepared, original['preparation_sha256'])
    if meta['layers'] != 64 or meta['hidden'] != 5120 or len(meta['ids']) != 20:
        raise ValueError('expected full-depth short fixture')
    base.check_reference_coverage(meta)
    refs = prepared/'references'
    base.verify({str(refs/p):h for p,h in meta['references'].items()})
    controls = {row['route']:row['outputs'] for row in original['variants']['expanded']['routes']}
    if set(controls) != set(runner.ROUTES):
        raise ValueError('incomplete historical controls')
    for route, result in known['routes'].items():
        outputs = smoke.check_outputs(smoke_dir, route, len(meta['ids']), meta['vocab'], controls[route])
        if not result['passed'] or outputs != result['outputs']:
            raise ValueError('unpatched smoke captures changed')
    spec = base.spec_from_model_dir(Path(meta['model']))
    selected = json.loads((kernels/'manifest.json').read_text())
    base.load_bundle(kernels, base.manifest(spec, max_ctx=selected['max_ctx_default']))
    if selected['manifest_version'] != 4 or selected['spec_hash'] != meta['spec_hash']:
        raise ValueError('integrated manifest/spec mismatch')
    offsets = base.layout(spec, max_ctx=meta['max_ctx'])
    return meta | {'kernels':str(kernels)}, refs, offsets, controls


def command(build, out, meta, offsets, route, repeated):
    return base.native_command(build/'build/open_qwen36_cli', out, meta, offsets, route,
                               runner.ROUTES[route]+(['--twice'] if repeated else []))


def capture_inventory(out, prefix, tokens, layers):
    expected = set(runner.capture_names(prefix, tokens, layers))
    actual = {p.name for pattern in (prefix+'_p*.bin', prefix+'_t*.bin',
                                    prefix+'_res*.bin', prefix+'-act.bin') for p in out.glob(pattern)}
    if actual != expected:
        raise ValueError('incomplete or extra capture inventory: '+prefix)


def measure(out, refs, meta, offsets, controls, route, repeated, cmd):
    capture_inventory(out, route, len(meta['ids']), meta['layers'])
    first = runner.score_route(out, refs, meta, offsets, route)
    row = dict(route=route, first=first, command=cmd,
               identity=identity(first['outputs'], controls[route], route, route, len(meta['ids']), meta['layers']))
    if repeated:
        capture_inventory(out, route+'_repeat', len(meta['ids']), meta['layers'])
        raw = out/(route+'-requests.log')
        parts = repeat_tool.sections(raw.read_text())
        for prefix, text in zip((route, route+'_repeat'), parts):
            if (out/(prefix+'.log')).read_text() != text:
                raise ValueError('request log extraction changed')
        second = runner.score_route(out, refs, meta, offsets, route+'_repeat')
        row.update(second=second, raw_log_sha256=digest(raw),
                   repeat_identity=identity(second['outputs'], first['outputs'], route+'_repeat', route,
                                            len(meta['ids']), meta['layers']))
    return row


def schedule_identity(rows):
    by_route = {r['route']:r for r in rows}
    a, b = by_route['hybrid']['first']['outputs'], by_route['block_major']['first']['outputs']
    checks = {n:a[n] == b[n.replace('hybrid','block_major',1)] for n in a if not n.endswith('-act.bin')}
    if not checks or not all(checks.values()):
        raise ValueError('hybrid schedule identity failed')
    return checks


def context(record):
    repeated = record['repeat_baseline'] is not None
    b = builder.verify(Path(record['build']), repeated)
    if b['cli_sha256'] != record['cli_sha256']:
        raise ValueError('capture CLI changed')
    meta, refs, offsets, controls = fixture(record['fixture'])
    if repeated:
        previous = Path(record['repeat_baseline'])
        old = json.loads((previous/'report.json').read_text())
        proof = json.loads((previous/'audit.json').read_text())
        if (not old['complete'] or not old['passed'] or old['repeat_baseline'] is not None
                or not proof['complete'] or not proof['passed']
                or digest(previous/'report.json') != record['baseline_sha256']
                or proof['report_sha256'] != record['baseline_sha256'] or old['fixture'] != record['fixture']):
            raise ValueError('repeat needs a passing audit of this single-request fixture')
        builder.verify(Path(old['build']), False)
        # Recompute the single-request result before using it as a reset control.
        measured = [measure(previous, refs, meta, offsets, controls, route, False,
                            command(Path(old['build']), previous, meta, offsets, route, False))
                    for route in runner.ROUTES]
        if measured != old['routes'] or not aggregate(measured, False):
            raise ValueError('single-request measurements changed')
        controls = {row['route']:row['first']['outputs'] for row in measured}
    return meta, refs, offsets, controls


def run(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repeated = a.repeat_baseline is not None
    r = dict(complete=False, passed=False, model_acceptance=False, routes=[], sources=sources(),
             build=str(a.build.resolve()), repeat_baseline=str(a.repeat_baseline.resolve()) if repeated else None,
             fixture={k:str(getattr(a,k).resolve()) for k in ('smoke','baseline','kernels')},
             scope='64 layers, 20-token prompt plus continuation; normal request reset, no serving/long-prefix claim')
    try:
        r['cli_sha256'] = builder.verify(a.build.resolve(), repeated)['cli_sha256']
        if repeated:
            r['baseline_sha256'] = digest(a.repeat_baseline.resolve()/'report.json')
        print('verifying sources, model, kernel bundle and independent references', flush=True)
        meta, refs, offsets, controls = context(r)
        r['fixture_sha256'] = digest(Path(r['fixture']['smoke'])/'report.json')
        (out/'ids.txt').write_bytes((refs/'ids.txt').read_bytes())
        env = smoke.capture.environment()
        env.update(XILINX_XRT='/opt/xilinx/xrt', LD_LIBRARY_PATH='/opt/xilinx/xrt/lib', OPENBLAS_NUM_THREADS='1')
        with smoke.capture.device_lock():
            for route in runner.ROUTES:
                base.memory_preflight(meta['layers'])
                runner.device_available()
                cmd = command(a.build.resolve(), out, meta, offsets, route, repeated)
                raw = out/(route+('-requests.log' if repeated else '.log'))
                print('running', 'resident repeat' if repeated else 'single trace', route, flush=True)
                with raw.open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=1800)
                if repeated:
                    for prefix,text in zip((route,route+'_repeat'), repeat_tool.sections(raw.read_text())):
                        (out/(prefix+'.log')).write_text(text)
                row = measure(out, refs, meta, offsets, controls, route, repeated, cmd)
                r['routes'].append(row)
                print(route, 'residuals', sum(x['passed'] for x in row['first']['residuals']),
                      '/', len(row['first']['residuals']), 'identity', len(row['identity']), flush=True)
                (out/'report.json').write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')
        r['schedule_identity'] = schedule_identity(r['routes'])
        aggregate(r['routes'], repeated)
        base.verify(r['sources'])
        context(r)  # input/build identities must still hold after execution
        r.update(complete=True, passed=True)
    except Exception as error:
        r['error'] = str(error)
        raise
    finally:
        (out/'report.json').write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')


def audit(a):
    out = a.out.resolve()
    proof = dict(complete=False, passed=False, model_acceptance=False)
    try:
        r = json.loads((out/'report.json').read_text())
        if not r['complete'] or not r['passed'] or r['model_acceptance'] is not False or r['sources'] != sources():
            raise ValueError('incomplete result or changed validator')
        base.verify(r['sources'])
        if digest(Path(r['fixture']['smoke'])/'report.json') != r['fixture_sha256']:
            raise ValueError('smoke fixture changed')
        meta, refs, offsets, controls = context(r)
        if digest(out/'ids.txt') != meta['ids_sha256']:
            raise ValueError('prompt changed')
        repeated = r['repeat_baseline'] is not None
        measured = [measure(out, refs, meta, offsets, controls, route, repeated,
                            command(Path(r['build']), out, meta, offsets, route, repeated))
                    for route in runner.ROUTES]
        if measured != r['routes'] or schedule_identity(measured) != r['schedule_identity']:
            raise ValueError('reported measurements differ')
        aggregate(measured, repeated)
        proof.update(complete=True, passed=True, report_sha256=digest(out/'report.json'))
    except Exception as error:
        proof['error'] = str(error)
        raise
    finally:
        (out/'audit.json').write_text(json.dumps(proof, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('run')
    for key in ('out','build','smoke','baseline','kernels'):
        q.add_argument('--'+key, type=Path, required=True)
    q.add_argument('--repeat-baseline', type=Path)
    sub.add_parser('audit').add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    {'run':run, 'audit':audit}[a.command](a)
