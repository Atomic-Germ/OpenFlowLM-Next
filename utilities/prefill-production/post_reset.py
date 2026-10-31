"""Audit two resident short requests against the fresh FP32 prefill baseline."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location('_post_prefill', Path(__file__).with_name('post_validate.py'))
post = importlib.util.module_from_spec(spec)
spec.loader.exec_module(post)
old, base, builder, digest = post.old, post.base, post.builder, post.digest


def sources():
    return post.sources() | {str(Path(__file__).resolve()): digest(__file__)}


def baseline(record):
    directory = Path(record['baseline'])
    hashes = {name: digest(directory/name) for name in ('report.json', 'audit.json')}
    if hashes != record['baseline_hashes']:
        raise ValueError('baseline report or audit changed')
    saved = json.loads((directory/'report.json').read_text())
    proof = json.loads((directory/'audit.json').read_text())
    if not (saved['complete'] and saved['passed'] and saved['model_acceptance'] is False
            and proof['complete'] and proof['passed'] and proof['model_acceptance'] is False
            and proof['report_sha256'] == hashes['report.json'] and saved['sources'] == post.sources()):
        raise ValueError('requires a bound passing current FP32 baseline audit')
    meta, refs, offsets = post.context(saved)
    for kind in ('plain', 'trace'):
        if digest(directory/kind/'ids.txt') != meta['ids_sha256']:
            raise ValueError('baseline prompt changed')
    plain, traced, schedule = post.measure(directory, refs, meta, offsets,
                                          Path(saved['plain_build']), Path(saved['trace_build']))
    if (plain, traced, schedule) != (saved['plain'], saved['routes'], saved['schedule_identity']):
        raise ValueError('baseline measurements changed')
    post.accept(plain, traced)
    return meta, refs, offsets, {row['route']: row['first']['outputs'] for row in traced}


def context(record):
    if record['sources'] != sources():
        raise ValueError('validator source inventory changed')
    base.verify(record['sources'])
    built = builder.verify(Path(record['build']), True)
    if built['cli_sha256'] != record['cli_sha256']:
        raise ValueError('repeat CLI changed')
    return baseline(record)


def measure(out, record, meta, refs, offsets, controls):
    rows = [old.measure(out, refs, meta, offsets, controls, route, True,
                        old.command(Path(record['build']), out, meta, offsets, route, True))
            for route in old.runner.ROUTES]
    return rows, old.schedule_identity(rows)


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(complete=False, passed=False, model_acceptance=False, routes=[], sources=sources(),
                  build=str(args.build.resolve()), baseline=str(args.baseline.resolve()),
                  scope='Two resident 20-token requests plus continuation, 64 layers, three routes; '
                        'ordinary request reset, no long-prefix/serving/performance acceptance')
    try:
        record['baseline_hashes'] = {name: digest(args.baseline.resolve()/name)
                                     for name in ('report.json', 'audit.json')}
        record['cli_sha256'] = builder.verify(Path(record['build']), True)['cli_sha256']
        print('verifying current builds, FP32 baseline and independent reference measurements', flush=True)
        meta, refs, offsets, controls = context(record)
        (out/'ids.txt').write_bytes((refs/'ids.txt').read_bytes())
        env = old.smoke.capture.environment()
        env.update(XILINX_XRT='/opt/xilinx/xrt', LD_LIBRARY_PATH='/opt/xilinx/xrt/lib', OPENBLAS_NUM_THREADS='1')
        with old.smoke.capture.device_lock():
            for route in old.runner.ROUTES:
                base.memory_preflight(meta['layers'])
                old.runner.device_available()
                cmd = old.command(Path(record['build']), out, meta, offsets, route, True)
                raw = out/(route+'-requests.log')
                print('running resident repeat', route, flush=True)
                with raw.open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=1800)
                for prefix, text in zip((route, route+'_repeat'), old.repeat_tool.sections(raw.read_text())):
                    (out/(prefix+'.log')).write_text(text)
                record['routes'].append(old.measure(out, refs, meta, offsets, controls, route, True, cmd))
                (out/'report.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        record['schedule_identity'] = old.schedule_identity(record['routes'])
        old.aggregate(record['routes'], True)
        context(record)
        if digest(out/'ids.txt') != meta['ids_sha256']:
            raise ValueError('prompt changed')
        record.update(complete=True, passed=True)
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        (out/'report.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def audit(args):
    out = args.out.resolve()
    proof = dict(complete=False, passed=False, model_acceptance=False)
    try:
        record = json.loads((out/'report.json').read_text())
        if not record['complete'] or not record['passed'] or record['model_acceptance'] is not False:
            raise ValueError('incomplete result')
        meta, refs, offsets, controls = context(record)
        if digest(out/'ids.txt') != meta['ids_sha256']:
            raise ValueError('prompt changed')
        rows, schedule = measure(out, record, meta, refs, offsets, controls)
        if (rows, schedule) != (record['routes'], record['schedule_identity']):
            raise ValueError('reported measurements differ')
        old.aggregate(rows, True)
        proof.update(complete=True, passed=True, report_sha256=digest(out/'report.json'))
    except Exception as error:
        proof['error'] = str(error)
        raise
    finally:
        (out/'audit.json').write_text(json.dumps(proof, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run_parser = sub.add_parser('run')
    for key in ('build', 'baseline', 'out'):
        run_parser.add_argument('--'+key, type=Path, required=True)
    sub.add_parser('audit').add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    {'run': run, 'audit': audit}[args.action](args)
