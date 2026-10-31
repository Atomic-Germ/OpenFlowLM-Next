"""Replay candidate linear layers causally; require exact native identity first."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import importlib.util

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load('_sublayer_runtime', ROOT/'utilities/refined-runtime/check.py')
native = load('_sublayer_native', ROOT/'utilities/prefill-sequential-drift/native_l0.py')
builder = load('_sublayer_harness', ROOT/'utilities/prefill-upgate-replay/check.py')
base, digest, write = runtime.base, runtime.digest, runtime.write
pack = native.pack
TARGETS = {50: 205, 54: 125, 60: 220}


def count(layer):
    if type(layer) is not int or layer not in TARGETS:
        raise ValueError('unknown causal target layer')
    return TARGETS[layer]+1


def names(layer):
    return [f'{kind}_{p}.bin' for p in range(count(layer)) for kind in ('xres', 'act', 'state')]


def program(out, kernels, sizes, layer):
    return native.program(out, kernels, sizes, count(layer))


def identity(actual, expected, layer):
    if len(actual) != count(layer) or len(expected) != count(layer):
        raise ValueError('incomplete causal identity coverage')
    return native.identity(actual, expected)


def complete_identity(rows):
    if set(rows) != {str(l) for l in TARGETS}:
        raise ValueError('incomplete target identity coverage')
    return all(rows[str(l)] == dict(passed=True, positions=count(l),
               matching_positions=count(l), first_mismatch=None) for l in TARGETS)


def require_baseline(report, audit, sha):
    if (report.get('complete') is not True or audit.get('complete') is not True or
            report.get('model_acceptance') is not False or audit.get('model_acceptance') is not False or
            type(report.get('candidate_passed')) is not bool or
            audit.get('candidate_passed') != report['candidate_passed'] or audit.get('report_sha256') != sha):
        raise ValueError('requires complete bound native audit, including its numerical verdict')


def sources():
    return runtime.sources() | native.sources() | {
        str(Path(p).resolve()): digest(p) for p in (__file__, builder.__file__)}


def baseline(directory):
    report = json.loads((directory/'report.json').read_text())
    audit = json.loads((directory/'audit.json').read_text())
    require_baseline(report, audit, digest(directory/'report.json'))
    prepared = Path(report['prepared'])
    if report['sources'] != runtime.sources() or report['preparation_sha256'] != digest(prepared/'preparation.json'):
        raise ValueError('changed native runtime provenance')
    record, meta, _, offsets = runtime.context(prepared)
    inputs = {str(directory/n): digest(directory/n) for n in ('report.json', 'audit.json')}
    inputs[str(prepared/'preparation.json')] = digest(prepared/'preparation.json')
    for role in runtime.ROLES:
        for kind in ('plain', 'trace'):
            outputs = report['scores'][role][kind]['outputs']
            runtime.post.inventory(directory/role/kind, 'sequential', runtime.TOKENS,
                                   runtime.LAYERS, kind == 'trace')
            inputs.update({str(directory/role/kind/n): h for n,h in outputs.items()})
    refs = Path(record['references'])
    inputs.update({str(refs/n): h for n,h in meta['references'].items()})
    base.verify(inputs)
    spec = base.spec_from_model_dir(Path(meta['model']))
    kernels = prepared/'candidate'
    bundle = base.load_bundle(kernels, base.manifest(spec, max_ctx=meta['max_ctx']))
    b = bundle['layer_types']['linear_attention']['buffers']
    sizes = dict(pool=bundle['layout']['pool_bytes'], consts=b['consts'], act=b['act'],
                 state=b['state']['bytes'], hidden=spec.hidden)
    if (spec.family != 'qwen35' or spec.hidden != 5120 or spec.num_layers != 64 or
            any(spec.layer_types[l] != 'linear_attention' for l in TARGETS)):
        raise ValueError('requires the retained 27B linear targets')
    return meta, refs, spec, bundle, sizes, kernels, inputs


def harness(build):
    evidence = json.loads((build/'build-evidence.json').read_text())
    if evidence.get('complete') is not True or not evidence['compiler_dependencies']:
        raise ValueError('incomplete harness build')
    required = {str(ROOT/'open_kernels/harness'/n) for n in ('run_kernel.cpp', 'stream_patch.hpp')}
    if not required <= set(evidence['compiler_dependencies']):
        raise ValueError('incomplete harness compiler dependency coverage')
    base.verify(evidence['sources'])
    base.verify(evidence['compiler_dependencies'])
    base.verify({str(build/'build/run_kernel'): evidence['harness_sha256'],
                 str(build/'build/CMakeCache.txt'): evidence['cmake_cache_sha256']})
    return digest(build/'build-evidence.json')


def trace(path, rows, hidden):
    return native.drift.trace.read_trace(path, rows, hidden).astype('<f4')


def prepare(args):
    out = args.out.resolve()
    if not out.is_relative_to(ROOT/'Models'):
        raise ValueError('private output must be under Models')
    out.mkdir(parents=True, exist_ok=False)
    record = dict(complete=False, model_acceptance=False, sources=sources())
    try:
        directory, build = args.baseline.resolve(), args.harness_build.resolve()
        record.update(baseline=str(directory), harness_build=str(build), harness_evidence=harness(build))
        meta, refs, spec, bundle, sizes, kernels, inputs = baseline(directory)
        model = base.Q4NX(Path(meta['model'])/'model.q4nx')
        try:
            plan = dict(pool_bytes=bundle['pack']['pool_bytes'], chunk_bytes=bundle['pack']['chunk_bytes'],
                        layer_types={k:v['pack'] for k,v in bundle['layer_types'].items()})
            for layer in TARGETS:
                target = out/f'L{layer}'
                target.mkdir()
                pack.build_layer_pool(plan, 'linear_attention', model, layer).tofile(target/'pool.bin')
                pack.build_consts(plan, 'linear_attention', model, layer, sizes['consts']).tofile(target/'consts.bin')
                xs = trace(directory/'candidate/trace'/f'sequential_res{layer-1}.bin', 258, spec.hidden)
                for p in range(count(layer)):
                    xs[p].tofile(target/f'input_{p}.bin')
                (target/'run.cfg').write_text(program(target, kernels, sizes, layer))
        finally:
            model.mm.close()
            model.f.close()
        artifacts = {str(p): digest(p) for p in out.rglob('*') if p.is_file()}
        base.verify(record['sources'])
        base.verify(inputs)
        record.update(complete=True, meta=meta, references=str(refs), kernels=str(kernels),
                      sizes=sizes, inputs=inputs, artifacts=artifacts,
                      targets={str(l):p for l,p in TARGETS.items()})
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        write(out/'preparation.json', record)


def verified(out):
    record = json.loads((out/'preparation.json').read_text())
    if (record.get('complete') is not True or record.get('model_acceptance') is not False or
            record['sources'] != sources() or record['targets'] != {str(l):p for l,p in TARGETS.items()}):
        raise ValueError('incomplete or changed replay preparation')
    meta, refs, _, _, sizes, kernels, inputs = baseline(Path(record['baseline']))
    if (meta != record['meta'] or str(refs) != record['references'] or sizes != record['sizes'] or
            str(kernels) != record['kernels'] or inputs != record['inputs'] or
            harness(Path(record['harness_build'])) != record['harness_evidence']):
        raise ValueError('replay input or harness identity changed')
    base.verify(record['artifacts'])
    for layer in TARGETS:
        target = out/f'L{layer}'
        if (target/'run.cfg').read_text() != program(target, kernels, sizes, layer):
            raise ValueError('replay schedule changed')
    return record


def read_outputs(out, sizes, layer):
    expected = set(names(layer))
    actual = {p.name for pattern in ('xres_*.bin', 'act_*.bin', 'state_*.bin') for p in out.glob(pattern)}
    if actual != expected:
        raise ValueError('incomplete or extra replay capture inventory')
    return native.read_outputs(out, sizes, count(layer))


def score(out, record):
    rows, hashes = {}, {}
    for layer in TARGETS:
        got, captured = read_outputs(out/f'L{layer}', record['sizes'], layer)
        expected = trace(Path(record['baseline'])/'candidate/trace'/f'sequential_res{layer}.bin',
                         258, record['sizes']['hidden'])[:count(layer)]
        rows[str(layer)] = identity(got, expected, layer)
        hashes.update({f'L{layer}/{n}': h for n,h in captured.items()})
    return dict(identity=rows, control_valid=complete_identity(rows), artifacts=hashes)


def run(args):
    out = args.out.resolve()
    if (out/'report.json').exists() or any((out/f'L{l}'/n).exists() for l in TARGETS for n in names(l)):
        raise FileExistsError('use a fresh replay preparation')
    report = dict(complete=False, model_acceptance=False, commands=[])
    try:
        record = verified(out)
        report['preparation_sha256'] = digest(out/'preparation.json')
        exe = Path(record['harness_build'])/'build/run_kernel'
        env = os.environ.copy()
        env.update(HARNESS_TIMEOUT_MS='30000', OPENBLAS_NUM_THREADS='1')
        with native.drift.boundary.runner.device_lock():
            for layer in TARGETS:
                runtime.post.old.runner.device_available()
                # Reuse the conservative four-layer budget: the shared helper
                # accepts only 4/64, and this replay holds just one layer pool.
                base.memory_preflight(4)
                target = out/f'L{layer}'
                cmd = [str(exe), str(target/'run.cfg')]
                report['commands'].append(cmd)
                print('replaying', layer, 'positions', count(layer), flush=True)
                with (target/'native.log').open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=1800)
        report.update(score(out, record))
        verified(out)
        report.update(complete=True, logs={f'L{l}/native.log': digest(out/f'L{l}/native.log') for l in TARGETS},
                      boundary_interpretation_allowed=report['control_valid'])
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        write(out/'report.json', report)
    return 0 if report['control_valid'] else 1


def audited(out):
    record = verified(out)
    report = json.loads((out/'report.json').read_text())
    if (report.get('complete') is not True or report.get('model_acceptance') is not False or
            report['preparation_sha256'] != digest(out/'preparation.json')):
        raise ValueError('incomplete or changed native replay')
    expected_commands = [[str(Path(record['harness_build'])/'build/run_kernel'), str(out/f'L{l}/run.cfg')]
                         for l in TARGETS]
    if report['commands'] != expected_commands or set(report['logs']) != {f'L{l}/native.log' for l in TARGETS}:
        raise ValueError('incomplete native execution inventory')
    measured = score(out, record)
    if any(report[k] != v for k,v in measured.items()) or report['boundary_interpretation_allowed'] != measured['control_valid']:
        raise ValueError('native replay identity or inventory changed')
    base.verify({str(out/n): h for n,h in report['logs'].items()})
    return record, measured


def audit(args):
    out = args.out.resolve()
    proof = dict(complete=False, model_acceptance=False)
    try:
        _, measured = audited(out)
        proof.update(complete=True, control_valid=measured['control_valid'], identity=measured['identity'],
                     artifacts=len(measured['artifacts']), report_sha256=digest(out/'report.json'))
    except Exception as error:
        proof['error'] = str(error)
        raise
    finally:
        write(out/'audit.json', proof)
    return 0 if proof['control_valid'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action, flags in (('build', ('out', 'xrt-include', 'xrt-lib')),
                          ('prepare', ('out', 'baseline', 'harness-build')),
                          ('run', ('out',)), ('audit', ('out',))):
        command = sub.add_parser(action)
        for flag in flags:
            command.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit((builder.build(args) if args.action == 'build' else globals()[args.action](args)) or 0)
