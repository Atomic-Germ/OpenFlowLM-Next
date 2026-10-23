"""Run and independently audit fresh combined prefill controls and candidate."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

spec = importlib.util.spec_from_file_location('_combined_preparation', Path(__file__).with_name('prepare.py'))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
base, trace, digest = prep.base, prep.audit.trace, prep.digest
ROUTES = {'sequential': [], 'hybrid': ['--gemm-block'],
          'block_major': ['--gemm-block', '--block-major']}


def capture_names(route, tokens, layers, residuals=True):
    return (base.expected_dumps(route, tokens)+[route+'-act.bin']+
            ([f'{route}_res{l}.bin' for l in range(layers)] if residuals else []))


def require_identity(before, after, tokens, layers, routes=tuple(ROUTES), residuals=True):
    checks = {name: digest(before/name) == digest(after/name)
              for route in routes for name in capture_names(route, tokens, layers, residuals)}
    if not all(checks.values()):
        raise ValueError('native control identity failed: '+', '.join(k for k,v in checks.items() if not v))
    return checks


def score_route(directory, references, meta, offsets, route):
    n, layers, hidden = len(meta['ids']), meta['layers'], meta['hidden']
    expected = {f'{route}_res{l}.bin' for l in range(layers)}
    if {p.name for p in directory.glob(route+'_res*.bin')} != expected:
        raise ValueError('incomplete or extra residual inventory')
    logits = []
    for p in range(n+1):
        name = f'{route}_p{p}.bin' if p < n else route+'_t1.bin'
        ref = f'ref_p{p}.bin' if p < n else 'ref_cont_logits.bin'
        logits.append(dict(position=p, **base.logit_score(base.read(directory/name, meta['vocab']),
                            base.read(references/ref, meta['vocab']), meta['real_vocab'])))
    residuals = []
    for layer in range(layers):
        ref = np.vstack((trace.read_trace(references/f'ref_res{layer}.bin', n, hidden),
                         trace.read_trace(references/f'ref_cont_res{layer}.bin', 1, hidden)))
        got = trace.read_trace(directory/f'{route}_res{layer}.bin', n+1, hidden)
        residuals.extend(dict(layer=layer, route=route, **r) for r in trace.score_rows(got, ref))
    log = directory/(route+'.log')
    tokens_match = True
    try:
        base.check_tokens(log.read_text(), meta['expected_tokens'])
    except ValueError:
        tokens_match = False
    alias = digest(directory/(route+'_t0.bin')) == digest(directory/f'{route}_p{n-1}.bin')
    tail = base.final_residual((directory/(route+'-act.bin')).read_bytes(), offsets, hidden)
    ref = base.read(references/f'ref_cont_res{layers-1}.bin', hidden)
    error = float(np.max(np.abs(tail-ref))/max(float(np.max(np.abs(ref))), 1e-30))
    logits_passed = tokens_match and alias and error < base.RESIDUAL_MAXREL and all(r['passed'] for r in logits)
    residuals_passed = all(r['passed'] for r in residuals)
    return dict(route=route, passed=logits_passed and residuals_passed,
                logits_passed=logits_passed, residuals_passed=residuals_passed,
                tokens_match=tokens_match, last_prompt_alias=alias,
                continuation_last_residual_maxrel=error, logits=logits, residuals=residuals,
                outputs={name: digest(directory/name) for name in capture_names(route,n,layers)},
                log_sha256=digest(log))


def preparation(directory):
    path = directory/'preparation.json'
    record = json.loads(path.read_text())
    if not record['complete'] or not record['manifest_matches_current']:
        raise ValueError('incomplete preparation')
    base.verify(record['inputs']); base.verify(record['artifacts'])
    for graphs in record['history'].values():
        for graph in graphs.values():
            for name, identity in graph['files'].items():
                if prep.history.verify_historical(name, identity['sha256']) != identity:
                    raise ValueError('historical/current identity changed: '+name)
    for variant, build in record['builds'].items():
        b = prep.verify_build(Path(build['directory']), variant)
        if b['cli_sha256'] != build['cli_sha256']:
            raise ValueError('prepared CLI changed')
    refs = directory/'references'
    meta = json.loads((refs/'reference.json').read_text())
    base.check_reference_coverage(meta)
    base.verify({str(refs/p): h for p,h in meta['references'].items()})
    if digest(refs/'ids.txt') != meta['ids_sha256']:
        raise ValueError('prepared prompt changed')
    model_spec = base.spec_from_model_dir(Path(meta['model']))
    prep.check_manifest(json.loads((Path(meta['kernels'])/'manifest.json').read_text()), model_spec)
    base.load_bundle(Path(meta['kernels']), base.manifest(model_spec, max_ctx=meta['max_ctx']))
    offsets = base.layout(model_spec, max_ctx=meta['max_ctx'])
    return record, meta, offsets


def baseline_path(record, variant):
    graph = record['history'][variant]['native_build']
    paths = [Path(p).parent/'run' for p in graph['records'] if Path(p).name == 'build-evidence.json']
    if len(paths) != 1:
        raise ValueError('ambiguous historical control build')
    return paths[0]


def device_available():
    device = Path('/dev/accel/accel0')
    if not device.exists():
        raise RuntimeError('NPU device missing')
    result = subprocess.run(['fuser', str(device)], capture_output=True, text=True)
    if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
        raise RuntimeError('NPU is busy or its users cannot be checked: '+result.stdout+result.stderr)


def audit_variant(directory, refs, meta, offsets):
    routes = [score_route(directory, refs, meta, offsets, route) for route in ROUTES]
    pairs = {name: digest(directory/name) == digest(directory/name.replace('hybrid','block_major',1))
             for name in capture_names('hybrid',len(meta['ids']),meta['layers']) if not name.endswith('-act.bin')}
    if not all(pairs.values()):
        raise ValueError('hybrid schedule identity failed')
    rows = [row for route in routes for row in route['residuals']]
    return dict(passed=all(r['passed'] for r in routes), routes=routes,
                localization=trace.localize(rows,meta['layers'],len(meta['ids'])+1),
                hybrid_identity=pairs)


def run(a):
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, passed=False, model_acceptance=False, variants={}, controls={})
    try:
        prepared = a.prepared.resolve()
        record, meta, offsets = preparation(prepared)
        refs = prepared/'references'
        n, layers = len(meta['ids']), meta['layers']
        result['preparation_sha256'] = digest(prepared/'preparation.json')
        result['prepared'] = str(prepared)
        result['runner_sha256'] = digest(Path(__file__))
        # Cooperating validators serialize. Check other device users as well.
        lock_path = prep.ROOT/'Models/qwen38-27b/.resident-validation.lock'
        with lock_path.open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for variant in ('paired', 'weighted', 'combined'):
                directory = out/variant; directory.mkdir()
                (directory/'ids.txt').write_bytes((refs/'ids.txt').read_bytes())
                build = record['builds'][variant]
                prep.verify_build(Path(build['directory']), variant)
                cli = Path(build['directory'])/'build/open_qwen36_cli'
                commands = []
                env = {k:v for k,v in os.environ.items() if not k.startswith('OFLM_OPEN_')}
                env.update(OMP_NUM_THREADS='2', OMP_WAIT_POLICY='PASSIVE', OFLM_OPEN_ATTN_BLOCK='0',
                           OFLM_OPEN_LAYER_MAJOR='1', OFLM_OPEN_STAGE_MAJOR='0')
                for route, extra in ROUTES.items():
                    base.memory_preflight(layers); device_available()
                    command = base.native_command(cli,directory,meta,offsets,route,extra)
                    commands.append(command)
                    print('running',variant,route,flush=True)
                    with (directory/(route+'.log')).open('w') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, check=True, timeout=1800)
                    score = score_route(directory,refs,meta,offsets,route)
                    print(variant,route,'logits',score['logits_passed'],'residuals',
                          sum(r['passed'] for r in score['residuals']), '/', len(score['residuals']),flush=True)
                report = audit_variant(directory,refs,meta,offsets)
                report.update(cli_sha256=digest(cli), commands=commands)
                result['variants'][variant] = report
                if variant in ('paired','weighted'):
                    result['controls'][variant] = require_identity(baseline_path(record,variant),directory,n,layers,
                                                                   residuals=variant=='weighted')
                if variant != 'paired':
                    require_identity(out/'paired',directory,n,layers,('sequential',))
                (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            result.update(complete=True, passed=result['variants']['combined']['passed'],
                          scope='64-layer 20-token prompt and one continuation; repeat/reset pending')
    except Exception as error:
        result.update(complete=False,passed=False,error=str(error))
        raise
    finally:
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['passed'] else 1


def audit(a):
    out = a.out.resolve(); result = dict(complete=False,passed=False,model_acceptance=False)
    try:
        original = json.loads((out/'report.json').read_text())
        prepared = Path(original['prepared'])
        if digest(prepared/'preparation.json') != original['preparation_sha256']:
            raise ValueError('preparation changed')
        if digest(Path(__file__)) != original['runner_sha256']:
            raise ValueError('runner changed')
        record,meta,offsets = preparation(prepared)
        for variant in ('paired','weighted','combined'):
            measured = audit_variant(out/variant,prepared/'references',meta,offsets)
            saved = original['variants'][variant]
            if any(saved.get(k) != v for k,v in measured.items()):
                raise ValueError('reported measurements differ: '+variant)
            if saved['cli_sha256'] != record['builds'][variant]['cli_sha256']:
                raise ValueError('run CLI differs from prepared build')
            if variant in ('paired','weighted'):
                identity = require_identity(baseline_path(record,variant),out/variant,len(meta['ids']),meta['layers'],
                                            residuals=variant=='weighted')
                if identity != original['controls'][variant]:
                    raise ValueError('reported control identity differs')
            if variant != 'paired':
                require_identity(out/'paired',out/variant,len(meta['ids']),meta['layers'],('sequential',))
        if not original['complete'] or original['passed'] != original['variants']['combined']['passed']:
            raise ValueError('incomplete or incorrect aggregate result')
        result.update(complete=True,passed=original['passed'],report_sha256=digest(out/'report.json'))
    except Exception as error:
        result['error']=str(error);raise
    finally:
        (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','audit'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepared',type=Path)
    args=parser.parse_args()
    if args.command=='run' and args.prepared is None:parser.error('--prepared is required to run')
    raise SystemExit(run(args) if args.command=='run' else audit(args))
