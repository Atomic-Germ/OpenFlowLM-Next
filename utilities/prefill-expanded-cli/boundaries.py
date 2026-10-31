"""Validate expanded 255/256/257-token prefill against fresh paired controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

from importlib.util import spec_from_file_location, module_from_spec


def load(name, path):
    s = spec_from_file_location(name, path)
    m = module_from_spec(s); s.loader.exec_module(m)
    return m


HERE = Path(__file__).resolve().parent
runner = load('_expanded_boundary_runner', HERE/'run.py')
base, digest = runner.base, runner.digest
boundary = load('_expanded_boundary_reference', HERE.parent/'prefill-slice/boundaries.py')
history = runner.original.prep.history
LENGTHS = (255, 256, 257)
VARIANTS = ('paired', 'expanded')


def check_cases(cases):
    if set(cases) != set(LENGTHS):
        raise ValueError('requires exactly 255/256/257 reference cases')
    for n, meta in cases.items():
        if meta['layers'] != 64 or meta['hidden'] != 5120 or len(meta['ids']) != n:
            raise ValueError('incorrect boundary depth/width/length')
        boundary.compatible(cases[255], meta)
        if cases[255]['ids'] != meta['ids'][:255]:
            raise ValueError('boundary prompt IDs differ')
    if cases[256]['ids'] != cases[257]['ids'][:256]:
        raise ValueError('boundary prompt IDs differ')


def aggregate(cases, prefixes):
    if set(cases) != {str(n) for n in LENGTHS} or not prefixes:
        raise ValueError('incomplete boundary aggregate')
    return all(c['passed'] for c in cases.values()) and all(prefixes.values())


def evidence():
    return runner.evidence() | {str(Path(p).resolve()): digest(p) for p in
                               (__file__, boundary.__file__, history.__file__)}


def short_context(directory):
    saved, record, meta, offsets = runner.check_run(directory)
    audit = json.loads((directory/'audit.json').read_text())
    if (not saved['passed'] or not audit['complete'] or not audit['passed'] or
            audit['report_sha256'] != digest(directory/'report.json')):
        raise ValueError('requires accepted and independently audited short run')
    return record, meta, offsets


def verify_build(record, variant):
    b = record['builds'][variant]; path = Path(b['directory'])
    check = runner.original.prep.verify_build if variant == 'paired' else runner.prep.verify_build
    if check(path, variant)['cli_sha256'] != b['cli_sha256']:
        raise ValueError('prepared boundary build changed')
    return path


def references(directory):
    cases = {n: boundary.validate_reference(directory/str(n), 64, n) for n in LENGTHS}
    check_cases(cases)
    prefixes = {f'{n}/{n+1}': boundary.compare_prefix(directory/str(n), cases[n],
                directory/str(n+1), cases[n+1]) for n in LENGTHS[:-1]}
    return cases, prefixes


def prepare(a):
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    r = dict(complete=False, model_acceptance=False, inputs={}, history={}, sources=evidence())
    try:
        short = a.baseline.resolve(); old, sm, _ = short_context(short)
        control = json.loads((Path(old['baseline_prepared'])/'preparation.json').read_text())
        r['builds'] = dict(paired=control['builds']['paired'], expanded=old['builds']['expanded'])
        for variant in VARIANTS: verify_build(r, variant)
        root = a.references.resolve(); cases, prefixes = references(root)
        # Short and long prompts differ, but model/export/reference geometry must not.
        for key in boundary.FIELDS:
            if key != 'message' and sm[key] != cases[255][key]:
                raise ValueError('long reference differs from accepted short geometry: '+key)
        def pin(path): r['inputs'][str(path)] = digest(path)
        for path in (short/'report.json', short/'audit.json', Path(old['baseline_prepared'])/'preparation.json'):
            pin(path)
        old_build = runner.prep.ROOT/'Models/qwen38-27b/prefill-paired-input/paired/build-evidence.json'
        historic_build = json.loads(old_build.read_text())
        if not historic_build['complete'] or historic_build['variant'] != 'paired':
            raise ValueError('incomplete historical paired build')
        graphs = [history.history_graph(old_build)]
        for n, meta in cases.items():
            d = root/str(n); graphs.append(history.history_graph(d/'reference.json'))
            report = json.loads((d/'report.json').read_text())
            if (report['reference_sha256'] != digest(d/'reference.json') or
                    report['cli_sha256'] != historic_build['cli_sha256'] or
                    [row['route'] for row in report['routes']] != list(runner.ROUTES)):
                raise ValueError('unbound historical long-prefix control')
            for name, sha in meta['references'].items(): r['inputs'][str(d/name)] = sha
            r['inputs'][str(d/'ids.txt')] = meta['ids_sha256']; pin(d/'report.json')
            for row in report['routes']:
                route = row['route']
                if set(row['outputs']) != set(base.expected_dumps(route,n)+[route+'-act.bin']):
                    raise ValueError('incomplete historical native captures')
                for name, sha in row['outputs'].items(): r['inputs'][str(d/name)] = sha
                r['inputs'][str(d/(route+'.log'))] = row['log_sha256']
        for graph in graphs:
            for name, value in graph['files'].items():
                if name in r['history'] and r['history'][name] != value:
                    raise ValueError('conflicting historical provenance')
                r['history'][name] = value
        base.verify(r['inputs']); base.verify(r['sources'])
        r.update(complete=True, baseline=str(short), reference_root=str(root),
                 reference_prefixes=prefixes, reference_files=sum(len(m['references']) for m in cases.values()),
                 scope='immutable independent long-prefix references and verified private builds; no native execution')
    except Exception as error: r['error'] = str(error); raise
    finally: (out/'preparation.json').write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')
    return 0


def verify_preparation(directory):
    r = json.loads((directory/'preparation.json').read_text())
    if not r['complete'] or set(r['builds']) != set(VARIANTS) or r['sources'] != evidence():
        raise ValueError('incomplete or changed boundary preparation')
    base.verify(r['inputs']); base.verify(r['sources'])
    for name, value in r['history'].items():
        if history.verify_historical(name, value['sha256']) != value:
            raise ValueError('historical/current source identity changed')
    _, sm, offsets = short_context(Path(r['baseline']))
    for variant in VARIANTS: verify_build(r, variant)
    cases, prefixes = references(Path(r['reference_root']))
    if prefixes != r['reference_prefixes'] or sum(len(m['references']) for m in cases.values()) != r['reference_files']:
        raise ValueError('reference inventory or prefixes changed')
    for key in boundary.FIELDS:
        if key != 'message' and sm[key] != cases[255][key]:
            raise ValueError('reference geometry changed')
    return r, cases, offsets


def native_prefixes(directory, cases):
    pairs = {}
    for n, longer in zip(LENGTHS, LENGTHS[1:]):
        a, b = directory/str(n), directory/str(longer)
        h, layers = cases[n]['hidden'], cases[n]['layers']
        for route in runner.ROUTES:
            for p in range(n):
                name = f'{route}_p{p}.bin'
                pairs[f'{n}/{longer}/{name}'] = digest(a/name) == digest(b/name)
            for layer in range(layers):
                name = f'{route}_res{layer}.bin'
                x = base.read(a/name, (n+1)*h).astype('<f4')[:n*h]
                y = base.read(b/name, (longer+1)*h).astype('<f4')[:n*h]
                pairs[f'{n}/{longer}/{name}'] = x.tobytes() == y.tobytes()
    return pairs


def control_identity(variant, out, record, n, meta):
    if variant == 'expanded':
        return runner.require_identity(out/'paired'/str(n), out/'expanded'/str(n), n, meta['layers'], ('sequential',))
    before = Path(record['reference_root'])/str(n); after = out/'paired'/str(n)
    names = [name for route in runner.ROUTES for name in base.expected_dumps(route,n)+[route+'-act.bin']]
    result = {name: digest(before/name) == digest(after/name) for name in names}
    if not all(result.values()):
        raise ValueError('historical paired control capture mismatch: '+', '.join(k for k,v in result.items() if not v))
    return result


def capture_case(variant, out, record, n, meta, offsets):
    directory = out/variant/str(n); directory.mkdir(parents=True)
    refs = Path(record['reference_root'])/str(n)
    (directory/'ids.txt').write_bytes((refs/'ids.txt').read_bytes())
    bp = verify_build(record, variant); commands = []
    for route, extra in runner.ROUTES.items():
        base.memory_preflight(64); runner.original.device_available()
        cmd = base.native_command(bp/'build/open_qwen36_cli',directory,meta,offsets,route,extra)
        commands.append(cmd); print('running',variant,n,route,flush=True)
        with (directory/(route+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=runner.environment(),check=True,timeout=1800)
        score = runner.original.score_route(directory,refs,meta,offsets,route)
        print(variant,n,route,'logits',sum(x['passed'] for x in score['logits']), '/', n+1,
              'residuals',sum(x['passed'] for x in score['residuals']), '/', (n+1)*64,flush=True)
    report = runner.original.audit_variant(directory,refs,meta,offsets)
    report.update(commands=commands,cli_sha256=record['builds'][variant]['cli_sha256'])
    return report


def run(a):
    out = a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    r = dict(complete=False,passed=False,model_acceptance=False,variants={},controls={},prefixes={})
    try:
        prepared = a.prepared.resolve(); record, cases, offsets = verify_preparation(prepared)
        r.update(prepared=str(prepared),preparation_sha256=digest(prepared/'preparation.json'),sources=evidence())
        with runner.device_lock():
            for variant in VARIANTS:
                r['variants'][variant] = {}; r['controls'][variant] = {}
                for n in LENGTHS:
                    base.memory_preflight(64)
                    r['variants'][variant][str(n)] = capture_case(variant,out,record,n,cases[n],offsets)
                    r['controls'][variant][str(n)] = control_identity(variant,out,record,n,cases[n])
                    (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
                r['prefixes'][variant] = native_prefixes(out/variant,cases)
                if variant == 'paired' and not all(r['prefixes'][variant].values()):
                    raise ValueError('fresh paired control causal prefixes differ')
        base.verify(r['sources'])
        r.update(complete=True,passed=aggregate(r['variants']['expanded'],r['prefixes']['expanded']))
    except Exception as error: r.update(complete=False,passed=False,error=str(error)); raise
    finally: (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    return 0 if r['passed'] else 1


def audit(a):
    out = a.out.resolve(); result = dict(complete=False,passed=False,model_acceptance=False)
    try:
        saved = json.loads((out/'report.json').read_text())
        if (not saved['complete'] or saved['sources'] != evidence() or
                any(set(saved[k]) != set(VARIANTS) for k in ('variants','controls','prefixes'))):
            raise ValueError('incomplete run or changed validator')
        prepared = Path(saved['prepared'])
        if saved['preparation_sha256'] != digest(prepared/'preparation.json'):
            raise ValueError('preparation changed')
        record, cases, offsets = verify_preparation(prepared)
        for variant in VARIANTS:
            if set(saved['variants'][variant]) != {str(n) for n in LENGTHS} or set(saved['controls'][variant]) != {str(n) for n in LENGTHS}:
                raise ValueError('incomplete boundary coverage')
            for n in LENGTHS:
                directory = out/variant/str(n); meta = cases[n]
                got = runner.original.audit_variant(directory,Path(record['reference_root'])/str(n),meta,offsets)
                bp = Path(record['builds'][variant]['directory'])
                got.update(commands=[base.native_command(bp/'build/open_qwen36_cli',directory,meta,offsets,route,extra)
                           for route,extra in runner.ROUTES.items()],cli_sha256=record['builds'][variant]['cli_sha256'])
                if got != saved['variants'][variant][str(n)] or control_identity(variant,out,record,n,meta) != saved['controls'][variant][str(n)]:
                    raise ValueError('measurements, commands or controls differ')
            if native_prefixes(out/variant,cases) != saved['prefixes'][variant]:
                raise ValueError('reported causal prefixes differ')
        passed = aggregate(saved['variants']['expanded'],saved['prefixes']['expanded'])
        if saved['passed'] != passed or saved['model_acceptance'] is not False or not all(saved['prefixes']['paired'].values()):
            raise ValueError('incorrect aggregate result')
        result.update(complete=True,passed=passed,report_sha256=digest(out/'report.json'))
    except Exception as error: result['error'] = str(error); raise
    finally: (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','audit'])
    p.add_argument('--out',type=Path,required=True)
    for flag in ('baseline','references','prepared'): p.add_argument('--'+flag,type=Path)
    a = p.parse_args()
    needed = {'prepare':('baseline','references'),'run':('prepared',),'audit':()}[a.command]
    for flag in needed:
        if getattr(a,flag) is None: p.error('--'+flag+' required')
    raise SystemExit(dict(prepare=prepare,run=run,audit=audit)[a.command](a))
