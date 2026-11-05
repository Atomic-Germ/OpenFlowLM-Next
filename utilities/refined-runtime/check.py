"""Private full-depth refined-ax experiment; immutable standard export and CPU oracle."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


post = load('_refined_post', ROOT/'utilities/prefill-production/post_validate.py')
boundary = load('_refined_boundary', ROOT/'utilities/prefill-slice/boundaries.py')
ax = load('_refined_ax_builder', ROOT/'utilities/l3-refined-ax/build.py')
base, digest = post.base, post.digest
ROLES = ('control', 'candidate')
TOKENS, LAYERS = 257, 64


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sources():
    return post.sources() | {str(Path(p).resolve()): digest(p) for p in
                            (__file__, boundary.__file__, ax.__file__,
                             ROOT/'open_kernels/model/decode_bundle.py')}


def derivative_key(parent, private, role):
    if role not in ROLES: raise ValueError('invalid private role')
    return 'private-refined-ax:'+hashlib.sha256(json.dumps([parent, private, role]).encode()).hexdigest()


def bundle(parent, private, out, role):
    """Copy only runtime binaries. Give the composite a distinct, non-production key."""
    built = ax.verify(private)
    if built['role'] != role: raise ValueError('private build role mismatch')
    original = json.loads((parent/'manifest.json').read_text())
    spec = ax.load_spec(parent/'spec.json')
    base.load_bundle(parent, base.manifest(spec, max_ctx=original['max_ctx_default']))
    parent_proof = json.loads((parent/'toolchain.json').read_text())
    if digest(parent/'ax/insts.bin') != digest(private/'insts.bin'):
        raise ValueError('private ax changed runtime instruction ABI')
    selected = copy.deepcopy(original)
    selected['build_key'] = derivative_key(digest(parent/'manifest.json'), digest(private/'build.json'), role)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in sorted(parent_proof['sha256']):
        source = private/Path(name).name if name in ('ax/final.xclbin', 'ax/insts.bin') else parent/name
        target = out/name
        if not target.resolve().is_relative_to(out.resolve()): raise ValueError('unsafe bundle path')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); hashes[name] = digest(target)
    shutil.copyfile(parent/'spec.json', out/'spec.json')
    write(out/'manifest.json', selected)
    write(out/'toolchain.json', dict(build_key=selected['build_key'], spec_hash=selected['spec_hash'],
          sha256=hashes, private_experiment=True, model_acceptance=False, role=role,
          parent=str(parent), parent_manifest_sha256=digest(parent/'manifest.json'),
          parent_toolchain_sha256=digest(parent/'toolchain.json'),
          ax_build=str(private), ax_build_sha256=digest(private/'build.json')))
    base.load_bundle(out, base.manifest(spec, max_ctx=original['max_ctx_default']))


def verify_bundle(directory, parent, private, role):
    built = ax.verify(private)
    record = json.loads((directory/'toolchain.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    original = json.loads((parent/'manifest.json').read_text())
    expected = original | {'build_key': derivative_key(digest(parent/'manifest.json'), digest(private/'build.json'), role)}
    if manifest != expected or built['role'] != role: raise ValueError('private bundle semantics/role changed')
    parent_record = json.loads((parent/'toolchain.json').read_text())
    hashes = parent_record['sha256'] | {'ax/'+n: built['sha256'][n] for n in ('final.xclbin', 'insts.bin')}
    proof = dict(build_key=expected['build_key'], spec_hash=expected['spec_hash'], sha256=hashes,
                 private_experiment=True, model_acceptance=False, role=role, parent=str(parent),
                 parent_manifest_sha256=digest(parent/'manifest.json'),
                 parent_toolchain_sha256=digest(parent/'toolchain.json'),
                 ax_build=str(private), ax_build_sha256=digest(private/'build.json'))
    if record != proof or digest(directory/'spec.json') != digest(parent/'spec.json'):
        raise ValueError('private bundle provenance changed')
    spec = ax.load_spec(parent/'spec.json')
    base.load_bundle(directory, base.manifest(spec, max_ctx=manifest['max_ctx_default']))
    if digest(private/'insts.bin') != digest(parent/'ax/insts.bin'): raise ValueError('instruction ABI changed')


def exact_control(actual, expected, tokens=TOKENS, layers=LAYERS):
    names = set(post.old.runner.capture_names('sequential', tokens, layers))
    if set(actual) != names or set(expected) != names: raise ValueError('incomplete control inventory')
    checks = {name: actual[name] == expected[name] for name in sorted(names)}
    if not all(checks.values()): raise ValueError('fresh full-depth control differs from retained native run')
    return checks


def compare(control, candidate):
    result = dict(candidate_passed=candidate['passed'])
    for field, metric in (('residuals', 'maxrel'), ('logits', 'correlation')):
        rows, regressions = [], []
        before, after = control[field], candidate[field]
        key = lambda r: (r.get('layer'), r['position'])
        if not before or list(map(key, before)) != list(map(key, after)) or len(set(map(key, before))) != len(before):
            raise ValueError('comparison coverage differs')
        improved = regressed = tied = 0
        for a, b in zip(before, after):
            av, bv = a[metric], b[metric]
            # A nonzero error against a zero reference is an infinite error.
            av = float('inf') if av is None else av
            bv = float('inf') if bv is None else bv
            better = bv < av if field == 'residuals' else bv > av
            worse = bv > av if field == 'residuals' else bv < av
            improved += better; regressed += worse; tied += not (better or worse)
            row = dict(position=b['position'], control=a[metric], candidate=b[metric])
            if 'layer' in b: row['layer'] = b['layer']
            if worse: regressions.append(row)
            rows.append(b)
        worst = (max(rows, key=lambda r: float('inf') if r[metric] is None else r[metric])
                 if field == 'residuals' else min(rows, key=lambda r: r[metric]))
        result[field] = dict(total=len(rows), passed=sum(r['passed'] for r in rows),
                             improved=improved, regressed=regressed, tied=tied,
                             worst=worst, regressions=regressions)
    return result


def historical(record):
    directory = Path(record['baseline']); refs = Path(record['references'])
    saved = json.loads((directory/'report.json').read_text())
    audit = json.loads((directory/'audit.json').read_text())
    if not saved['complete'] or not audit['complete'] or audit['report_sha256'] != digest(directory/'report.json'):
        raise ValueError('historical control lacks complete bound audit')
    if audit['passed'] != saved['passed']: raise ValueError('historical numerical verdict differs')
    # Historical FAIL is deliberate, never a waiver of the candidate's gates.
    meta = boundary.validate_reference(refs, LAYERS, TOKENS)
    proof = json.loads((refs.parent/'reference-audit.json').read_text())
    if not proof['passed'] or proof['evidence'].get(str(refs/'reference.json')) != digest(refs/'reference.json'):
        raise ValueError('independent reference lacks bound audit')
    prepared = Path(saved['prepared'])
    if digest(prepared/'preparation.json') != saved['preparation_sha256']: raise ValueError('historical preparation changed')
    prep = json.loads((prepared/'preparation.json').read_text())
    if Path(prep['references'])/str(TOKENS) != refs: raise ValueError('historical reference path differs')
    base.verify(prep['inputs'])
    short = json.loads((Path(prep['baseline'])/'report.json').read_text())
    if short['kernels'] != record['parent']: raise ValueError('control parent differs from historical export')
    paths = [directory/'report.json', directory/'audit.json', prepared/'preparation.json',
             refs/'reference.json', refs.parent/'reference-audit.json',
             Path(prep['baseline'])/'report.json']
    inputs = {str(p): digest(p) for p in paths}
    oracle = [Path(meta['model'])/n for n in ('model.q4nx', 'config.json', 'tokenizer.json')]
    oracle += [ROOT/'utilities/prefill-slice/check.py']
    oracle += [ROOT/'open_kernels/model'/n for n in ('replica_qwen35.py','replica_prompt.py','q4nx.py','compare_decode.py')]
    inputs.update({str(p): meta['sources'][str(p)] for p in oracle})
    inputs.update(prep['inputs'])
    native = next(r['first'] for r in saved['cases'][str(TOKENS)]['routes'] if r['route'] == 'sequential')
    for name, h in native['outputs'].items(): inputs[str(directory/str(TOKENS)/'trace'/name)] = h
    base.verify(inputs)
    return meta, native, inputs


def context(out):
    record = json.loads((out/'preparation.json').read_text())
    if not record['complete'] or record['sources'] != sources(): raise ValueError('incomplete or changed preparation')
    base.verify(record['artifacts'])
    meta, native, inputs = historical(record)
    if inputs != record['inputs']: raise ValueError('historical input identity changed')
    spec = base.spec_from_model_dir(Path(meta['model']))
    parent = Path(record['parent'])
    base.load_bundle(parent, base.manifest(spec, max_ctx=meta['max_ctx']))
    for role in ROLES:
        verify_bundle(out/role, parent, Path(record[role+'_ax']), role)
    for kind in ('plain', 'trace'):
        build = Path(record[kind+'_build'])
        evidence = post.builder.verify(build, False, plain=kind == 'plain')
        if digest(build/'build-evidence.json') != record[kind+'_evidence'] or evidence['cli_sha256'] != record[kind+'_sha256']:
            raise ValueError('runtime build identity changed')
    return record, meta, native, base.layout(spec, max_ctx=meta['max_ctx'])


def prepare(args):
    out = args.out.resolve()
    if not out.is_relative_to(ROOT/'Models'): raise ValueError('private outputs must be under Models')
    out.mkdir(parents=True, exist_ok=False)
    record = dict(complete=False, model_acceptance=False, sources=sources())
    try:
        for name in ('baseline', 'references', 'parent', 'control_ax', 'candidate_ax', 'plain_build', 'trace_build'):
            record[name] = str(getattr(args, name).resolve())
        _, _, record['inputs'] = historical(record)
        for kind in ('plain', 'trace'):
            build = Path(record[kind+'_build'])
            record[kind+'_sha256'] = post.builder.verify(build, False, plain=kind == 'plain')['cli_sha256']
            record[kind+'_evidence'] = digest(build/'build-evidence.json')
        for role in ROLES: bundle(Path(record['parent']), Path(record[role+'_ax']), out/role, role)
        paths = [p for role in ROLES for p in (out/role).rglob('*') if p.is_file()]
        paths += [Path(record[role+'_ax'])/'build.json' for role in ROLES]
        record['artifacts'] = {str(p):digest(p) for p in paths}
        record['complete'] = True
        write(out/'preparation.json', record); context(out)
    except Exception as error:
        record.update(complete=False, error=str(error)); raise
    finally: write(out/'preparation.json', record)


def measure(out, prepared):
    record, meta, native, offsets = context(prepared)
    refs = Path(record['references']); scores = {}
    for role in ROLES:
        target = out/role; view = meta | {'kernels': str(prepared/role)}
        for kind in ('plain', 'trace'):
            if digest(target/kind/'ids.txt') != meta['ids_sha256']: raise ValueError('prompt changed')
            post.inventory(target/kind, 'sequential', TOKENS, LAYERS, kind == 'trace')
        plain = post.plain_score(target/'plain', refs, view, offsets, 'sequential')
        trace = post.old.runner.score_route(target/'trace', refs, view, offsets, 'sequential')
        identity = post.plain_identity(trace['outputs'], plain['outputs'], 'sequential', TOKENS, LAYERS)
        scores[role] = dict(plain=plain, trace=trace, identity=identity)
    controls = exact_control(scores['control']['trace']['outputs'], native['outputs'])
    return dict(scores=scores, control_identity=controls,
                comparison=compare(scores['control']['trace'], scores['candidate']['trace']))


def run(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, candidate_passed=False, model_acceptance=False, sources=sources(), commands=[])
    try:
        prepared = args.prepared.resolve(); report['prepared'] = str(prepared)
        report['preparation_sha256'] = digest(prepared/'preparation.json')
        record, meta, _, offsets = context(prepared)
        env = post.old.smoke.capture.environment()
        env.update(XILINX_XRT='/opt/xilinx/xrt', LD_LIBRARY_PATH='/opt/xilinx/xrt/lib', OPENBLAS_NUM_THREADS='1')
        with post.old.smoke.capture.device_lock():
            for role in ROLES:
                view = meta | {'kernels': str(prepared/role)}
                for kind in ('plain', 'trace'):
                    base.memory_preflight(LAYERS); post.old.runner.device_available()
                    target = out/role/kind; target.mkdir(parents=True)
                    shutil.copyfile(Path(record['references'])/'ids.txt', target/'ids.txt')
                    cmd = post.command(Path(record[kind+'_build']), target, view, offsets, 'sequential')
                    report['commands'].append(cmd); print('running', role, kind, flush=True)
                    with (target/'sequential.log').open('w') as log:
                        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=3600)
                if role == 'control':
                    names = post.old.runner.capture_names('sequential', TOKENS, LAYERS)
                    _, _, native, _ = context(prepared)
                    exact_control({n:digest(out/role/'trace'/n) for n in names}, native['outputs'])
        report.update(measure(out, prepared))
        if digest(prepared/'preparation.json') != report['preparation_sha256']: raise ValueError('preparation changed')
        report.update(complete=True, candidate_passed=report['comparison']['candidate_passed'])
    except Exception as error: report['error'] = str(error); raise
    finally: write(out/'report.json', report)
    return 0 if report['candidate_passed'] else 1


def audit(args):
    out = args.out.resolve(); proof = dict(complete=False, candidate_passed=False, model_acceptance=False)
    try:
        report = json.loads((out/'report.json').read_text()); prepared = Path(report['prepared'])
        if not report['complete'] or report['sources'] != sources() or report['model_acceptance'] is not False:
            raise ValueError('incomplete run or changed validator')
        if digest(prepared/'preparation.json') != report['preparation_sha256']: raise ValueError('preparation changed')
        actual = measure(out, prepared)
        if any(report[k] != v for k,v in actual.items()): raise ValueError('measurements changed')
        if report['candidate_passed'] != actual['comparison']['candidate_passed']: raise ValueError('incorrect verdict')
        proof.update(complete=True, candidate_passed=report['candidate_passed'], report_sha256=digest(out/'report.json'))
    except Exception as error: proof['error'] = str(error); raise
    finally: write(out/'audit.json', proof)
    return 0 if proof['candidate_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    for action, flags in [('prepare', ('baseline','references','parent','control-ax','candidate-ax','plain-build','trace-build','out')),
                          ('run', ('prepared','out')), ('audit', ('out',))]:
        command = sub.add_parser(action)
        for flag in flags: command.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args(); raise SystemExit(globals()[args.action](args) or 0)
