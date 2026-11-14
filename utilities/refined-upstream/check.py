"""Causal scanner entry point and explicit retained-evidence audit after a merge.

scan.py preserves the original scanner bytewise. Retained verification checks
its historical source alias, not the freshness of old compiler inputs.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np

HERE = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location('_retained_upstream_scan', HERE/'scan.py')
scan = importlib.util.module_from_spec(loader); loader.loader.exec_module(scan)
# Preserve the public API used by the existing contract tests and later tools.
for name in ('ROOT', 'replay', 'sublayers', 'drift', 'base', 'digest', 'write',
             'ROLES', 'POSITIONS', 'LAYERS', 'COMPONENTS', 'DOWNSTREAM_NODES',
             'coverage', 'artifact_names', 'reference_identity', 'downstream_identity',
             'summarize', 'store', 'validate', 'run', 'audit'):
    globals()[name] = getattr(scan, name)


def retained_sources(expected, original=None, snapshot=None):
    original = (original or HERE/'check.py').resolve()
    snapshot = (snapshot or HERE/'scan.py').resolve()
    if str(original) not in expected: raise ValueError('missing historical scanner source')
    mapping = {}
    for name, sha in expected.items():
        path = snapshot if Path(name).resolve() == original else Path(name)
        if digest(path) != sha: raise ValueError('changed retained source: '+name)
        mapping[name] = str(path)
    return mapping


def retained_verdict(report, audit, sha):
    verdict = all(r['metrics']['total']['passed'] for r in report['cases'] if r['role'] == 'candidate')
    if (report.get('complete') is not True or audit.get('complete') is not True or
            report.get('model_acceptance') is not False or audit.get('model_acceptance') is not False or
            report.get('numerical_passed') is not verdict or audit.get('numerical_passed') is not verdict or
            audit.get('report_sha256') != sha or audit.get('cases') != 384 or
            audit.get('reference_reconstructions') != 192 or audit.get('downstream_controls') != 3):
        raise ValueError('incomplete or inconsistent retained causal audit')


def retained_case(arrays, row):
    nodes = ('native', 'ideal_native', 'mixed', 'ideal_reference', 'reference')
    expected = {*nodes, 'native_input', 'reference_input', *COMPONENTS}
    if set(arrays) != expected: raise ValueError('incomplete retained attribution arrays')
    if any(a.ndim != 1 or a.shape != arrays['reference'].shape or not a.size or
           not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('invalid retained attribution vectors')
    metrics, effects = drift.decompose(*(arrays[k] for k in nodes))
    for key, value in effects.items():
        if arrays[key].dtype != value.dtype or arrays[key].tobytes() != value.tobytes():
            raise ValueError('retained signed effect differs: '+key)
    incoming = drift.scan.error_score(arrays['native_input'].astype(np.float64)-arrays['reference_input'],
                                      arrays['reference_input'])
    if metrics != row['metrics'] or incoming != row['input']: raise ValueError('retained metrics changed')


def retained(args):
    directory, out = args.out.resolve(), args.scan.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    proof = dict(complete=False, model_acceptance=False, numerical_passed=False,
                 scope='Verify retained causal audit and recompute signed arrays/metrics; no fresh causal replay, build or NPU execution',
                 validator_sha256=digest(__file__), scanner_sha256=digest(HERE/'scan.py'))
    try:
        report = json.loads((out/'report.json').read_text())
        audit = json.loads((out/'audit.json').read_text())
        retained_verdict(report, audit, digest(out/'report.json')); coverage(report['cases'])
        mapping = retained_sources(report['sources'])
        # These hashes bind native captures, CPU references and the downstream
        # report/audit. Metadata remains historical; old builds are not relabelled.
        base.verify(report['inputs'])
        if set(report['artifacts']) != artifact_names() or {p.name for p in out.glob('*.npz')} != artifact_names():
            raise ValueError('incomplete or extra retained inventory')
        base.verify({str(out/n):h for n,h in report['artifacts'].items()})
        baseline = Path(report['baseline'])
        native_report = json.loads((baseline/'report.json').read_text())
        prepared = Path(native_report['prepared'])
        if digest(prepared/'preparation.json') != native_report['preparation_sha256']:
            raise ValueError('native preparation changed')
        prep = json.loads((prepared/'preparation.json').read_text())
        refs = Path(prep['references'])
        meta = scan.replay.runtime.boundary.validate_reference(refs, 64, 257)
        # Bind weights and oracle arithmetic independently of obsolete build keys.
        model = Path(meta['model'])
        oracle = [model/n for n in ('model.q4nx','config.json','tokenizer.json')]
        oracle += [ROOT/'utilities/prefill-slice/check.py']
        oracle += [ROOT/'open_kernels/model'/n for n in ('replica_qwen35.py','replica_prompt.py','q4nx.py','compare_decode.py')]
        oracle_hashes = {str(p):meta['sources'][str(p)] for p in oracle}; base.verify(oracle_hashes)
        geometry = base.spec_from_model_dir(model)
        reconstructions = controls = 0
        weights = base.Q4NX(model/'model.q4nx')
        try:
            embeddings = {p:weights.embed(meta['ids'][p], geometry.hidden).astype('<f4') for p in POSITIONS}
            for row in report['cases']:
                l, role, p = row['layer'], row['role'], row['position']
                name = f'{role}_L{l}_P{p}.npz'
                with np.load(out/name, allow_pickle=False) as f: arrays = {k:f[k] for k in f.files}
                retained_case(arrays, row)
                expected_y = replay.trace(refs/f'ref_res{l}.bin',257,geometry.hidden)[p]
                reference_identity(arrays['ideal_reference'], expected_y)
                expected_x = embeddings[p] if l == 0 else replay.trace(refs/f'ref_res{l-1}.bin',257,geometry.hidden)[p]
                native_dir = baseline/role/'trace'
                native_x = embeddings[p] if l == 0 else replay.trace(native_dir/f'sequential_res{l-1}.bin',258,geometry.hidden)[p]
                native_y = replay.trace(native_dir/f'sequential_res{l}.bin',258,geometry.hidden)[p]
                for key, value in [('reference',expected_y.astype(np.float64)),('reference_input',expected_x),
                                   ('native_input',native_x),('native',native_y.astype(np.float64))]:
                    if arrays[key].dtype != value.dtype or arrays[key].tobytes() != value.tobytes():
                        raise ValueError('retained node differs from pinned native/oracle data: '+key)
                expected_control = role == 'candidate' and replay.TARGETS.get(l) == p
                if row['kind'] != geometry.layer_types[l] or row['reference_exact'] is not True or row['downstream_control'] != expected_control:
                    raise ValueError('incorrect retained case metadata')
                if expected_control:
                    downstream_identity(Path(report['downstream'])/f'L{l}_P{p}.npz',arrays); controls += 1
                reconstructions += role == 'control'
        finally: weights.mm.close(); weights.f.close()
        if reconstructions != 192 or controls != 3 or summarize(report['cases']) != report['summary']:
            raise ValueError('retained coverage or summary changed')
        retained_sources(report['sources']); base.verify(report['inputs'])
        proof.update(complete=True, numerical_passed=report['numerical_passed'], cases=384,
                     reference_rows=192, downstream_controls=3, source_mapping=mapping,
                     oracle_hashes=oracle_hashes, report_sha256=digest(out/'report.json'),
                     causal_audit_sha256=digest(out/'audit.json'), inputs=len(report['inputs']), artifacts=384)
    except Exception as error: proof['error']=str(error); raise
    finally: write(directory/'audit.json',proof)
    return 0 if proof['numerical_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action, flags in (('run', ('out','baseline','downstream')), ('audit', ('out',)), ('retained', ('scan','out'))):
        command = sub.add_parser(action)
        for flag in flags: command.add_argument('--'+flag,type=Path,required=True)
    args = parser.parse_args(); raise SystemExit(globals()[args.action](args))
