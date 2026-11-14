"""CPU causal-prefix diagnostics on retained native residuals; no model acceptance."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
s = importlib.util.spec_from_file_location('_prefix_sublayers', ROOT/'utilities/prefill-weighted-sublayers/check.py')
sub = importlib.util.module_from_spec(s)
s.loader.exec_module(sub)
base, trace = sub.base, sub.trace
ROUTES = ('sequential', 'hybrid')
COMPONENTS = ('propagated', 'ideal_history_gap', 'total')


def replay(model, spec, layer, inputs, positions):
    """Reset ideal state per call, advance every causal row, evaluate selected FFNs."""
    xs = np.asarray(inputs)
    if (xs.ndim != 2 or not np.isfinite(xs).all() or not positions
            or any(not isinstance(p, (int, np.integer)) or isinstance(p, bool)
                   or p < 0 or p >= len(xs) for p in positions)
            or len(set(positions)) != len(positions)):
        raise ValueError('invalid causal prefix or target positions')
    conv = np.zeros((spec.conv_kernel-1, spec.lin_qkv_dim))
    state = np.zeros((spec.lin_value_heads, spec.lin_value_dim, spec.lin_value_dim))
    k = np.zeros((0, spec.num_kv_heads, spec.head_dim))
    v = k.copy()
    result = {}
    for p in range(max(positions)+1):
        x = xs[p].copy()
        if spec.layer_types[layer] == 'full_attention':
            out, k, v = base.oracle.attn_decode(model, spec, layer, x, k, v, p)
        else:
            out, conv, state = base.oracle.linear_decode(model, spec, layer, x, conv, state)
        if p in positions:
            res = x + out
            result[p] = res + base.oracle.ffn(model, spec, layer, res)
    return result


def error_score(delta, reference):
    scale = float(np.max(np.abs(reference)))
    error = float(np.max(np.abs(delta)))
    ratio = error/scale if scale else (0.0 if error == 0 else None)
    return dict(maxabs=error, reference_maxabs=scale, maxrel=ratio,
                passed=ratio is not None and ratio < base.RESIDUAL_MAXREL)


def decompose(native, ideal, reference):
    values = [np.asarray(x, dtype=np.float64) for x in (native, ideal, reference)]
    if (any(x.ndim != 1 or not x.size or not np.isfinite(x).all() for x in values)
            or any(x.shape != values[0].shape for x in values)):
        raise ValueError('requires equal finite nonempty output vectors')
    native, ideal, reference = values
    vectors = dict(propagated=ideal-reference, ideal_history_gap=native-ideal,
                   total=native-reference)
    closure = vectors['propagated'] + vectors['ideal_history_gap'] - vectors['total']
    bound = 8*np.finfo(np.float64).eps*(np.abs(native)+np.abs(ideal)+np.abs(reference))
    if np.any(np.abs(closure) > bound):
        raise ValueError('signed diagnostic vectors do not close')
    return {k: error_score(v, reference) for k, v in vectors.items()}, vectors


def summarize(rows, layers, positions):
    required = {(r,l,p) for r in ROUTES for l in range(layers) for p in positions}
    index = {(c['route'],c['layer'],c['position']): c for c in rows}
    if not required or len(rows) != len(index) or set(index) != required:
        raise ValueError('incomplete or duplicate scan coverage')
    result = {}
    for route in ROUTES:
        group = [index[route,l,p] for l in range(layers) for p in positions]
        first = {str(p): next((index[route,l,p] for l in range(layers)
                              if not index[route,l,p]['metrics']['total']['passed']), None)
                 for p in positions}
        ranked = {key: sorted(group, key=lambda c: (
            float('inf') if c['metrics'][key]['maxrel'] is None else c['metrics'][key]['maxrel']),
            reverse=True)[:10] for key in COMPONENTS}
        result[route] = dict(total=len(group), passed=sum(c['metrics']['total']['passed'] for c in group),
                             first_failure_by_position=first, largest=ranked)
    return result


def validate(run, baseline, build):
    audit_path = run/'sublayer-audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['capture_valid'] or not audit['conditioned_ffn_passed']:
        raise ValueError('invalid retained sublayer audit')
    base.verify(audit['evidence'])
    base.verify(audit['cpu_outputs'])
    sub.inventory(run)
    base.verify({str(run/name): h for name,h in audit['capture_hashes'].items()})
    if set(audit['capture_hashes']) != sub.expected_captures():
        raise ValueError('incomplete sublayer audit capture coverage')
    b = sub.weighted.build(build, 'weighted-sublayer')
    candidate = sub.weighted.build(baseline.parent, 'candidate')
    prior_path = baseline.parent.parent/'comparison.json'
    prior = json.loads(prior_path.read_text())
    if not prior['capture_valid']:
        raise ValueError('invalid weighted baseline')
    base.verify(prior['evidence'])
    meta, report = trace.paired.validate_run(run)
    baseline_meta = json.loads((baseline/'reference.json').read_text())
    baseline_report = json.loads((baseline/'report.json').read_text())
    if (meta != baseline_meta or report['cli_sha256'] != b['cli_sha256']
            or baseline_report['cli_sha256'] != candidate['cli_sha256']):
        raise ValueError('reference or native build mismatch')
    for path in (build/'build-evidence.json', baseline.parent/'build-evidence.json',
                 prior_path, run/'reference.json', run/'report.json'):
        if audit['evidence'].get(str(path)) != base.digest(path):
            raise ValueError('scan input not bound to retained sublayer audit')
    identities = trace.require_identity(baseline, run, len(meta['ids']))
    sub.residual_inventory(run, meta['layers'])
    expected = {f'{r}_res{l}.bin' for r in trace.ROUTES for l in range(meta['layers'])}
    hashes = prior['trace_hashes']['candidate']
    if set(hashes) != expected:
        raise ValueError('incomplete baseline residual coverage')
    base.verify({str(baseline/name): h for name,h in hashes.items()})
    base.verify({str(run/name): h for name,h in hashes.items()})
    if any(hashes[f'hybrid_res{l}.bin'] != hashes[f'block_major_res{l}.bin']
           for l in range(meta['layers'])):
        raise ValueError('hybrid residual schedules differ')
    return meta, dict(sublayer_audit_sha256=base.digest(audit_path),
                     residual_hashes=hashes, logit_identity=identities,
                     hybrid_residual_identity=meta['layers'])


def run(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, model_acceptance=False, cases=[])
    try:
        directory = a.run.resolve()
        # Fail early with a persisted diagnostic, even before optional arguments.
        (directory/'sublayer-audit.json').read_text()
        meta, evidence = validate(directory, a.baseline.resolve(), a.build.resolve())
        layers, positions = a.layers, tuple(a.positions)
        if (not 0 < layers <= meta['layers'] or meta['hidden'] != 5120 or meta['layers'] != 64
                or len(meta['ids']) != 20 or not positions or len(set(positions)) != len(positions)
                or any(p < 0 or p >= len(meta['ids']) for p in positions)):
            raise ValueError('invalid retained short-prompt scan coverage')
        spec = base.spec_from_model_dir(Path(meta['model']))
        model = base.Q4NX(Path(meta['model'])/'model.q4nx')
        model.hidden = spec.hidden
        model.native_q8 = lambda name: False
        plain, cache = model.matmul_w, {}
        def cached(name, o, i):
            if name not in cache:
                cache[name] = plain(name, o, i)
            return cache[name]
        model.matmul_w = cached
        n, hidden, stop = len(meta['ids']), meta['hidden'], max(positions)+1
        embeddings = np.array([model.embed(t,hidden).astype(np.float32) for t in meta['ids'][:stop]])
        artifacts = {}
        result.update(layers=layers, positions=positions, evidence=evidence,
                      scope='ideal causal state driven by native prefixes; gap includes history and local arithmetic',
                      reference_reconstructions=0)
        for layer in range(layers):
            cache.clear()
            inputs = embeddings if layer == 0 else trace.read_trace(
                directory/f'ref_res{layer-1}.bin',n,hidden)[:stop].astype(np.float32)
            reference = trace.read_trace(directory/f'ref_res{layer}.bin',n,hidden)
            ideal_reference = replay(model,spec,layer,inputs,positions)
            for p in positions:
                if ideal_reference[p].astype('<f4').tobytes() != reference[p].astype('<f4').tobytes():
                    raise ValueError(f'independent reference reconstruction differs at L{layer}/P{p}')
                result['reference_reconstructions'] += 1
            for route in ROUTES:
                xs = embeddings if layer == 0 else trace.read_trace(
                    directory/f'{route}_res{layer-1}.bin',n+1,hidden)[:stop].astype(np.float32)
                native = trace.read_trace(directory/f'{route}_res{layer}.bin',n+1,hidden)
                ideal = replay(model,spec,layer,xs,positions)
                for p in positions:
                    metrics, vectors = decompose(native[p],ideal[p],reference[p])
                    path = out/f'{route}_L{layer}_P{p}.npz'
                    np.savez(path,**vectors,native=native[p],ideal=ideal[p],reference=reference[p])
                    artifacts[path.name] = base.digest(path)
                    case = dict(route=route,layer=layer,position=p,metrics=metrics,
                                input=error_score(xs[p].astype(np.float64)-inputs[p],inputs[p]))
                    result['cases'].append(case)
            print(json.dumps(dict(layer=layer, cases=result['cases'][-len(positions)*len(ROUTES):])), flush=True)
        files = [Path(__file__),Path(sub.__file__),Path(sub.weighted.__file__),Path(trace.__file__),
                 Path(trace.paired.__file__),Path(base.__file__),Path(base.oracle.__file__),
                 directory/'reference.json',directory/'report.json',directory/'sublayer-audit.json']
        result.update(complete=True,summary=summarize(result['cases'],layers,positions),artifacts=artifacts,
                      sources={str(p.resolve()):base.digest(p) for p in files})
    except Exception as e:
        result.update(complete=False,error=str(e))
        raise
    finally:
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','baseline','build','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--layers',type=int,default=56)
    parser.add_argument('--positions',type=int,nargs='+',default=[7,17])
    raise SystemExit(run(parser.parse_args()))
