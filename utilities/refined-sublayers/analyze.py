"""Conditioned DeltaNet/FFN and signed causal attribution; not model acceptance."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('_refined_sublayer_replay', Path(__file__).with_name('replay.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
boundary = replay.load('_refined_sublayer_boundaries', replay.ROOT/'utilities/prefill-sequential-drift/l0_boundaries.py')
attribution = replay.load('_refined_sublayer_attribution', replay.ROOT/'utilities/prefill-sequential-drift/post_attribution.py')
base, digest, write = replay.base, replay.digest, replay.write
drift, di, ds = replay.native.drift, boundary.di, boundary.ds
NODES = ('reference', 'oracle_reference', 'current_input', 'input_history', 'fp64',
         'recurrence', 'z', 'post', 'projection', 'residual_add', 'ffn_norm', 'ffn_activation', 'down', 'native')
PARTS = ('reference_rounding', 'current_input', 'input_history', 'reference_math',
         'recurrence', 'z', 'post', 'projection', 'residual_add', 'ffn_norm', 'ffn_activation', 'down', 'final_add')


def sources():
    return replay.sources() | boundary.sources() | attribution.sources() | {str(Path(__file__).resolve()): digest(__file__)}


def region(data, offsets, name, count):
    off = getattr(offsets, name)
    if len(data) != offsets.A_BYTES or off < 0 or off % 4 or count <= 0 or off+4*count > len(data):
        raise ValueError('invalid FP32 scratch region')
    return ds.finite(np.frombuffer(data, '<f4', count=count, offset=off)).astype(np.float64)


def decompose(nodes):
    if set(nodes) != set(NODES):
        raise ValueError('incomplete attribution nodes')
    values = [np.asarray(nodes[n], np.float64) for n in NODES]
    if any(v.ndim != 1 or not v.size or v.shape != values[0].shape or not np.isfinite(v).all() for v in values):
        raise ValueError('requires equal complete finite attribution vectors')
    vectors = {part: b-a for part,a,b in zip(PARTS, values, values[1:])}
    vectors['total'] = values[-1]-values[0]
    bound = 32*np.finfo(np.float64).eps*sum(np.abs(v) for v in values)
    if np.any(np.abs(sum(vectors[p] for p in PARTS)-vectors['total']) > bound):
        raise ValueError('signed attribution does not close')
    vectors['propagated'] = values[3]-values[1]
    vectors['local'] = values[-1]-values[3]
    return {k: drift.scan.error_score(v, values[0]) for k,v in vectors.items()}, vectors


def coverage(rows):
    if [(r['layer'], r['position']) for r in rows] != list(replay.TARGETS.items()):
        raise ValueError('incomplete or reordered target coverage')


def validate(capture):
    prep, measured = replay.audited(capture)
    proof = json.loads((capture/'audit.json').read_text())
    if (not measured['control_valid'] or proof.get('complete') is not True or
            proof.get('control_valid') is not True or proof.get('model_acceptance') is not False or
            proof.get('identity') != measured['identity'] or proof.get('artifacts') != len(measured['artifacts']) or
            proof.get('report_sha256') != digest(capture/'report.json')):
        raise ValueError('requires separately audited exact native replay')
    inputs = prep['inputs'] | prep['artifacts']
    inputs.update({str(capture/n): h for n,h in measured['artifacts'].items()})
    inputs.update({str(capture/n): digest(capture/n) for n in ('preparation.json', 'report.json', 'audit.json')})
    return prep, inputs


def store(out, expected, name, arrays):
    if out is not None:
        np.savez(out/name, **arrays)
    else:
        with np.load(expected/name, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(saved[k].dtype != v.dtype or saved[k].shape != v.shape or
                                                     saved[k].tobytes() != v.tobytes() for k,v in arrays.items()):
                raise ValueError('independent causal re-audit differs: '+name)


def evaluate(capture, prep, out=None, expected=None):
    model_dir = Path(prep['meta']['model'])
    spec = base.spec_from_model_dir(model_dir)
    off = boundary.layout(spec, max_ctx=prep['meta']['max_ctx'])
    bundle = base.load_bundle(Path(prep['kernels']), base.manifest(spec, max_ctx=prep['meta']['max_ctx']))
    norm_dtype, conv_dtype = ds.linear_dtypes(bundle)
    if norm_dtype != 'float32' or conv_dtype != 'fp32':
        raise ValueError('requires production FP32 linear entry/history')
    # Confirm the appended FP32 slot, rather than interpreting the legacy BF16 slot.
    from recipes.qwen36moe import linear_post_fp32
    if not linear_post_fp32(spec):
        raise ValueError('requires production FP32 post-output')
    refs = Path(prep['references'])
    model = base.Q4NX(model_dir/'model.q4nx')
    model.native_q8 = lambda name: False
    cached = attribution.Cached(model)
    rows, artifacts = [], {}
    peak = 0
    try:
        for layer, position in replay.TARGETS.items():
            print('analyzing layer', layer, 'through position', position, flush=True)
            target = capture/f'L{layer}'
            xs = np.array([base.read(target/f'input_{p}.bin', spec.hidden) for p in range(position+1)], dtype='<f4')
            reference_x = replay.trace(refs/f'ref_res{layer-1}.bin', 257, spec.hidden)
            reference_y = replay.trace(refs/f'ref_res{layer}.bin', 257, spec.hidden)[position]
            # The retained mixed-precision reference must reconstruct bytewise.
            ref, states = drift.replay(cached, spec, layer, reference_x, (position,))
            attribution.exact_reference(ref[position], reference_y)
            mixed = drift.on_history(cached, spec, layer, xs[position], states[position], position)
            native_oracle, _ = drift.replay(cached, spec, layer, xs, (position,))
            dw, weight_hashes = di.load_weights(model, spec, layer)
            captures = []
            for p in range(position+1):
                data = (target/f'act_{p}.bin').read_bytes()
                c = ds.read_boundaries(data, (target/f'state_{p}.bin').read_bytes(), spec, off, conv_dtype=conv_dtype)
                c['xn'] = di.read_entry(data, off, spec.hidden, norm_dtype=norm_dtype)
                captures.append(c)
            delta = di.score_sequence(spec, dw, xs, captures, conv_dtype=conv_dtype)
            history = np.zeros((spec.conv_kernel-1, spec.lin_qkv_dim))
            state = np.zeros((spec.lin_value_heads, spec.lin_key_dim, spec.lin_value_dim))
            for x in xs:
                xn = base.oracle.rms(x, spec.norm_eps)*dw['norm']
                records, history = di.records_reference(spec, xn, dw['qkv']@xn, history, dw)
                state, independent_o = ds.step_reference(state, records)
            prefix = f'model.layers.{layer}.'
            w = {k: cached.matmul_w(prefix+n, o, i).astype(np.float64) for k,n,o,i in (
                ('z', 'self_attn.gate_proj.weight', spec.lin_value_width, spec.hidden),
                ('out', 'linear_attn.ssm_out_proj.weight', spec.hidden, spec.lin_value_width),
                ('up', 'mlp.up_proj.weight', spec.intermediate, spec.hidden),
                ('gate', 'mlp.gate_proj.weight', spec.intermediate, spec.hidden),
                ('down', 'mlp.down_proj.weight', spec.hidden, spec.intermediate))}
            w['post_norm'] = model.bf16(prefix+'post_attention_layernorm.weight').astype(np.float64)
            nw = model.bf16(prefix+'linear_attn.ssm_norm.weight').astype(np.float64)
            data = (target/f'act_{position}.bin').read_bytes()
            c = captures[-1]
            x = xs[-1].astype(np.float64)
            z = region(data, off, 'A_Z', spec.lin_value_width)
            og = region(data, off, 'A_OG', spec.lin_value_width)
            projected = region(data, off, 'A_OUT', spec.hidden)
            res = region(data, off, 'A_RES', spec.hidden)
            f = boundary.front.read_front(target/f'act_{position}.bin', off, spec.hidden, spec.intermediate,
                                          False, norm_dtype='float32')
            d = boundary.down.read_regions(target/f'act_{position}.bin', off, spec.hidden, spec.intermediate, False)
            actual = base.read(target/f'xres_{position}.bin', spec.hidden)

            def post(o, z):
                return (base.oracle.rms(o, spec.norm_eps)*nw).reshape(-1)*base.oracle.silu(z)

            def tail(r):
                return attribution.tail(r, w, spec.norm_eps)

            def compose(post_output):
                return tail(x+w['out']@post_output)

            independent_z = w['z']@xn
            ideal_post = post(independent_o, independent_z)
            conditioned_post = post(c['output'].astype(np.float64), z)
            nodes = dict(reference=reference_y.astype(np.float64), oracle_reference=ref[position].astype(np.float64),
                         current_input=mixed.astype(np.float64), input_history=native_oracle[position].astype(np.float64),
                         fp64=compose(ideal_post), recurrence=compose(post(c['output'].astype(np.float64), independent_z)),
                         z=compose(conditioned_post), post=compose(og), projection=tail(x+projected), residual_add=tail(res),
                         ffn_norm=res+w['down']@boundary.front.activation(w['up'], w['gate'], f['xm']),
                         ffn_activation=res+w['down']@f['h'], down=res+sum(d['partials']), native=actual.astype(np.float64))
            metrics, vectors = decompose(nodes)
            front = boundary.front.compare_front(res, w['post_norm'], w['up'], w['gate'], f['xm'], f['h'], spec.norm_eps)
            down = boundary.down.compare_partials(f['h'], w['down'], res, d['partials'], actual,
                                                  tuple(boundary.ffn_geometry(spec).DOWN_SPLIT))
            projections = {k: boundary.front.metric(a,b) for k,a,b in (
                ('z', z, independent_z), ('z_conditioned', z, w['z']@c['xn']),
                ('post_conditioned', og, conditioned_post), ('out_conditioned', projected, w['out']@og),
                ('residual_add', res, x+projected))}
            summary = dict(total=len(delta), passed=sum(r['passed'] for r in delta),
                           strict_heads_passed=sum(r['recurrence']['head_diagnostics_passed'] for r in delta),
                           first_failure=next((r['token'] for r in delta if not r['passed'] or
                                               not r['recurrence']['head_diagnostics_passed']), None))
            detail = dict(layer=layer, cases=delta, summary=summary, weights=weight_hashes)
            detail_name = f'L{layer}-delta.json'
            if out is not None:
                write(out/detail_name, detail)
            elif json.loads((expected/detail_name).read_text()) != detail:
                raise ValueError('causal DeltaNet re-audit differs')
            arrays = {'node_'+k:v for k,v in nodes.items()} | {'effect_'+k:v for k,v in vectors.items()}
            arrays.update(native_input=x, reference_input=reference_x[position], post_independent=ideal_post,
                          post_conditioned=conditioned_post, post_native=og,
                          recurrence_independent=independent_o, recurrence_native=c['output'],
                          z_independent=independent_z, z_native=z)
            name = f'L{layer}_P{position}.npz'
            store(out, expected, name, arrays)
            if out is not None:
                artifacts.update({n: digest(out/n) for n in (name, detail_name)})
            local_passed = (summary['passed'] == len(delta) and summary['strict_heads_passed'] == len(delta) and
                            front['passed'] and down['passed'] and all(m['passed'] for m in projections.values()))
            row = dict(layer=layer, position=position, reference_exact=True, metrics=metrics,
                       input=boundary.front.metric(x, reference_x[position].astype(np.float64)),
                       delta=summary, selected_delta=delta[-1], front=front, down=down, projections=projections,
                       local_passed=local_passed)
            rows.append(row)
            print('target', layer, position, 'total', metrics['total']['maxrel'],
                  'propagated', metrics['propagated']['maxrel'], 'local', metrics['local']['maxrel'], flush=True)
            peak = max(peak, sum(v.nbytes for v in cached.cache.values())+sum(v.nbytes for v in dw.values())+
                       sum(v.nbytes for v in w.values()))
            cached.cache.clear()
            del captures, dw, w, states, state
    finally:
        cached.cache.clear()
        model.mm.close()
        model.f.close()
    coverage(rows)
    return rows, artifacts, peak


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, model_acceptance=False, sources=sources())
    try:
        capture = args.capture.resolve()
        prep, inputs = validate(capture)
        rows, artifacts, peak = evaluate(capture, prep, out=out)
        base.verify(inputs)
        base.verify(result['sources'])
        result.update(complete=True, capture=str(capture), inputs=inputs, cases=rows, artifacts=artifacts,
                      materialized_weight_bytes=peak, local_passed=all(r['local_passed'] for r in rows),
                      numerical_passed=all(r['metrics']['total']['passed'] for r in rows),
                      scope='causal candidate inputs; independent mixed/FP64 histories from zero; ordered signed attribution through nonlinear FFN; no native correction or performance acceptance')
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        write(out/'report.json', result)
    return 0 if result['numerical_passed'] else 1


def audit(args):
    out = args.out.resolve()
    proof = dict(complete=False, model_acceptance=False)
    try:
        saved = json.loads((out/'report.json').read_text())
        if saved.get('complete') is not True or saved.get('model_acceptance') is not False or saved['sources'] != sources():
            raise ValueError('incomplete or changed analysis')
        coverage(saved['cases'])
        names = {n for l,p in replay.TARGETS.items() for n in (f'L{l}_P{p}.npz', f'L{l}-delta.json')}
        actual = {p.name for p in out.glob('*.npz')} | {p.name for p in out.glob('L*-delta.json')}
        if set(saved['artifacts']) != names or actual != names:
            raise ValueError('incomplete analysis artifact inventory')
        base.verify({str(out/n): h for n,h in saved['artifacts'].items()})
        prep, inputs = validate(Path(saved['capture']))
        if saved['inputs'] != inputs:
            raise ValueError('analysis inputs changed')
        rows, _, peak = evaluate(Path(saved['capture']), prep, expected=out)
        if (rows != saved['cases'] or peak != saved['materialized_weight_bytes'] or
                saved['local_passed'] != all(r['local_passed'] for r in rows) or
                saved['numerical_passed'] != all(r['metrics']['total']['passed'] for r in rows)):
            raise ValueError('analysis measurements or verdict changed')
        base.verify(inputs)
        base.verify(saved['sources'])
        proof.update(complete=True, numerical_passed=saved['numerical_passed'], local_passed=saved['local_passed'],
                     report_sha256=digest(out/'report.json'), cases=len(rows))
    except Exception as error:
        proof['error'] = str(error)
        raise
    finally:
        write(out/'audit.json', proof)
    return 0 if proof['numerical_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action, flags in (('run', ('out', 'capture')), ('audit', ('out',))):
        command = sub.add_parser(action)
        for flag in flags:
            command.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(globals()[args.action](args))
