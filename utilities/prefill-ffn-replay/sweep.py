"""Replay paired-weight FFN on selected retained causal rows, without model residency."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

s = importlib.util.spec_from_file_location('_ffn_composed', Path(__file__).with_name('check.py'))
composed = importlib.util.module_from_spec(s)
s.loader.exec_module(composed)
proj, parent, base, paired = composed.proj, composed.parent, composed.base, composed.paired
ROOT, DEFAULT = composed.ROOT, composed.DEFAULT
H, F, T = composed.H, composed.F, composed.T
TARGETS = ((60, 38), (35, 80))
digest, verify, identity = composed.digest, composed.verify, composed.identity


def slots(position):
    if type(position) is not int or not 0 <= position < T:
        raise ValueError('captured position must be an integer in the first T256 block')
    return position, (position + T//2) % T


def operation(manifest, layer, role):
    # Linear and attention FFNs share geometry but not their pool offsets.
    tag = f'L{3 if layer % 4 == 3 else 0}_{role}'
    case = next(c for c in proj.cases(manifest) if c['tag'] == tag)
    expected = (2*F, H) if role == 'up_gate' else (H, F)
    if (case['n'], case['k']) != expected or case['storage'] != 'pool':
        raise ValueError('unsupported FFN geometry/storage')
    return case


def summarize(cases):
    expected = {(l, p, s) for l, p in TARGETS for s in slots(p)}
    actual = [(c['layer'], c['position'], c['slot']) for c in cases]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('incomplete/duplicate selected-row coverage')
    if not all(c['control_exact'] and c['slot_exact'] for c in cases):
        raise ValueError('invalid native control/slot identity')
    local = all(c['projections_passed'] and c['ffn_passed'] for c in cases)
    return dict(control_valid=True, local_passed=local,
                passed=local and all(c['output_passed'] for c in cases), model_acceptance=False)


def prepare(a):
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    audit = dict(complete=False)
    try:
        host = composed.Host(a.bridge)
        meta, _ = parent.attribution.verify_inputs(SimpleNamespace(
            run=a.capture.resolve(), build=a.capture_build.resolve(), baseline=a.baseline.resolve()))
        spec = proj.load_spec(a.kernels/'spec.json')
        manifest = proj.load_bundle(a.kernels, proj.manifest(spec, max_ctx=meta['max_ctx']))
        if (meta['layers'] != 64 or meta['hidden'] != H or spec.hidden != H or spec.intermediate != F
                or meta['build_key'] != manifest['build_key'] or meta['spec_hash'] != manifest['spec_hash']):
            raise ValueError('incompatible capture/export')
        for mode in ('control', 'low'):
            paired.verify_build(a.kernel_builds/mode, mode)
        model_path = Path(meta['model'])/'model.q4nx'
        model = proj.Q4NX(model_path)
        model.hidden = H
        model.native_q8 = lambda name: False
        prior = {str(Path(p).resolve()): h for p, h in json.loads(a.original.read_text()).items()}
        files = [Path(__file__), Path(composed.__file__), Path(parent.__file__), Path(paired.__file__),
                 Path(proj.__file__), Path(parent.attribution.__file__), Path(base.__file__), Path(base.oracle.__file__),
                 Path(__file__).with_name('bridge.py'), a.bridge/'build.json', a.capture/'sublayer-audit.json',
                 a.capture/'reference.json', a.original, model_path, a.kernels/'manifest.json',
                 a.kernels/'spec.json', a.kernels/'toolchain.json',
                 ROOT/'open_kernels/q4_1_pack.py', ROOT/'open_kernels/model/q4nx.py']
        files += list((ROOT/'open_kernels/recipes').glob('*.py'))
        artifacts, targets = {}, []

        def save(path, values):
            path.write_bytes(np.ascontiguousarray(values).tobytes())
            artifacts[str(path)] = digest(path)

        for layer, pos in TARGETS:
            directory = out/f'L{layer}-P{pos}'
            directory.mkdir()
            prefix = a.capture/f'hybrid_L{layer}_P{pos}'
            captures = {}
            for key, count in (('res', H), ('norm', H), ('upgate', 2*F), ('h', F), ('down', H)):
                path = Path(str(prefix)+'_'+key+'.bin')
                captures[key] = base.read(path, count).astype(np.float32)
                files.append(path)
                save(directory/(key+'.bin'), captures[key])
            nw = model.bf16(f'model.layers.{layer}.post_attention_layernorm.weight')
            identity(host.norm(captures['res'], nw, spec.norm_eps), captures['norm'], 'host norm')
            identity(host.activation(captures['upgate']), captures['h'], 'host activation')
            n = len(meta['ids'])
            output_path = a.capture/f'hybrid_res{layer}.bin'
            reference_path = a.capture/f'ref_res{layer}.bin'
            save(directory/'output.bin', base.read(output_path, (n+1)*H).reshape(n+1, H)[pos].astype(np.float32))
            save(directory/'reference.bin', base.read(reference_path, n*H).reshape(n, H)[pos].astype(np.float32))
            save(directory/'ideal-ffn.bin', base.oracle.ffn(model, spec, layer, captures['res']).astype('<f8'))
            files += [output_path, reference_path]
            pool_path = a.pools.resolve()/f'pool_L{layer}.bin'
            if prior.get(str(pool_path)) != digest(pool_path):
                raise ValueError('retained pool provenance mismatch')
            files.append(pool_path)
            kernels = {}
            for role in ('up_gate', 'down'):
                case = operation(manifest, layer, role)
                nout, kin = case['n'], case['k']
                pool = np.zeros(nout*kin*5//8, np.uint8)
                dst = 0
                with pool_path.open('rb') as f:
                    for op in case['ops']:
                        size = op['nch']*5120
                        proj.apply_op(dict(op, dst=dst), model, layer, pool)
                        f.seek(op['dst'])
                        if pool[dst:dst+size].tobytes() != f.read(size):
                            raise ValueError('packed FFN weights differ from retained pool')
                        dst += size
                if dst != pool.size:
                    raise ValueError('incomplete packing')
                save(directory/(role+'-weights.bin'), pool)
                roles = ('up', 'gate') if role == 'up_gate' else ('down',)
                refs = []
                chunk = pool.size//len(roles)
                for i, name in enumerate(roles):
                    rows = nout//len(roles)
                    w = proj.dequant_pool(pool[i*chunk:(i+1)*chunk], rows, kin, rs=2)
                    original = model.matmul_w(f'model.layers.{layer}.mlp.{name}_proj.weight', rows, kin)
                    if not np.array_equal(w, original):
                        raise ValueError('effective FFN source weights differ')
                    del original
                    if role == 'up_gate':
                        # Independent exact-effective-weight reference, conditioned on original FP32 norm.
                        refs.append(w.astype(np.float64) @ captures['norm'].astype(np.float64))
                    del w
                if refs:
                    save(directory/'projection-reference.bin', np.concatenate(refs).astype('<f8'))
                del pool
                kernel = a.kernels.resolve()/case['kernel']
                kernels[role] = str(kernel)
                files += [kernel/'final.xclbin', kernel/'insts.bin']
            targets.append(dict(layer=layer, position=pos, slots=slots(pos), kernels=kernels,
                                norm_exact=True, activation_exact=True, pool_exact=True, effective_weights_exact=True))
            print(f'Prepared L{layer}/P{pos}: host identity and both weight layouts exact', flush=True)
        for mode in ('control', 'low'):
            files.append(a.kernel_builds/mode/'build.json')
        audit.update(complete=True, targets=targets, artifacts=artifacts,
                     scope='two retained hybrid causal rows; no attention/state propagation',
                     bridge=str(a.bridge.resolve()), kernel_builds=str(a.kernel_builds.resolve()),
                     inputs={str(p.resolve()): digest(p) for p in files})
    finally:
        (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')


def run(a):
    out = a.out.resolve()
    result = dict(passed=False, control_valid=False, model_acceptance=False)
    try:
        audit = json.loads((out/'audit.json').read_text())
        if not audit['complete'] or [(t['layer'], t['position']) for t in audit['targets']] != list(TARGETS):
            raise ValueError('incomplete preparation')
        verify(audit['inputs'])
        verify(audit['artifacts'])
        host = composed.Host(Path(audit['bridge']))
        builds = Path(audit['kernel_builds'])
        for mode in ('control', 'low'):
            paired.verify_build(builds/mode, mode)
        hb = a.harness_build.resolve()
        h = json.loads((hb/'build-evidence.json').read_text())
        if not h['complete']:
            raise ValueError('invalid harness')
        verify(h['sources'])
        verify(h['compiler_dependencies'])
        harness = hb/'build/run_kernel'
        verify({str(harness): h['harness_sha256'], str(hb/'build/CMakeCache.txt'): h['cmake_cache_sha256']})
        outputs, evidence, cases = {}, {}, []
        dispatches = 0

        def save(path, values):
            path.write_bytes(np.ascontiguousarray(values).tobytes())
            outputs[str(path)] = digest(path)

        def products(directory, tag, x, kernel, role, slot):
            nonlocal dispatches
            kin, nout = (H, 2*F) if role == 'up_gate' else (F, H)
            ys = []
            for part, v in zip(('high', 'low'), parent.split_inputs(x)):
                stem = directory/f'{tag}-s{slot}-{part}'
                xp, yp, cfg = [Path(str(stem)+suffix) for suffix in ('-x.bin', '-y.bin', '.cfg')]
                full = np.zeros((T, kin), np.float32)
                full[slot] = v
                save(xp, proj.tile_input(full))
                yp.unlink(missing_ok=True)
                cfg.write_text('\n'.join(['device', f'xclbin G {kernel}/final.xclbin',
                    f'kernelx k G {kernel}/insts.bin', f'buf w {nout*kin*5//8} {directory}/{role}-weights.bin',
                    f'buf x {kin*T*2} {xp}', f'buf y {nout*T*4}', 'run k w x y',
                    f'dump y {yp} {nout*T*4}', '']))
                with cfg.with_suffix('.log').open('w') as f:
                    subprocess.run([str(harness), str(cfg)], check=True, timeout=120,
                                   stdout=f, stderr=subprocess.STDOUT,
                                   env=os.environ | {'HARNESS_TIMEOUT_MS': '30000'})
                y = base.read(yp, nout*T).astype(np.float32).reshape(nout, T)
                if np.any(np.delete(y, slot, axis=1) != 0):
                    raise ValueError('nonzero inactive output column')
                ys.append(y[:, slot].copy())
                outputs[str(yp)] = digest(yp)
                for p in (cfg, cfg.with_suffix('.log')):
                    evidence[str(p)] = digest(p)
                dispatches += 1
            return ys

        for target in audit['targets']:
            layer, pos = target['layer'], target['position']
            if target['slots'] != list(slots(pos)):
                raise ValueError('changed token placement')
            directory = out/f'L{layer}-P{pos}'
            read = lambda name, count: base.read(directory/(name+'.bin'), count).astype(np.float32)
            norm, captured, res = read('norm', H), read('upgate', 2*F), read('res', H)
            ref = read('reference', H)
            ideal = np.fromfile(directory/'ideal-ffn.bin', '<f8')
            ugref = np.fromfile(directory/'projection-reference.bin', '<f8')
            selected = []
            for slot in slots(pos):
                production = products(directory, 'production-upgate', norm, target['kernels']['up_gate'], 'up_gate', slot)
                control = np.add(*production, dtype=np.float32)
                identity(control, captured, 'production up/gate')
                private = products(directory, 'private-upgate', norm, builds/'control', 'up_gate', slot)
                paired.require_control(*private, *production)
                act = host.activation(control)
                identity(act, read('h', F), 'control activation')
                cy = np.add(*products(directory, 'control-down', act, target['kernels']['down'], 'down', slot), dtype=np.float32)
                identity(cy, read('down', H), 'control down')
                identity(np.add(res, cy, dtype=np.float32), read('output', H), 'control output')
                low = products(directory, 'low-upgate', norm, builds/'low', 'up_gate', slot)
                ug = paired.combine(*private, *low)
                hact = host.activation(ug)
                down = np.add(*products(directory, 'candidate-down', hact, target['kernels']['down'], 'down', slot), dtype=np.float32)
                output = np.add(res, down, dtype=np.float32)
                for name, values in (('upgate', ug), ('h', hact), ('down', down), ('output', output)):
                    save(directory/f'candidate-s{slot}-{name}.bin', values)
                selected.append((ug, hact, down, output))
                projections = paired.scores(control, ug, ugref)
                baseline = composed.score(res, cy, ideal, ref)
                candidate = composed.score(res, down, ideal, ref)
                case = dict(layer=layer, position=pos, slot=slot, control_exact=True,
                            zero_inactive_columns=True, projections=projections, control=baseline, candidate=candidate,
                            projections_passed=projections['candidate_passed'], ffn_passed=candidate['ffn']['passed'],
                            output_passed=candidate['output']['passed'])
                cases.append(case)
                print(json.dumps(case), flush=True)
            for i, name in enumerate(('upgate', 'h', 'down', 'output')):
                identity(selected[0][i], selected[1][i], name+' placement')
            for case in cases[-2:]:
                case['slot_exact'] = True
        verify(audit['artifacts'])
        for p in (Path(__file__), out/'audit.json', hb/'build-evidence.json'):
            evidence[str(p.resolve())] = digest(p)
        result.update(**summarize(cases), cases=cases, dispatches=dispatches,
                      outputs=outputs, evidence=evidence, scope=audit['scope'])
    except Exception as e:
        result.update(passed=False, control_valid=False, error=str(e))
        raise
    finally:
        (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep, run_parser = sub.add_parser('prepare'), sub.add_parser('run')
    for parser in (prep, run_parser):
        parser.add_argument('--out', type=Path, required=True)
    for name, default in {'bridge': 'prefill-ffn-replay/bridge', 'capture': 'prefill-sublayer-trace/run-257',
        'capture-build': 'prefill-sublayer-trace/verified', 'baseline': 'prefill-residual-trace/run-257',
        'kernel-builds': 'prefill-paired-weights/v2', 'kernels': 'production-consistent/kernels',
        'pools': 'production-full/pools', 'original': 'production-consistent/input-sha256.json'}.items():
        prep.add_argument('--'+name, type=Path, default=DEFAULT/default)
    run_parser.add_argument('--harness-build', type=Path, default=DEFAULT/'prefill-upgate-replay/harness')
    args = p.parse_args()
    raise SystemExit(globals()[args.command](args))
