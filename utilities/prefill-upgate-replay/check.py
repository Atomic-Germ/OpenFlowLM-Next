"""Isolated paired-input L63/P7 up/gate replay using unchanged production GEMM."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


proj = load('_replay_projections', ROOT / 'utilities/prefill-projections/check.py')
attribution = load('_replay_attribution', ROOT / 'utilities/prefill-sublayer-trace/attribute.py')
digest, verify = proj.digest, proj.verify
SLOTS = (7, 135)
N, K, T = 34816, 5120, 256


def split_inputs(x):
    x = np.asarray(x, np.float32)
    if not x.size or not np.isfinite(x).all():
        raise ValueError('finite nonempty input required')
    high = proj.bf16_to_f32(proj.f32_to_bf16(x))
    if not np.isfinite(high).all():
        raise ValueError('input outside finite BF16 range')
    return x.copy(), np.subtract(x, high, dtype=np.float32)


def control(high, low, captured, slot):
    high, low, captured = [np.asarray(x, np.float32) for x in (high, low, captured)]
    if (high.ndim != 2 or high.shape[1] != T or not high.shape[0]
            or high.shape != low.shape or captured.shape != (high.shape[0],)
            or type(slot) != int or not 0 <= slot < T
            or not all(np.isfinite(x).all() for x in (high, low, captured))):
        raise ValueError('invalid control shape/slot/finite data')
    if any(np.any(np.delete(x, slot, axis=1) != 0) for x in (high, low)):
        raise ValueError('nonzero inactive column')
    selected = np.add(high[:, slot], low[:, slot], dtype=np.float32)
    if selected.tobytes() != captured.tobytes():
        raise ValueError('paired control lacks byte-exact capture identity')
    return selected


def projection_metrics(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    if (got.ndim != 1 or got.shape != ref.shape or not got.size or got.size % 2
            or not np.isfinite(got).all() or not np.isfinite(ref).all()):
        raise ValueError('invalid projection comparison')
    half = got.size // 2
    return {name: proj.metric(got[lo:hi], ref[lo:hi])
            for name, lo, hi in (('up', 0, half), ('gate', half, got.size))}


def build(a):
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    evidence = dict(complete=False, commands=[])
    try:
        sources = [Path(__file__), ROOT / 'open_kernels/harness/run_kernel.cpp',
                   ROOT / 'open_kernels/harness/CMakeLists.txt']
        evidence['sources'] = {str(p): digest(p) for p in sources}
        commands = [
            ['cmake', '-S', str(ROOT / 'open_kernels/harness'), '-B', str(out / 'build'),
             '-DCMAKE_BUILD_TYPE=Release', f'-DXRT_INCLUDE_DIR={a.xrt_include.resolve()}',
             f'-DXRT_LIB_DIR={a.xrt_lib.resolve()}'],
            ['cmake', '--build', str(out / 'build'), '-j', '2'],
        ]
        for cmd in commands:
            evidence['commands'].append(cmd)
            subprocess.run(cmd, check=True)
        deps = list((out / 'build/CMakeFiles/run_kernel.dir').glob('*.o.d'))
        if len(deps) != 1:
            raise ValueError('missing harness compiler dependencies')
        names = shlex.split(deps[0].read_text().replace('\\\n', ' ').split(':', 1)[1])
        evidence.update(
            complete=True,
            compiler_dependencies={str(Path(p).resolve()): digest(p) for p in names},
            harness_sha256=digest(out / 'build/run_kernel'),
            cmake_cache_sha256=digest(out / 'build/CMakeCache.txt'),
            compiler=subprocess.check_output(['c++', '--version'], text=True))
    finally:
        (out / 'build-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')


def prepare(a):
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    audit = dict(complete=False)
    try:
        meta, _ = attribution.verify_inputs(SimpleNamespace(
            run=a.capture.resolve(), build=a.capture_build.resolve(), baseline=a.baseline.resolve()))
        spec = proj.load_spec(a.kernels / 'spec.json')
        manifest = proj.load_bundle(a.kernels, proj.manifest(spec, max_ctx=meta['max_ctx']))
        if (meta['layers'] != 64 or meta['hidden'] != K or meta['build_key'] != manifest['build_key']
                or meta['spec_hash'] != manifest['spec_hash']):
            raise ValueError('requires matching complete production capture/export')
        case = next(c for c in proj.cases(manifest) if c['tag'] == 'L3_up_gate')
        if case['n'] != N or case['k'] != K or case['storage'] != 'pool':
            raise ValueError('unsupported up/gate geometry')
        prior = json.loads(a.original.read_text())
        pool_path = a.pools.resolve() / 'pool_L63.bin'
        prior = {str(Path(p).resolve()): h for p, h in prior.items()}
        if prior.get(str(pool_path)) != digest(pool_path):
            raise ValueError('retained pool provenance mismatch')
        model_path = Path(meta['model']) / 'model.q4nx'
        model = proj.Q4NX(model_path)
        model.native_q8 = lambda name: False
        pool = np.zeros(N * K * 5 // 8, np.uint8)
        dst = 0
        with pool_path.open('rb') as f:
            for op in case['ops']:
                size = op['nch'] * 5120
                proj.apply_op(dict(op, dst=dst), model, 63, pool)
                f.seek(op['dst'])
                if pool[dst:dst+size].tobytes() != f.read(size):
                    raise ValueError('packed GEMM weight differs from retained pool')
                dst += size
        if dst != pool.size:
            raise ValueError('incomplete weight coverage')
        artifacts = {}

        def save(name, values):
            path = out / name
            path.write_bytes(np.ascontiguousarray(values).tobytes())
            artifacts[str(path)] = digest(path)

        save('weights.bin', pool)
        x = attribution.base.read(a.capture / 'hybrid_L63_P7_norm.bin', K).astype(np.float32)
        captured = attribution.base.read(a.capture / 'hybrid_L63_P7_upgate.bin', N).astype(np.float32)
        save('captured.bin', captured)
        save('input.bin', x)
        high, low = split_inputs(x)
        xh, xl = [proj.bf16_to_f32(proj.f32_to_bf16(v)).astype(np.float64) for v in (high, low)]
        inputs = np.stack((x.astype(np.float64), xh + xl, xh, xl), axis=1)
        refs = np.empty((N, 4), np.float64)
        rounded = np.empty_like(refs)
        # Each projection occupies its own RS2 pool; retain the concatenated
        # up/gate layout used by the production manifest.
        size = pool.size // 2
        for role, name in enumerate(('up', 'gate')):
            w = proj.dequant_pool(pool[role*size:(role+1)*size], N//2, K, rs=2)
            source_w = model.matmul_w(f'model.layers.63.mlp.{name}_proj.weight', N//2, K)
            if not np.array_equal(w, source_w):
                raise ValueError('packed/source effective weights differ')
            del source_w
            for row in range(0, N//2, 256):
                rows = slice(role*N//2 + row, role*N//2 + row + 256)
                ww = w[row:row+256]
                refs[rows] = ww.astype(np.float64) @ inputs
                wr = proj.bf16_to_f32(proj.f32_to_bf16(ww)).astype(np.float64)
                rounded[rows] = wr @ inputs
            del w
        del pool
        save('exact-weight-ref.bin', refs)
        save('bf16-weight-ref.bin', rounded)
        commands = []
        kernel_dir = a.kernels.resolve() / case['kernel']
        for slot in SLOTS:
            for part, vec in (('high', high), ('low', low)):
                tag = f'slot{slot}-{part}'
                full = np.zeros((T, K), np.float32)
                full[slot] = vec
                save(tag + '-x.bin', proj.tile_input(full))
                cfg = out / (tag + '.cfg')
                cfg.write_text('\n'.join([
                    'device', f'xclbin G {kernel_dir}/final.xclbin',
                    f'kernelx k G {kernel_dir}/insts.bin',
                    f'buf w {N*K*5//8} {out}/weights.bin',
                    f'buf x {K*T*2} {out}/{tag}-x.bin', f'buf y {N*T*4}',
                    'run k w x y', f'dump y {out}/{tag}-y.bin {N*T*4}', '']))
                artifacts[str(cfg)] = digest(cfg)
                commands.append(str(cfg))
        files = [Path(__file__), Path(proj.__file__), Path(attribution.__file__),
                 a.capture / 'sublayer-audit.json', a.original, pool_path, model_path,
                 a.capture / 'hybrid_L63_P7_norm.bin', a.capture / 'hybrid_L63_P7_upgate.bin',
                 a.kernels / 'manifest.json', a.kernels / 'spec.json', a.kernels / 'toolchain.json',
                 kernel_dir / 'final.xclbin', kernel_dir / 'insts.bin',
                 ROOT / 'open_kernels/q4_1_pack.py', ROOT / 'open_kernels/model/q4nx.py']
        files += list((ROOT / 'open_kernels/recipes').glob('*.py'))
        audit.update(complete=True, scope='L63/P7 paired up/gate; sparse T256 control',
                     slots=SLOTS, n=N, k=K, commands=commands, artifacts=artifacts,
                     source_weight_identity=True, retained_pool_identity=True,
                     build_key=manifest['build_key'], spec_hash=manifest['spec_hash'],
                     inputs={str(p.resolve()): digest(p) for p in files})
    finally:
        (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')


def run(a):
    out = a.out.resolve()
    result = dict(control_valid=False, model_acceptance=False)
    try:
        audit = json.loads((out / 'audit.json').read_text())
        build_record = a.build.resolve() / 'build-evidence.json'
        b = json.loads(build_record.read_text())
        if not audit['complete'] or not b['complete'] or audit['slots'] != list(SLOTS):
            raise ValueError('incomplete or incompatible replay')
        verify(audit['inputs'])
        verify(audit['artifacts'])
        verify(b['sources'])
        verify(b['compiler_dependencies'])
        harness = a.build.resolve() / 'build/run_kernel'
        verify({str(harness): b['harness_sha256'],
                str(a.build.resolve() / 'build/CMakeCache.txt'): b['cmake_cache_sha256']})
        for slot in SLOTS:
            for part in ('high', 'low'):
                (out / f'slot{slot}-{part}-y.bin').unlink(missing_ok=True)
        for cfg in audit['commands']:
            with Path(cfg).with_suffix('.log').open('w') as f:
                subprocess.run([str(harness), cfg], check=True, timeout=120, stdout=f,
                               stderr=subprocess.STDOUT,
                               env=os.environ | {'HARNESS_TIMEOUT_MS': '30000'})
        captured = attribution.base.read(out / 'captured.bin', N).astype(np.float32)
        raw = np.fromfile(out / 'exact-weight-ref.bin', '<f8').reshape(N, 4)
        rounded = np.fromfile(out / 'bf16-weight-ref.bin', '<f8').reshape(N, 4)
        results, hashes = [], {}
        for slot in SLOTS:
            ys = []
            for part in ('high', 'low'):
                path = out / f'slot{slot}-{part}-y.bin'
                ys.append(attribution.base.read(path, N*T).astype(np.float32).reshape(N, T))
                hashes[str(path)] = digest(path)
            selected = control(*ys, captured, slot)
            results.append(dict(slot=slot, byte_exact=True, zero_inactive_columns=True,
                exact_weight_fp32_input=projection_metrics(selected, raw[:, 0]),
                exact_weight_paired_input=projection_metrics(selected, raw[:, 1]),
                bf16_weight_paired_input=projection_metrics(selected, rounded[:, 1]),
                weight_rounding_effect=projection_metrics(rounded[:, 1], raw[:, 1]),
                high_vs_bf16_weights=projection_metrics(ys[0][:, slot], rounded[:, 2]),
                low_vs_bf16_weights=projection_metrics(ys[1][:, slot], rounded[:, 3])))
        verify(audit['artifacts'])
        evidence = [Path(__file__), build_record, out / 'audit.json']
        evidence += [Path(c).with_suffix('.log') for c in audit['commands']]
        result.update(control_valid=True, cases=results, outputs=hashes,
                      scope=audit['scope'], native_dispatches=len(audit['commands']),
                      evidence={str(p.resolve()): digest(p) for p in evidence})
    except Exception as e:
        result.update(control_valid=False, error=str(e))
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    b = subs.add_parser('build')
    for name in ('out', 'xrt-include', 'xrt-lib'):
        b.add_argument('--' + name, type=Path, required=True)
    prep = subs.add_parser('prepare')
    for name in ('out', 'capture', 'capture-build', 'baseline', 'kernels', 'pools', 'original'):
        prep.add_argument('--' + name, type=Path, required=True)
    r = subs.add_parser('run')
    for name in ('out', 'build'):
        r.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(globals()[args.command](args))
