"""Serial isolated QKV control/candidate sweep; one layer's weights in RAM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from diagnose import dp, digest
from qkv_build import verify_generated


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('model-dir', 'kernel-dir', 'fixture', 'pools', 'build-root', 'out', 'harness'):
        p.add_argument('--'+n, type=Path, required=True)
    a = p.parse_args()
    for n in ('model_dir', 'kernel_dir', 'fixture', 'pools', 'build_root'):
        src = getattr(a, n).resolve()
        if a.out.resolve() == src or a.out.resolve().is_relative_to(src):
            raise ValueError('sweep output must be separate from inputs')
    s = dp.spec_from_model_dir(a.model_dir); meta = dp.load_reference(a.fixture)
    if not 0 < meta['layers'] <= s.num_layers:
        raise ValueError('invalid layer coverage')
    layers = [l for l in range(meta['layers']) if s.layer_types[l] == dp.LINEAR]
    if not layers:
        raise ValueError('no linear layers')
    a.out.mkdir(parents=True, exist_ok=True)
    summary = a.out/'results.json'; summary.unlink(missing_ok=True)
    progress = a.out/'progress.json'; progress.unlink(missing_ok=True)
    results = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', HARNESS_TIMEOUT_MS='30000')
    for layer in layers:
        pair = {}
        for mode in ('control', 'consistent'):
            out = a.out/f'L{layer}'/mode; out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(Path(__file__).with_name('qkv_replay.py')),
                   '--build', str(a.build_root/f'{mode}-build'), '--out', str(out), '--layer', str(layer)]
            for n in ('model-dir', 'kernel-dir', 'fixture', 'pools'):
                cmd += ['--'+n, str(getattr(a, n.replace('-', '_')))]
            with (out/'prepare.log').open('w') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (out/'hardware.log').open('w') as log:
                subprocess.run([str(a.harness.resolve()), str(out/'run.cfg')], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            with (out/'score.log').open('w') as log:
                run = subprocess.run(cmd+['--score'], env=env, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads((out/'score.json').read_text())
            if run.returncode not in (0, 1) or (run.returncode == 0) != report['passed']:
                raise ValueError('inconsistent checker exit status')
            pair[mode] = report
        pins = [json.loads((a.out/f'L{layer}'/m/'inputs.json').read_text()) for m in pair]
        if pins[0]['provenance'] != pins[1]['provenance']:
            raise ValueError('control/candidate input provenance differs')
        # The change is exclusively in the sum fields, not INT16 values/scales.
        for t in range(meta['tokens']):
            tables = [(a.out/f'L{layer}'/m/f'table{t}.bin').read_bytes() for m in pair]
            if tables[0][:10880] != tables[1][:10880]:
                raise ValueError('candidate changed activation quantization')
        results.append(dict(layer=layer, **pair))
        progress.write_text(json.dumps(dict(completed=len(results), total=len(layers), last_layer=layer), indent=2)+'\n')
        print('layer', layer, *(f'{m}: '+('PASS' if r['passed'] else 'FAIL') for m, r in pair.items()), flush=True)
    report = dict(scope=__doc__, passed=all(r[m]['passed'] for r in results for m in ('control', 'consistent')),
                  layers=layers, tokens=meta['tokens'], build_key=meta['build_key'],
                  harness_sha256=digest(a.harness), source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                  evidence_sha256={str(p): digest(p) for p in sorted(a.out.glob('L*/*/*'))
                                   if p.name in ('inputs.json', 'score.json', 'hardware.log')}, results=results)
    summary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
