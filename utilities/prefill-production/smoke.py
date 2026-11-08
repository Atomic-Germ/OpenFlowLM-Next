"""Check unpatched shared-CLI logits against a retained, audited hybrid capture.

This is an integration smoke gate, not an all-layer or request/reset audit.
Historical validator sources are not rerun or rewritten. Their captured logit
hashes are the golden inputs; model bytes must still match their reference.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_production_capture', ROOT/'utilities/prefill-expanded-cli/run.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
base, digest = capture.base, capture.digest
names = base.expected_dumps


def read_fixture(prepared, expected_sha):
    path = prepared/'preparation.json'
    if digest(path) != expected_sha:
        raise ValueError('baseline preparation changed')
    record = json.loads(path.read_text())
    meta = prepared/'references/reference.json'
    if record['artifacts'].get(str(meta)) != digest(meta):
        raise ValueError('baseline reference metadata changed')
    return json.loads(meta.read_text())


def check_outputs(directory, route, tokens, vocab, expected):
    wanted = set(names(route, tokens))
    actual = {p.name for p in directory.glob(route+'_p*.bin')} | {p.name for p in directory.glob(route+'_t*.bin')}
    if actual != wanted:
        raise ValueError('incomplete or extra logit inventory')
    result = {}
    for name in sorted(wanted):
        base.read(directory/name, vocab, expected[name])
        result[name] = digest(directory/name)
    if result[route+'_t0.bin'] != result[f'{route}_p{tokens-1}.bin']:
        raise ValueError('last prompt logit alias differs')
    return result


def run(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    r = dict(complete=False, passed=False, model_acceptance=False, routes={},
             scope='64-layer short hybrid logits; no all-layer or reset acceptance')
    try:
        baseline, kernels, cli = a.baseline.resolve(), a.kernels.resolve(), a.cli.resolve()
        saved = json.loads((baseline/'report.json').read_text())
        audit = json.loads((baseline/'audit.json').read_text())
        if not (saved['complete'] and saved['passed'] and audit['complete'] and audit['passed']
                and audit['report_sha256'] == digest(baseline/'report.json')):
            raise ValueError('baseline is not bound to a passing audit')
        prepared = Path(saved['prepared'])
        refs = prepared/'references'
        meta = read_fixture(prepared, saved['preparation_sha256'])
        if meta['layers'] != 64 or len(meta['ids']) != 20 or meta['hidden'] != 5120:
            raise ValueError('expected the full-depth 20-token fixture')
        manifest = json.loads((kernels/'manifest.json').read_text())
        if manifest['manifest_version'] != 4 or manifest['spec_hash'] != meta['spec_hash']:
            raise ValueError('paired manifest/model mismatch')
        toolchain = json.loads((kernels/'toolchain.json').read_text())
        if toolchain['build_key'] != manifest['build_key'] or len(toolchain['sets']) != 46:
            raise ValueError('incomplete paired export')
        inputs = {str(kernels/p): h for p,h in toolchain['sha256'].items()}
        inputs.update({p:h for p,h in meta['sources'].items() if Path(p).is_relative_to(Path(meta['model']))})
        inputs[str(refs/'ids.txt')] = meta['ids_sha256']
        # Pin the retained references as data; do not pretend the old source
        # preparation passes unchanged after the production recipe promotion.
        inputs.update({str(refs/p): h for p,h in meta['references'].items()})
        for p in (baseline/'report.json', baseline/'audit.json', prepared/'preparation.json',
                  refs/'reference.json', kernels/'manifest.json', kernels/'toolchain.json', cli,
                  Path(__file__), Path(capture.__file__), Path(base.__file__)):
            inputs[str(p)] = digest(p)
        print('verifying model, kernel and reference hashes', flush=True)
        base.verify(inputs)
        r['inputs'] = inputs
        (out/'ids.txt').write_bytes((refs/'ids.txt').read_bytes())
        env = capture.environment()
        env.update(XILINX_XRT='/opt/xilinx/xrt', LD_LIBRARY_PATH='/opt/xilinx/xrt/lib', OPENBLAS_NUM_THREADS='1')
        controls = {row['route']:row for row in saved['variants']['expanded']['routes']}
        with capture.device_lock():
            for route, extra in (('hybrid', []), ('block_major', ['--block-major'])):
                base.memory_preflight(64)
                capture.original.device_available()
                cmd = [str(cli), '--model', meta['model'], '--kernels', str(kernels),
                       '--ids-file', str(out/'ids.txt'), '--layers', '64', '--max-ctx', str(meta['max_ctx']),
                       '--max-tokens', '2', '--decode-route', 'npu', '--pmode', 'none',
                       '--prefill-logits', '--dump-logits', str(out/route), '--gemm-block', *extra]
                r['routes'][route] = dict(command=cmd, passed=False)
                print('running standard CLI', route, flush=True)
                with (out/(route+'.log')).open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=1800)
                base.check_tokens((out/(route+'.log')).read_text(), meta['expected_tokens'])
                outputs = check_outputs(out, route, len(meta['ids']), meta['vocab'], controls[route]['outputs'])
                scores = []
                for p in range(len(meta['ids'])+1):
                    got = f'{route}_p{p}.bin' if p < len(meta['ids']) else route+'_t1.bin'
                    ref = f'ref_p{p}.bin' if p < len(meta['ids']) else 'ref_cont_logits.bin'
                    scores.append(base.logit_score(base.read(out/got, meta['vocab']),
                                                  base.read(refs/ref, meta['vocab']), meta['real_vocab']))
                if not all(s['passed'] for s in scores):
                    raise ValueError('independent logit gate failed')
                r['routes'][route].update(passed=True, outputs=outputs, logits=scores,
                                          log_sha256=digest(out/(route+'.log')))
                print(route, len(outputs), 'exact captures;', len(scores), 'independent logits pass', flush=True)
                (out/'report.json').write_text(json.dumps(r, indent=2)+'\n')
        base.verify(inputs)
        r.update(complete=True, passed=True)
    except Exception as error:
        r['error'] = str(error)
        raise
    finally:
        (out/'report.json').write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cli', 'kernels', 'baseline', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
