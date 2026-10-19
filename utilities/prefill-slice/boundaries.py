"""Audit full-depth 255/256/257 references and their retained four-layer prefixes."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re

s = importlib.util.spec_from_file_location('_boundary_reference_base', Path(__file__).with_name('check.py'))
base = importlib.util.module_from_spec(s); s.loader.exec_module(base)
FIELDS = ('hidden', 'vocab', 'real_vocab', 'model', 'kernels', 'build_key', 'spec_hash', 'max_ctx', 'message')


def compatible(a, b):
    if any(a[k] != b[k] for k in FIELDS):
        raise ValueError('boundary reference geometry/provenance differs')


def exact_prefix(short, long, short_count, long_count):
    a = base.read(short, short_count).astype('<f4').tobytes()
    b = base.read(long, long_count).astype('<f4').tobytes()
    if a != b[:len(a)]:
        raise ValueError(f'causal prefix differs: {short} / {long}')


def compare_prefix(short, sm, long, lm):
    compatible(sm, lm)
    n, full = len(sm['ids']), len(lm['ids'])
    if sm['layers'] != lm['layers'] or not n < full or sm['ids'] != lm['ids'][:n]:
        raise ValueError('layers or prompt prefix IDs differ')
    for layer in range(sm['layers']):
        name = f'ref_res{layer}.bin'
        exact_prefix(short / name, long / name, n * sm['hidden'], full * sm['hidden'])
    for pos in range(n):
        name = f'ref_p{pos}.bin'
        exact_prefix(short / name, long / name, sm['vocab'], sm['vocab'])
    return dict(residual_arrays=sm['layers'], logit_vectors=n)


def compare_slice(full, fm, small, sm):
    compatible(fm, sm)
    if fm['layers'] != 64 or sm['layers'] != 4 or fm['ids'] != sm['ids']:
        raise ValueError('requires matching full64 and four-layer slice prompt')
    # Different tails select different continuation IDs. Only prompt layers 0..3
    # can be identical; comparing continuation state across depths is invalid.
    for layer in range(4):
        name = f'ref_res{layer}.bin'; count = len(sm['ids']) * sm['hidden']
        exact_prefix(full / name, small / name, count, count)
    return 4


def validate_reference(directory, layers, tokens):
    meta = json.loads((directory / 'reference.json').read_text())
    base.check_reference_coverage(meta)
    if meta['layers'] != layers or len(meta['ids']) != tokens or meta['prompt_kind'] != 'chat_prefix':
        raise ValueError('incorrect boundary length/depth/kind')
    if base.digest(directory / 'ids.txt') != meta['ids_sha256']:
        raise ValueError('changed prompt IDs file')
    ids = [int(x) for x in (directory / 'ids.txt').read_text().strip().split(',')]
    if ids != meta['ids'] or any(not 0 <= x < meta['real_vocab'] for x in ids):
        raise ValueError('prompt IDs differ from metadata')
    for name, sha in meta['references'].items():
        residual = re.fullmatch(r'ref_(cont_)?res\d+\.bin', name)
        count = meta['hidden'] * (1 if residual[1] else tokens) if residual else meta['vocab']
        base.read(directory / name, count, sha)
    last = base.read(directory / f'ref_p{tokens-1}.bin', meta['vocab'])
    cont = base.read(directory / 'ref_cont_logits.bin', meta['vocab'])
    if meta['expected_tokens'] != [int(x[:meta['real_vocab']].argmax()) for x in (last, cont)]:
        raise ValueError('continuation IDs were not selected by CPU logits')
    return meta


def native_prefixes(reports):
    lengths = sorted(reports)
    pairs = {}
    for n, longer in zip(lengths, lengths[1:]):
        before = {r['route']: r for r in reports[n]['routes']}
        after = {r['route']: r for r in reports[longer]['routes']}
        for route in ('sequential', 'hybrid', 'block_major'):
            for pos in range(n):
                name = f'{route}_p{pos}.bin'
                pairs[f'{n}/{longer}/{name}'] = before[route]['outputs'][name] == after[route]['outputs'][name]
    return pairs


def audit(a):
    out = a.out.resolve(); native = getattr(a, 'cli', None) is not None
    report_path = out / ('native-audit.json' if native else 'reference-audit.json')
    report = dict(passed=False, native_acceptance=False,
                  scope='independent full-depth boundary references only')
    try:
        cases = {n: validate_reference(out / str(n), 64, n) for n in (255, 256, 257)}
        sources = {}
        for meta in cases.values():
            for p, sha in meta['sources'].items():
                if p in sources and sources[p] != sha:
                    raise ValueError('boundary cases have different source provenance')
                sources[p] = sha
        base.verify(sources)
        first = cases[255]
        spec = base.spec_from_model_dir(Path(first['model']))
        bundle = base.load_bundle(Path(first['kernels']), base.manifest(spec, max_ctx=first['max_ctx']))
        if bundle['build_key'] != first['build_key'] or bundle['spec_hash'] != first['spec_hash']:
            raise ValueError('reference/export identity differs')
        prefixes = {f'{n}/{n+1}': compare_prefix(out / str(n), cases[n], out / str(n+1), cases[n+1])
                    for n in (255, 256)}
        slices = {}; evidence = {str(Path(__file__).resolve()): base.digest(__file__)}
        for n, meta in cases.items():
            small = a.slice.resolve() / str(n)
            sm = validate_reference(small, 4, n)
            # The earlier checker acquired full-depth support later. Allow that
            # source hash to differ; require identical oracle/packing/recipe inputs.
            if set(meta['sources']) != set(sm['sources']):
                raise ValueError('slice/reference source coverage differs')
            for p, sha in sm['sources'].items():
                if Path(p).resolve() != Path(base.__file__).resolve() and meta['sources'][p] != sha:
                    raise ValueError(f'slice/reference source changed: {p}')
            slices[str(n)] = compare_slice(out / str(n), meta, small, sm)
            for d in (out / str(n), small):
                evidence[str(d / 'reference.json')] = base.digest(d / 'reference.json')
        report.update(passed=True, references=sum(len(m['references']) for m in cases.values()),
                      prompt_prefixes=prefixes, four_layer_prompt_arrays=slices,
                      expected_tokens={str(n): m['expected_tokens'] for n, m in cases.items()},
                      build_key=first['build_key'], spec_hash=first['spec_hash'], evidence=evidence)
        if native:
            path = Path(__file__).resolve().parents[1] / 'prefill-paired-input/audit.py'
            loader = importlib.util.spec_from_file_location('_boundary_native_audit', path)
            module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
            reports = {}
            cli_hash = base.digest(a.cli)
            for n in cases:
                _, r = module.validate_run(out / str(n))
                if r['cli_sha256'] != cli_hash:
                    raise ValueError('boundary native runs used a different CLI')
                reports[n] = r
                evidence[str(out / str(n) / 'report.json')] = base.digest(out / str(n) / 'report.json')
            pairs = native_prefixes(reports)
            passed = all(r['passed'] for r in reports.values()) and all(pairs.values())
            evidence[str(path)] = base.digest(path)
            report.update(passed=passed, native_acceptance=passed, cli_sha256=cli_hash,
                          scope='64-layer 255/256/257 causal prefill and final-layer NPU continuation',
                          native_prompt_prefixes=pairs,
                          native_cases={str(n): r['passed'] for n, r in reports.items()})
    except Exception as e:
        report.update(passed=False, native_acceptance=False, error=str(e))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--slice', type=Path, required=True)
    p.add_argument('--cli', type=Path, help='also audit existing native runs from this CLI; does not launch NPU')
    raise SystemExit(audit(p.parse_args()))
