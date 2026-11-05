"""Pin three fresh builds and copy unchanged independent references, without NPU use."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load('_combined_build', Path(__file__).with_name('build.py'))
history = load('_combined_history', ROOT/'utilities/prefill-out-replay/check.py')
audit = load('_combined_audit', ROOT/'utilities/prefill-weighted-cli/audit.py')
base, digest = audit.base, builder.digest


def check_manifest(record, spec):
    from recipes.manifest import manifest
    expected = manifest(spec, max_ctx=record['max_ctx_default'], key=record['build_key'])
    if expected != record:
        raise ValueError('retained manifest differs from current model-derived recipe')
    return expected


def verify_build(directory, variant):
    record = audit.trace.verify_build(directory, variant)
    if not record.get('compiler_dependencies'):
        raise ValueError('missing CLI compiler dependency inventory')
    base.verify(record['compiler_dependencies'])
    base.verify(record['kernel_inputs'])
    roles = set() if variant == 'paired' else {'ffn'} if variant == 'weighted' else set(builder.SHAPES)
    if set(record['low_kernels']) != roles:
        raise ValueError('missing or unexpected low-weight kernels')
    for role, name in record['low_kernels'].items():
        path = Path(name)
        if record['kernel_inputs'].get(str(path/'build.json')) != digest(path/'build.json'):
            raise ValueError('unbound low-weight build')
        builder.verify_kernel(path, role)
    return record


def baseline_inventory(directory):
    """Verify old captures as historical inputs; do not rewrite their source hashes."""
    meta = json.loads((directory/'reference.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    base.check_reference_coverage(meta)
    if meta['layers'] != 64 or meta['hidden'] != 5120 or len(meta['ids']) != 20:
        raise ValueError('expected full-depth 20-token reference')
    if report['reference_sha256'] != digest(directory/'reference.json'):
        raise ValueError('unbound historical reference')
    build_path = directory.parent/'build-evidence.json'
    old_build = json.loads(build_path.read_text())
    if not old_build['complete'] or old_build['cli_sha256'] != report['cli_sha256']:
        raise ValueError('historical captures are not bound to a complete native build')
    graph = dict(reference=history.history_graph(directory/'reference.json'),
                 native_build=history.history_graph(build_path))
    inputs = {str(directory/name): sha for name, sha in meta['references'].items()}
    inputs[str(directory/'ids.txt')] = meta['ids_sha256']
    routes = ('sequential', 'hybrid', 'block_major')
    if [r['route'] for r in report['routes']] != list(routes):
        raise ValueError('incomplete historical routes')
    for result in report['routes']:
        route = result['route']
        names = set(base.expected_dumps(route, len(meta['ids']))+[route+'-act.bin'])
        if set(result['outputs']) != names:
            raise ValueError('incomplete historical capture inventory')
        inputs.update({str(directory/name): sha for name, sha in result['outputs'].items()})
        inputs[str(directory/(route+'.log'))] = result['log_sha256']
    base.verify(inputs)
    inputs.update({str(directory/name): digest(directory/name) for name in ('report.json', 'reference.json')})
    return meta, graph, inputs


def prepare(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, native_validated=False, model_acceptance=False,
                  builds={}, inputs={}, artifacts={}, history={})
    try:
        for variant in ('paired', 'weighted', 'combined'):
            directory = a.builds.resolve()/variant
            record = verify_build(directory, variant)
            result['builds'][variant] = dict(directory=str(directory), cli_sha256=record['cli_sha256'])
            result['inputs'][str(directory/'build-evidence.json')] = digest(directory/'build-evidence.json')
        metas = {}
        for variant in ('paired', 'weighted'):
            directory = getattr(a, variant+'_baseline').resolve()
            meta, graph, inputs = baseline_inventory(directory)
            metas[variant] = meta
            result['history'][variant] = graph
            result['inputs'].update(inputs)
            if variant == 'weighted':
                comparison_path = directory.parent.parent/'comparison.json'
                comparison = json.loads(comparison_path.read_text())
                if not comparison['capture_valid']:
                    raise ValueError('invalid retained weighted control')
                residuals = comparison['trace_hashes']['candidate']
                expected = {f'{r}_res{l}.bin' for r in audit.trace.ROUTES for l in range(64)}
                if set(residuals) != expected:
                    raise ValueError('incomplete retained weighted residuals')
                hashes = {str(directory/name): sha for name, sha in residuals.items()}
                base.verify(hashes)
                result['inputs'].update(hashes)
                result['inputs'][str(comparison_path)] = digest(comparison_path)
        if metas['paired'] != metas['weighted']:
            raise ValueError('control baselines use different independent references')
        meta = metas['paired']
        # Model-specific real_vocab is derived from its tokenizer, not the
        # rounded catalogue default. Compare the entire retained manifest.
        model = Path(meta['model'])
        manifest_path = Path(meta['kernels'])/'manifest.json'
        manifest_record = json.loads(manifest_path.read_text())
        spec = base.spec_from_model_dir(model)
        check_manifest(manifest_record, spec)
        if manifest_record['spec_hash'] != meta['spec_hash'] or manifest_record['build_key'] != meta['build_key']:
            raise ValueError('manifest is not the retained reference export')
        for path in (manifest_path, model/'config.json', model/'tokenizer.json'):
            result['inputs'][str(path)] = digest(path)
        for path in (Path(__file__), Path(builder.__file__), Path(history.__file__),
                     Path(audit.__file__), Path(base.__file__)):
            result['inputs'][str(path.resolve())] = digest(path)
        refs = out/'references'
        refs.mkdir()
        for name in ('reference.json', 'ids.txt', *meta['references']):
            path = refs/name
            shutil.copy2(a.paired_baseline/name, path)
            result['artifacts'][str(path)] = digest(path)
        # The reference metadata and arrays remain byte-identical. The separate
        # history record binds old source identity to current tracked sources.
        base.verify(result['inputs'])
        result.update(complete=True, manifest_matches_current=True,
                      reference_files=len(meta['references']),
                      scope='build and host preparation only; no model weights loaded',
                      pending=['fresh paired and weighted native baseline identity',
                               'combined full-depth residual/logit gates',
                               'repeated-request/reset instrumentation and execution'])
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        (out/'preparation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'builds', 'paired-baseline', 'weighted-baseline'):
        parser.add_argument('--'+name, type=Path, required=True)
    raise SystemExit(prepare(parser.parse_args()))
