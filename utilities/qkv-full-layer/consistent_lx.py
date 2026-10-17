"""Private full-lx affine-sum experiment, preserving production factored recurrence."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LAYER = ROOT / 'open_kernels/designs/layer_x'
GEMV = LAYER.parent / 'gemv_q4'
for path in ('open_kernels', 'open_kernels/model', 'utilities/qkv-precision', 'utilities/deltanet-replay'):
    sys.path.insert(0, str(ROOT / path))
from qkv_build import consistent_header, digest
from full_layer_common import geometry
from recipes.cache import build_key
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes.qwen35 import recipe
from decode_bundle import load_bundle

# Verify the compiler's actual include resolution, including the unchanged
# compensated recurrence. IRON copies TUs, so quoted includes alone are unsafe.
DEPENDENCIES = {
    name: [('gemv_tab.h', True), ('gemv_q4.h', True)]
    for name in ('dense_prep_f32', 'dense_prep', 'gemv_q4_gy', 'gemv_q4_gms')
}
DEPENDENCIES.update({
    'dnx_' + name: [('dnx_factored.h', False), ('dnx_precise.h', False)]
    for name in ('vcopy', 'pass1', 'delta', 'row', 'ofin')
})


def generated_sources(spec):
    geometry(spec)
    r = recipe(spec)
    if not r.linear.FACTORED_READOUT or r.q8:
        raise ValueError('requires the validated Q4 factored production recipe')
    loader = importlib.util.spec_from_file_location('_consistent_lx_gen', LAYER / 'gen_kernels.py')
    gen = importlib.util.module_from_spec(loader); loader.loader.exec_module(gen)
    return {**gen.files(r), 'gemv_q4.h': (GEMV/'gemv_q4.h').read_text(),
            'gemv_tab.h': consistent_header((GEMV/'gemv_tab.h').read_text())}


def source_hash(spec):
    h = hashlib.sha256(build_key(spec).encode())
    for name, code in sorted(generated_sources(spec).items()):
        h.update(name.encode() + b'\0' + code.encode())
    h.update(Path(__file__).read_bytes())
    h.update((LAYER/'lx.py').read_bytes())
    return h.hexdigest()


def private_kernels(X, inc, t, directory):
    original = X.HERE
    try:
        X.HERE = directory
        return X.kernels([str(directory), *inc, str(original)], t)
    finally:
        X.HERE = original


def render_design(spec, source=None):
    source = (LAYER/'lx.py').read_text() if source is None else source
    changes = [
        ('HERE = Path(__file__).parent', f'HERE = Path({str(LAYER)!r})'),
        ('import linear_fp32 as LF', 'import linear_fp32 as LF\n' +
         f'sys.path.insert(0, {str(HERE)!r})\nfrom consistent_lx import private_kernels'),
        ('K = X.kernels(inc, t)', 'K = private_kernels(X, inc, t, Path(__file__).parent / "generated")'),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('full-layer source anchor changed: ' + old)
        source = source.replace(old, new)
    return source + f'\nSPECIALIZE["srchash"] = {int(source_hash(spec)[:8], 16)}\n'


def verify_generated(directory, spec):
    expected = generated_sources(spec)
    if {p.name for p in directory.iterdir() if p.is_file()} != set(expected):
        raise ValueError('generated source coverage mismatch')
    for name, code in expected.items():
        if (directory/name).read_text() != code:
            raise ValueError('stale generated source: ' + name)
    return {name: digest(directory/name) for name in expected}


def verify_dependencies(build):
    result = {}
    for obj, headers in DEPENDENCIES.items():
        path = build/'final.prj'/(obj + '.o.d')
        deps = shlex.split(path.read_text().replace('\\\n', ''))
        for header, private in headers:
            expected = (build/'generated' if private else LAYER)/header
            if str(expected) not in deps:
                raise ValueError(f'compiler used wrong header: {obj}: {header}')
        result[path.name] = digest(path)
    return result


def verify_build(build, spec, bundle, kernels):
    b = json.loads((build/'build.json').read_text())
    if (b['production_build_key'] != bundle['build_key'] or b['geometry'] != geometry(spec)
            or b['source_hash'] != source_hash(spec)):
        raise ValueError('invalid private build identity')
    if set(b['sha256']) != {'final.xclbin', 'insts.bin'}:
        raise ValueError('incomplete binary coverage')
    for name, checksum in b['sha256'].items():
        if digest(build/name) != checksum: raise ValueError('changed private binary')
    if b['mode'] == 'control':
        if any(digest(kernels/'lx'/n) != h for n, h in b['sha256'].items()):
            raise ValueError('changed production control')
    elif b['mode'] == 'consistent':
        if (verify_generated(build/'generated', spec) != b['generated_sha256']
                or verify_dependencies(build) != b['dependency_sha256']
                or digest(build/'full_lx.py') != b['design_sha256']):
            raise ValueError('changed compilation')
    else:
        raise ValueError('unknown private mode')
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('kernel-dir', 'out'): ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--mode', choices=['control', 'consistent'], required=True)
    a = ap.parse_args(); a.out = a.out.resolve(); a.kernel_dir = a.kernel_dir.resolve()
    if (not a.out.is_relative_to(ROOT/'Models') or a.out == ROOT/'Models'
            or a.out.is_relative_to(a.kernel_dir) or a.kernel_dir.is_relative_to(a.out)):
        raise ValueError('separate private build under Models required')
    spec = load_spec(a.kernel_dir/'spec.json')
    old = json.loads((a.kernel_dir/'manifest.json').read_text())
    bundle = load_bundle(a.kernel_dir, manifest(spec, max_ctx=old['max_ctx_default']))
    a.out.mkdir(parents=True, exist_ok=True); record = a.out/'build.json'; record.unlink(missing_ok=True)
    meta = dict(mode=a.mode, geometry=geometry(spec), production_build_key=bundle['build_key'], source_hash=source_hash(spec))
    if a.mode == 'control':
        for name in ('final.xclbin', 'insts.bin'): shutil.copyfile(a.kernel_dir/'lx'/name, a.out/name)
    else:
        generated = a.out/'generated'; generated.mkdir(exist_ok=True)
        for name, code in generated_sources(spec).items(): (generated/name).write_text(code)
        meta['generated_sha256'] = verify_generated(generated, spec)
        design = a.out/'full_lx.py'; design.write_text(render_design(spec))
        meta['design_sha256'] = digest(design)
        (a.out/'attempt.json').write_text(json.dumps(meta, indent=2)+'\n')
        subprocess.run([sys.executable, str(ROOT/'open_kernels/build_design.py'), str(design), str(a.out)],
                       env=dict(os.environ, OPEN_KERNELS_SPEC=str(a.kernel_dir/'spec.json')), check=True)
        if (verify_generated(generated, spec) != meta['generated_sha256']
                or digest(design) != meta['design_sha256'] or source_hash(spec) != meta['source_hash']):
            raise ValueError('sources changed during build')
        meta['dependency_sha256'] = verify_dependencies(a.out)
    meta['sha256'] = {n: digest(a.out/n) for n in ('final.xclbin', 'insts.bin')}
    record.write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__': main()
