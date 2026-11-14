"""Build private control/consistent-sum GEMV probes; production sources stay intact."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GEMV = ROOT/'open_kernels/designs/gemv_q4'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def consistent_header(text):
    anchor = '__attribute__((noinline)) inline void gemv_q4_prep_f32_direct_blocks('
    if text.count(anchor) != 1:
        raise ValueError('direct preparation anchor changed')
    prefix, body = text.split(anchor)
    old = '''    // Sum the original FP32 values, preserving the same reduction tree and
    // BF16 hi/lo ABI as the BF16 preparation path.
    aie::accum<accfloat, 16> a16, b16;
    a16.from_vector(x.template extract<16>(0));
    b16.from_vector(x.template extract<16>(1));'''
    new = '''    // Use the same quantized activation in both Q4 affine terms.
    // Otherwise a constant zero-weight block can produce a nonzero dot product.
    const auto sx = aie::to_float<float>(aie::load_v<32>(xi + kb * 32), s);
    aie::accum<accfloat, 16> a16, b16;
    a16.from_vector(sx.template extract<16>(0));
    b16.from_vector(sx.template extract<16>(1));'''
    if body.count(old) != 1:
        raise ValueError('direct block-sum anchor changed')
    return prefix+anchor+body.replace(old, new)


def sources(mode):
    if mode not in ('control', 'consistent'):
        raise ValueError('unknown projection mode')
    tab = (GEMV/'gemv_tab.h').read_text()
    if mode == 'consistent':
        tab = consistent_header(tab)
    return {'gemv_tab.h': tab, 'gemv_q4.h': (GEMV/'gemv_q4.h').read_text(),
            'prep.cc': '''#include "gemv_tab.h"
extern "C" void prep(const float* x, uint8_t* table, int32_t block) {
  gemv_q4_prep_f32_direct_blocks(x, table, 5120, block * 32, 32);
}
''', 'gemv.cc': '''#define GEMV_PER_CALL 1
#include "gemv_q4.h"
extern "C" void gemv(const uint8_t* w, const uint8_t* table, float* y, int32_t group) {
  gemv_q4_pool_group_rt(w, table, group, y, 40, 2);
}
''', 'design.py': 'import sys\nsys.path.insert(0, '+repr(str(ROOT/'open_kernels'))+')\n'+(HERE/'design.py').read_text()}


def verify_generated(directory, mode):
    expected = sources(mode)
    if {p.name for p in directory.iterdir() if p.is_file()} != set(expected):
        raise ValueError('generated file coverage mismatch')
    for name, text in expected.items():
        if (directory/name).read_text() != text:
            raise ValueError('stale generated source: '+name)
    return {n: digest(directory/n) for n in expected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['control', 'consistent'], required=True)
    a = p.parse_args(); a.out = a.out.resolve()
    if a.out == ROOT or not a.out.is_relative_to(ROOT/'Models'):
        raise ValueError('private build must be under Models')
    a.out.mkdir(parents=True, exist_ok=True)
    record = a.out/'build.json'; record.unlink(missing_ok=True)
    generated = a.out/'generated'; generated.mkdir(exist_ok=True)
    for n, text in sources(a.mode).items():
        (generated/n).write_text(text)
    before = verify_generated(generated, a.mode)
    subprocess.run([sys.executable, str(ROOT/'open_kernels/build_design.py'), str(generated/'design.py'), str(a.out)], check=True)
    if verify_generated(generated, a.mode) != before:
        raise ValueError('sources changed during build')
    deps = {}
    for name in ('prep', 'gemv'):
        path = a.out/'final.prj'/f'{name}.o.d'
        tokens = shlex.split(path.read_text().replace('\\\n', ''))
        if str(generated/'gemv_tab.h') not in tokens:
            raise ValueError('compiler used wrong table header')
        if name == 'gemv' and str(generated/'gemv_q4.h') not in tokens:
            raise ValueError('compiler used wrong GEMV header')
        deps[path.name] = digest(path)
    record.write_text(json.dumps(dict(mode=a.mode, generated_sha256=before, dependency_sha256=deps,
        builder_sha256=digest(__file__), sha256={n: digest(a.out/n) for n in ('final.xclbin', 'insts.bin')}), indent=2)+'\n')


if __name__ == '__main__':
    main()
