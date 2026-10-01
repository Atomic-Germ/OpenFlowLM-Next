"""Residual addition must preserve FP32 rounding before the next BF16 boundary."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def probe():
    spec = importlib.util.spec_from_file_location('ln_rne_probe',ROOT/'utilities/test-wide-ln.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_residual_gate_rejects_one_ulp_and_preserves_default(tmp_path):
    m = probe()
    for name in ('final.xclbin','insts.bin'): (tmp_path/name).write_bytes(b'fixture')
    m.prepare(tmp_path,64,1e-6,exact_residual=True)
    import json
    meta=json.loads((tmp_path/'ln-fixture.json').read_text())
    assert meta['exact_residual']
    assert 'real_layer1_3390' in meta['cases']
    for i in range(len(meta['cases'])):
        for t in ('y','xn'):
            (tmp_path/f'got_{t}{i}.bin').write_bytes((tmp_path/f'ref_{t}{i}.bin').read_bytes()+m.GUARD)
    assert m.compare(tmp_path)==0
    i=meta['cases'].index('real_layer1_3390')
    y=np.fromfile(tmp_path/f'ref_y{i}.bin',np.float32)
    y[0]=np.nextafter(y[0],np.float32(-np.inf))
    (tmp_path/f'got_y{i}.bin').write_bytes(y.tobytes()+m.GUARD)
    assert m.compare(tmp_path)==1
    meta['exact_residual']=False
    (tmp_path/'ln-fixture.json').write_text(json.dumps(meta))
    assert m.compare(tmp_path)==0


def test_integer_adder_matches_ieee_float32_on_boundaries_and_random_bits(tmp_path):
    import ctypes,subprocess,shutil
    compiler = shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        pytest.skip('C++ compiler required for production integer-adder test')
    source=tmp_path/'add.cc'
    source.write_text('''#include "fp32_add_rne.h"
struct Scalar { using V=uint32_t;
 static V v(uint32_t x) { return x; }
 static V choose(bool c,V a,V b) { return c?a:b; }
};
extern "C" void add(const uint32_t *a,const uint32_t *b,uint32_t *y,unsigned n) {
 for(unsigned i=0;i<n;++i) y[i]=fp32_add_rne<Scalar>(a[i],b[i]);
}
''')
    library=tmp_path/'add.so'
    subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-O2','-I',str(ROOT/'open_kernels/include'),
                    str(source),'-o',str(library)],check=True,capture_output=True)
    fn=ctypes.CDLL(str(library)).add
    rng=np.random.default_rng(3390)
    a=rng.integers(0,1<<32,200000,dtype=np.uint32)
    b=rng.integers(0,1<<32,200000,dtype=np.uint32)
    edges=np.array([0,0x80000000,1,0x80000001,0x7fffff,0x800000,0x807fffff,0x80800000,
                    0x3f800000,0xbf800000,0x33800000,0x33c00000,0x7f7fffff,0xff7fffff,
                    0x7f800000,0xff800000,0x7fc00000],np.uint32)
    a=np.concatenate([a,np.repeat(edges,len(edges))]);b=np.concatenate([b,np.tile(edges,len(edges))])
    got=np.zeros_like(a)
    ptr=np.ctypeslib.ndpointer(np.uint32,flags='C_CONTIGUOUS');fn.argtypes=[ptr,ptr,ptr,ctypes.c_uint]
    fn(a,b,got,len(a))
    with np.errstate(all='ignore'): ref=a.view(np.float32)+b.view(np.float32)
    nan=np.isnan(ref)
    np.testing.assert_array_equal(np.isnan(got.view(np.float32)),nan)
    np.testing.assert_array_equal(got[~nan],ref.view(np.uint32)[~nan])


def test_residual_diagnosis_checks_captured_operands(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'utilities'))
    spec=importlib.util.spec_from_file_location('rounding_diagnosis',ROOT/'utilities/diagnose-wide-model-rounding.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    x=np.array([-0.03440730646252632],np.float32)
    add=np.array([0.005703141447156668],np.float32)
    bad=np.array([-0.028704166412353516],np.float32)
    assert mod.residual_error(x,add,bad)['mismatches']==1
    assert mod.residual_error(x,add,x+add)['mismatches']==0
