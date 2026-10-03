"""Post replay must expose BF16 boundary errors without changing acceptance."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from ml_dtypes import bfloat16

ROOT=Path(__file__).resolve().parents[3]


def module():
    s=importlib.util.spec_from_file_location('post_probe',ROOT/'utilities/test-wide-post.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_reference_uses_per_head_norm_and_fp32_before_bf16():
    m=module();o=np.ones((32,128),np.float32);o[1]*=-2
    z=np.full_like(o,.25);w=np.ones(128,bfloat16)
    r=m.reference(o,z,w)
    assert r.dtype==np.dtype(bfloat16)
    expected=np.float32(.25/(1+np.exp(-.25))/np.sqrt(1+1e-6))
    assert r[0,0]==bfloat16(expected) and r[1,0]<0


def test_fixture_repeat_exact_gate_and_hashes(tmp_path):
    m=module();o=np.full((32,128),.01,np.float32);z=np.full_like(o,.25);w=np.ones(128,bfloat16)
    kernel=tmp_path/'kernel';kernel.mkdir()
    for n in ('final.xclbin','insts.bin'):(kernel/n).write_bytes(b'kernel')
    out=tmp_path/'probe'
    m.write_fixture(out,kernel,[(o,z,w)],trace=False,diagnostic=True)
    expected=m.reference(o,z,w)
    def writes(a):
        for i in range(2):(out/f'got0-{i}.bin').write_bytes(a.tobytes()+m.GUARD)
    writes(expected)
    assert m.compare(out,True)==0
    import json
    r=json.loads((out/'post-results.json').read_text());assert r['diagnostic_only']
    bad=expected.copy();bad.view(np.uint16)[0,0]+=1;writes(bad)
    assert m.compare(out,True)==1
    (out/'got0-1.bin').write_bytes(expected.tobytes()+m.GUARD)
    assert m.compare(out,False)==1
    (out/'got0-1.bin').write_bytes(expected.tobytes()+bytes(64))
    with pytest.raises(ValueError,match='canary'):m.compare(out,False)
    (out/'w0.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):m.compare(out,False)


def test_trace_must_reconstruct_actual_bf16_output(tmp_path):
    m=module();o=np.full((32,128),.01,np.float32);z=np.full_like(o,.25);w=np.ones(128,bfloat16)
    k=tmp_path/'k';k.mkdir()
    for n in ('final.xclbin','insts.bin'):(k/n).write_bytes(b'kernel')
    out=tmp_path/'p';m.write_fixture(out,k,[(o,z,w)],trace=True,diagnostic=True)
    r=m.reference(o,z,w)
    for i in range(2):
        (out/f'got0-{i}.bin').write_bytes(r.tobytes()+m.GUARD)
        (out/f'trace0-{i}.bin').write_bytes(np.zeros(4096*6,np.float32).tobytes()+m.GUARD)
    assert m.compare(out,True)==1
    tr=np.zeros((4,6,1024),np.float32);tr[:,5]=r.astype(np.float32).reshape(4,1024)
    for i in range(2):(out/f'trace0-{i}.bin').write_bytes(tr.tobytes()+m.GUARD)
    assert m.compare(out,True)==0


def test_compensated_five_factor_product_closes_real_boundary(tmp_path):
    import subprocess
    source=tmp_path/'test.cpp';exe=tmp_path/'test'
    source.write_text(r'''
#include <cmath>
#include <cstdio>
#include "fp32_product_chain.h"
struct O {
  using V=float;
  static V add(V a,V b){return a+b;}
  static V sub(V a,V b){return a-b;}
  static V mul(V a,V b){return a*b;}
  static V high(V a){ union {float f; unsigned u;} v{a};v.u&=0xfffff000u;return v.f; }
};
int main(){
  float x,i,w,g,r;
  while(scanf("%f %f %f %f %f",&x,&i,&w,&g,&r)==5)
    printf("%a\n",fp32_product_chain<O>(x,i,w,g,r));
}
''')
    subprocess.run(['g++','-O2','-ffp-contract=off','-I'+str(ROOT/'open_kernels/include'),str(source),'-o',str(exe)],check=True)
    rng=np.random.default_rng(604)
    values=rng.uniform(-2,2,(10000,5)).astype(np.float32)
    values[0]=[-2.351502644160064e-6,997.9894409179688,.9296875,-.29373395442962646,.42708998918533325]
    inputs='\n'.join(' '.join(map(str,row)) for row in values)+'\n'
    result=subprocess.run([str(exe)],input=inputs,text=True,capture_output=True,check=True)
    got=np.array([float.fromhex(v) for v in result.stdout.splitlines()],np.float32)
    expected=np.prod(values.astype(np.float64),axis=1).astype(np.float32)
    np.testing.assert_array_equal(got,expected)
    assert bfloat16(got[0])==bfloat16(.0002727508544921875)
    sequential=np.prod(values[0],dtype=np.float32)
    assert bfloat16(sequential)!=bfloat16(got[0])
