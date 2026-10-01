"""The production integer-lane multiplier must implement IEEE FP32 rounding."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]


def test_integer_product_matches_float32_on_boundaries_and_random_bits(tmp_path):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if compiler is None:pytest.skip('C++ compiler required')
    src=tmp_path/'mul.cc'
    src.write_text('''#include "fp32_mul_rne.h"
struct Scalar {using V=uint32_t;
 static V v(uint32_t x){return x;}
 static V choose(bool c,V a,V b){return c?a:b;}
 static V mul(V a,V b){return a*b;}
};
extern "C" void mul(const uint32_t *a,const uint32_t *b,uint32_t *y,unsigned n){
 for(unsigned i=0;i<n;++i)y[i]=fp32_mul_rne<Scalar>(a[i],b[i]);
}
''')
    lib=tmp_path/'mul.so'
    subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-O2','-I',str(ROOT/'open_kernels/include'),str(src),'-o',str(lib)],check=True,capture_output=True)
    fn=ctypes.CDLL(str(lib)).mul
    rng=np.random.default_rng(786)
    a=rng.integers(0,1<<32,300000,dtype=np.uint32);b=rng.integers(0,1<<32,300000,dtype=np.uint32)
    edges=np.array([0,0x80000000,1,0x80000001,0x7fffff,0x800000,0x80800000,0x3f800000,
        0x3f800001,0x3f7fffff,0x3f000000,0x40000000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001],np.uint32)
    a=np.r_[a,np.repeat(edges,len(edges))];b=np.r_[b,np.tile(edges,len(edges))];got=np.zeros_like(a)
    ptr=np.ctypeslib.ndpointer(np.uint32,flags='C_CONTIGUOUS');fn.argtypes=[ptr,ptr,ptr,ctypes.c_uint]
    fn(a,b,got,len(a))
    with np.errstate(all='ignore'):ref=a.view(np.float32)*b.view(np.float32)
    nan=np.isnan(ref)
    np.testing.assert_array_equal(np.isnan(got.view(np.float32)),nan)
    np.testing.assert_array_equal(got[~nan],ref.view(np.uint32)[~nan])
