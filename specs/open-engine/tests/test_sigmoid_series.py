"""The production small-argument polynomial against an independent FP64 sigmoid."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from ml_dtypes import bfloat16


def test_small_sigmoid_accuracy_and_real_ffn_boundaries(tmp_path):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler:pytest.skip('C++ compiler required')
    root=Path(__file__).resolve().parents[3]
    src=tmp_path/'sigmoid.cc';lib=tmp_path/'sigmoid.so'
    src.write_text('''#include "sigmoid_series.h"
struct Scalar {
 using V=float;
 static V v(float x){return x;}
 static V add(V a,V b){return a+b;}
 static V mul(V a,V b){return a*b;}
};
extern "C" void run(const float *x,float *out,unsigned n){
 for(unsigned i=0;i<n;++i)out[i]=sigmoid_series<Scalar>(x[i]);
}
''')
    subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-O2','-ffp-contract=off',
                    '-I',str(root/'open_kernels/designs/layer_x'),str(src),'-o',str(lib)],check=True,capture_output=True)
    fn=ctypes.CDLL(str(lib)).run;ptr=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
    fn.argtypes=[ptr,ptr,ctypes.c_uint]
    x=np.r_[np.array([-.11944595724344254,.02902720682322979],np.float32),
            np.linspace(-.5,.5,200001,dtype=np.float32)]
    got=np.zeros_like(x);fn(x,got,len(x));ref=1/(1+np.exp(-x.astype(np.float64)))
    assert np.max(np.abs(got.astype(np.float64)-ref))<4e-8
    assert np.all(np.diff(got[2:])>=0)
    u=np.array([.18203915655612946,.12876082956790924],np.float32)
    h=((x[:2]*got[:2])*u).astype(bfloat16)
    expected=(x[:2].astype(np.float64)*ref[:2]*u.astype(np.float64)).astype(np.float32).astype(bfloat16)
    np.testing.assert_array_equal(h,expected)
