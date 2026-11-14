"""Private integer FP32 addition must reproduce host RNE, including cancellation."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def add(tmp_path_factory):
    header=ROOT/'utilities/l3-refined-ax/add_rne.h'
    assert header.is_file(), 'missing exact FP32 sum helper'
    compiler=shutil.which('g++')
    if compiler is None:pytest.skip('host C++ compiler unavailable')
    out=tmp_path_factory.mktemp('refined-add');src=out/'bridge.cc'
    src.write_text('#include "add_rne.h"\nextern "C" void add(const uint32_t *a,const uint32_t *b,uint32_t *c,unsigned n){for(unsigned i=0;i<n;++i)c[i]=add_rne(a[i],b[i]);}\n')
    subprocess.run([compiler,'-O2','-shared','-fPIC','-I',str(header.parent),str(src),'-o',str(out/'add.so')],check=True,capture_output=True)
    lib=ctypes.CDLL(str(out/'add.so'));fn=lib.add
    ptr=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
    fn.argtypes=[ptr,ptr,ptr,ctypes.c_uint];fn.restype=None
    def apply(a,b):
        a,b=[np.ascontiguousarray(v,dtype='<f4') for v in (a,b)]
        c=np.empty(a.shape,dtype=np.uint32);fn(a.view(np.uint32),b.view(np.uint32),c,c.size)
        with np.errstate(over='ignore',invalid='ignore'):expected=np.add(a,b,dtype=np.float32)
        assert np.array_equal(c,expected.view(np.uint32))
    return apply


def test_add_cancellation_rounding_ties_and_signed_zero(add):
    a=np.array([0.,-0.,1.,-1.,1.,1.,np.nextafter(np.float32(1),np.float32(2)),0.012217708,-5.7775527e-5],dtype='f4')
    b=np.array([-0.,-0.,-1.,1.,2**-24,-2**-25,2**-24,6.128612e-5,7.204815e-5],dtype='f4')
    add(a,b)


def test_add_random_finite_and_subnormal_operands(add):
    rng=np.random.default_rng(38)
    a=rng.integers(0,2**32,size=300000,dtype=np.uint32)
    b=rng.integers(0,2**32,size=300000,dtype=np.uint32)
    good=((a&0x7f800000)!=0x7f800000)&((b&0x7f800000)!=0x7f800000)
    add(a[good].view('<f4'),b[good].view('<f4'))
    tiny=np.arange(1,8192,dtype=np.uint32).view('<f4')
    add(tiny,-tiny[::-1]);add(tiny,tiny)
