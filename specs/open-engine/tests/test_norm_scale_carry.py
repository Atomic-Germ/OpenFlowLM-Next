"""Exercise the shared scale algorithm, including the real layer5 BF16 boundary."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[3]


def test_compensated_scale_preserves_weight_product_residual(tmp_path):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        pytest.skip('C++ compiler required')
    source = tmp_path / 'scale.cc'
    source.write_text('''#include <cstring>
#include "fp32_scale_carry.h"
struct Scalar {
  using V = float;
  static V add(V a,V b) { return a+b; }
  static V sub(V a,V b) { return a-b; }
  static V mul(V a,V b) { return a*b; }
  static V high(V a) {
    unsigned u; std::memcpy(&u,&a,4); u &= 0xfffff000;
    std::memcpy(&a,&u,4); return a;
  }
};
extern "C" void scale(const float *x,const float *w,const float *r,float *out,unsigned n) {
  for(unsigned i=0;i<n;++i) out[i]=fp32_scale_carry<Scalar>(x[i],w[i],r[i]);
}
''')
    lib = tmp_path / 'scale.so'
    subprocess.run([compiler, '-std=c++17', '-shared', '-fPIC', '-O2',
                    '-ffp-contract=off', '-I', str(ROOT/'open_kernels/include'),
                    str(source), '-o', str(lib)], check=True, capture_output=True)
    fn = ctypes.CDLL(str(lib)).scale
    ptr = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    fn.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_uint]
    rng = np.random.default_rng(3295)
    x = np.ldexp(rng.uniform(-1, 1, 200000), rng.integers(-20, 21, 200000)).astype(np.float32)
    w = np.ldexp(rng.uniform(-1, 1, 200000), rng.integers(-4, 5, 200000)).astype(bfloat16).astype(np.float32)
    r = np.ldexp(rng.uniform(.5, 1, 200000), rng.integers(-20, 21, 200000)).astype(np.float32)
    # Both signs of the captured boundary, with the device's unchanged inv.
    x = np.r_[np.float32(.028717929497361183), np.float32(-.028717929497361183), x]
    w = np.r_[np.float32(1.078125), np.float32(1.078125), w]
    r = np.r_[np.float32(.8180990815162659), np.float32(.8180990815162659), r]
    expected = (x.astype(np.longdouble)*w.astype(np.longdouble)*r.astype(np.longdouble)).astype(np.float32)
    old = (x*w)*r
    assert old[0].astype(bfloat16) != expected[0].astype(bfloat16)
    got = np.zeros_like(x)
    fn(x, w, r, got, len(x))
    np.testing.assert_array_equal(got.astype(bfloat16), expected.astype(bfloat16))
    np.testing.assert_array_equal(got[:2], expected[:2])
    # Compensation targets the BF16 boundary, not a universal correctly-rounded
    # triple-product contract; require FP32 accuracy as well as exact BF16 here.
    np.testing.assert_array_max_ulp(got, expected, maxulp=1)
