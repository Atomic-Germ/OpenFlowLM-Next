"""Paired KV and FP32 output must form one explicit production ABI."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
from recipes.load import load_spec
from recipes import qwen35 as Q
from recipes.manifest import manifest
from attention_heads import read_attention

ROOT=Path(__file__).resolve().parents[3]


def test_export_can_use_a_separate_build_workspace(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'open_kernels/export_qwen36_kernels.py'),
        '--spec',str(ROOT/'open_kernels/recipes/specs/qwen35-27b.json'),
        '--out',str(tmp_path/'export'),'--build-root',str(tmp_path/'work'),
        '--only','ax','--no-build'],capture_output=True,text=True)
    assert p.returncode!=0
    assert str(tmp_path/'work/ax/final.xclbin') in p.stderr and 'missing after build' in p.stderr


def spec27():return load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')


def test_paired_cache_and_float_output_share_the_production_geometry():
    s=spec27();r=Q.recipe(s,257);a=r.attn;l=r.layout;m=manifest(s,max_ctx=257)
    assert a.KV_PAIR and a.HPO==2 and a.OG_ELEMS==6
    assert l.KV_ROW==8192 and l.KV_BYTES==257*8192 and l.E_A==2048
    assert l.AA_OUT-l.AA_OG==6144*4
    assert m['layout']['attn_kv_dtype']=='bfloat16_pair'
    assert m['layout']['attn_output_dtype']=='float32'
    assert m['layer_types']['full_attention']['buffers']['state']['row']==8192


def test_other_attention_shapes_keep_their_cache_abi():
    s=load_spec(ROOT/'open_kernels/recipes/specs/qwen35-9b.json');a=Q.recipe(s).attn
    assert not a.KV_PAIR
    assert Q.layout(s).KV_ROW==4*s.attn_kv_width
    assert 'attn_kv_dtype' not in manifest(s)['layout']


def test_attention_capture_keeps_low_fp32_output_bits(tmp_path):
    o=SimpleNamespace(AA_BYTES=256,AA_QG=0,AA_KVN=64,AA_OG=96)
    b=bytearray(o.AA_BYTES);v=np.full(8,100.2,np.float32);b[96:128]=v.tobytes()
    p=tmp_path/'act';p.write_bytes(b)
    a=read_attention(p,o,2,1,4,output_dtype='float32')
    np.testing.assert_array_equal(a['og'].ravel(),v.astype(np.float64))
    with pytest.raises(ValueError,match='dtype'):read_attention(p,o,2,1,4,output_dtype='float16')


def test_hybrid_cache_preserves_low_planes_across_blocks(tmp_path):
    # Uniform Q/K makes two-position output the independent arithmetic mean of V/2.
    p=tmp_path/'test.cc';p.write_text(r'''
#include "block_host.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>
using namespace open_qwen36;
int main() {
  host::AttnGeom g;g.T=g.t_real=g.nh=g.kvh=1;g.hd=32;g.kv_pair=true;
  std::vector<float> q(32,1),k(32,1),v(32,100.2f),gate(32,0),norm(32,1),out(32);
  std::vector<uint16_t> kv(2*128+16,0x1234);
  host::attention_block(g,q.data(),k.data(),v.data(),gate.data(),norm.data(),norm.data(),nullptr,kv.data(),128,out.data());
  if(std::abs(out[0]-50.1f)>.001f || kv[96]==0) return 1;
  g.pos0=1;v.assign(32,200.4f);
  host::attention_block(g,q.data(),k.data(),v.data(),gate.data(),norm.data(),norm.data(),nullptr,kv.data(),128,out.data());
  if(std::abs(out[0]-75.15f)>.002f) return 2;
  for(int i=256;i<272;++i) if(kv[i]!=0x1234) return 3;
  try { host::attention_prep(g,q.data(),k.data(),v.data(),norm.data(),norm.data(),nullptr,kv.data(),128,out.data()); }
  catch(const std::runtime_error&) { return 0; }
  return 4; // BF16-only NPU block prep must refuse the paired cache.
}
''')
    exe=tmp_path/'test'
    subprocess.run(['g++','-std=c++17','-O2','-mavx2','-mfma','-fopenmp',
                    '-I'+str(ROOT/'src/open_qwen36'),'-I'+str(ROOT/'src'),'-I'+str(ROOT/'src/include'),str(p),str(ROOT/'src/open_qwen36/block_host.cpp'),'-o',str(exe)],check=True,capture_output=True)
    subprocess.run([str(exe)],check=True)
