"""Integrated prefill precision: fail-closed manifests and exact four-product sums."""
import json
from pathlib import Path
import subprocess
import pytest
from recipes.load import load_spec
from recipes.manifest import manifest

ROOT = Path(__file__).resolve().parents[3]


def paired_manifest():
    m = manifest(load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json'))
    m['manifest_version'] = 4
    for lt in m['layer_types'].values():
        g = lt['gemm_block']
        g['low_kernels'] = {}
        for s in g['program'] + g['ffn_program']:
            low = s['kernel'] + '_low'
            g['low_kernels'][s['kernel']] = low
            m['contexts'][low] = low + '/final.xclbin'
            m['kernels'][low] = dict(context=low, insts=low+'/insts.bin', weight_component='bf16_residual')
    return m


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    p = tmp_path_factory.mktemp('precision-runtime')
    source = p/'test.cpp'
    source.write_text(r'''
#include "open_qwen36/manifest.hpp"
#include "open_qwen36/prefill_precision.hpp"
#include <iostream>
using namespace open_qwen36;
int main(int argc, char** argv) {
 try {
  if (argc > 1) { auto m=Manifest::load(argv[1]);
   for (const auto& [name,t]:m.layer_types)
    for (const auto& [high,low]:t.gemm_block.low_kernels)
     if (!m.kernels.count(low)) return 2;
   return 0;
  }
  prefill_precision::Workspace w;
  float input[]={1.00390625f,-2.0078125f};
  float shared[2]; int call=0;
  auto dispatch=[&](bool low,const float* x) {
   if (low != (call>=2)) throw std::runtime_error("weight ordering");
   const float expected=(call%2==0)?input[0]:input[0]-bf16_to_f32(f32_to_bf16(input[0]));
   if (x[0]!=expected) throw std::runtime_error("input split");
   // Strong cancellation distinguishes (HH+LH)+(HL+LL) from reassociation.
   const float values[]={16777216.f,1.f,-16777216.f,2.f};
   shared[0]=values[call++]; shared[1]=-shared[0]; return shared;
  };
  auto y=w.run(input,2,2,dispatch);
  if(call!=4 || y[0]!=2.f || y[1]!=-2.f) return 3;
  shared[0]=99.f; if(y[0]!=2.f) return 4;
  float bad[]={std::numeric_limits<float>::infinity()};
  try {w.run(bad,1,1,dispatch);return 5;} catch(const std::runtime_error&) {}
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what();return 1;}
}
''')
    exe = p/'test'
    subprocess.run(['c++','-std=c++17','-O2','-I'+str(ROOT/'src'),'-I'+str(ROOT/'src/include'),
                    '-I'+str(ROOT/'open_kernels/harness'),str(source),str(ROOT/'src/open_qwen36/manifest.cpp'),
                    '-o',str(exe)], check=True)
    return exe


def run_manifest(native, tmp_path, m):
    p=tmp_path/'manifest.json'; p.write_text(json.dumps(m))
    return subprocess.run([str(native),str(p)],capture_output=True,text=True)


def test_four_product_order_and_output_ownership(native):
    subprocess.run([str(native)],check=True)


def test_paired_manifest_and_legacy_both_load(native,tmp_path):
    m=paired_manifest()
    assert run_manifest(native,tmp_path,m).returncode == 0
    m=manifest(load_spec(ROOT/'open_kernels/recipes/specs/qwen35-9b.json'))
    assert m['manifest_version'] < 4
    assert run_manifest(native,tmp_path,m).returncode == 0


@pytest.mark.parametrize('fault', ['old_version','missing_low','same_kernel','wrong_component','partial',
                                  'wrong_role','split','epsilon','kv','history','unknown_role','wrong_shape','swapped_low','same_context','same_stream','empty'])
def test_malformed_precision_is_refused(native,tmp_path,fault):
    m=paired_manifest();g=m['layer_types']['linear_attention']['gemm_block']
    high=g['program'][0]['kernel'];low=g['low_kernels'][high]
    if fault=='old_version':m['manifest_version']=3
    elif fault=='missing_low':del m['kernels'][low]
    elif fault=='same_kernel':g['low_kernels'][high]=high
    elif fault=='wrong_component':m['kernels'][low]['weight_component']=''
    elif fault=='partial':del g['low_kernels'][high]
    elif fault=='wrong_role':g['program'][1]['args'][0]='go_w'
    elif fault=='split':g['program'][0]['split']=True
    elif fault=='epsilon':g['eps']=1e-5
    elif fault=='kv':m['layout'].pop('attn_kv_dtype')
    elif fault=='history':g['conv_dtype']='bfloat16'
    elif fault=='unknown_role':g['low_kernels']['undeclared']=low
    elif fault=='wrong_shape':g['vw']=4096
    elif fault=='swapped_low':g['low_kernels'][high]=g['low_kernels'][g['program'][1]['kernel']]
    elif fault=='same_context':m['kernels'][low]['context']=m['kernels'][high]['context']
    elif fault=='same_stream':m['kernels'][low]['insts']=m['kernels'][high]['insts']
    elif fault=='empty':g['low_kernels']={}
    r=run_manifest(native,tmp_path,m)
    assert r.returncode != 0, fault
