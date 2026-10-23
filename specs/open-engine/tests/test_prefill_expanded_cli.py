"""Expanded prefill policy must preserve controls and reject role/ABI drift."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from recipes.load import load_spec
from recipes.manifest import manifest
ROOT=Path(__file__).resolve().parents[3]
TOOL=ROOT/'utilities/prefill-expanded-cli'


def load(name='build'):
    s=importlib.util.spec_from_file_location('expanded_'+name,TOOL/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_native_expanded_roles_and_abi_guards(tmp_path):
    f=tmp_path/'manifest.json';f.write_text(json.dumps(manifest(load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json'))))
    src=tmp_path/'test.cpp';src.write_text(r'''
#include "expanded_policy.hpp"
#include <functional>
using namespace open_qwen36;
void check(bool v) { if (!v) throw std::runtime_error("bad selection"); }
void refuses(std::function<void()> f) { bool failed=false; try {f();} catch(const std::runtime_error&) {failed=true;} check(failed); }
int main(int argc,char** argv) {
 auto m=Manifest::load(argv[1]); expanded::validate_manifest(m);
 auto l=m.layer_types.at("linear_attention").gemm_block;
 auto a=m.layer_types.at("full_attention").gemm_block;
 auto select=[](const GemmBlockProgram& g,const Step& s,size_t k,size_t n) {return expanded::low_kernel(g,s,256,k,n);};
 check(select(l,l.program[0],5120,16384)=="private_entry_low");
 check(select(a,a.program[0],5120,14336)=="private_attention_low");
 for (auto g : {l,a}) {
  check(select(g,g.program[1],6144,5120)=="private_out_low");
  check(select(g,g.ffn_program[0],5120,34816)=="private_ffn_low");
  check(select(g,g.ffn_program[1],17408,5120)=="private_down_low");
 }
 refuses([&]{select(l,a.program[1],6144,5120);});
 refuses([&]{select(a,l.program[1],6144,5120);});
 refuses([&]{select(a,a.program[0],5120,16384);});
 refuses([&]{expanded::low_kernel(a,a.program[0],128,5120,14336);});
 auto bad=a;bad.program[0].args[0]="gqkvz_w";refuses([&]{expanded::validate(bad,5120);});
 bad=a;bad.ffn_program[1].split=true;refuses([&]{expanded::validate(bad,5120);});
 bad=a;bad.eps=1e-5;refuses([&]{expanded::validate(bad,5120);});
 bad=a;bad.act="gelu_tanh";refuses([&]{expanded::validate(bad,5120);});
 bad=a;bad.norm_groups=2;refuses([&]{expanded::validate(bad,5120);});
 bad=a;bad.sandwich=true;refuses([&]{expanded::validate(bad,5120);});
 m.attn_kv_pair=false;refuses([&]{expanded::validate_manifest(m);});
}
''')
    exe=tmp_path/'test'
    subprocess.run(['c++','-std=c++17','-O0','-I'+str(TOOL),'-I'+str(ROOT/'utilities/prefill-combined-cli'),
                    '-I'+str(ROOT/'src'),'-I'+str(ROOT/'src/include'),'-I'+str(ROOT/'open_kernels/harness'),
                    str(src),str(ROOT/'src/open_qwen36/manifest.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe),str(f)],check=True)


def test_control_patch_exact_and_original_dispatch_preserved():
    m=load();cpp=(ROOT/'src/open_qwen36/core.cpp').read_text();hpp=(ROOT/'src/open_qwen36/core.hpp').read_text()
    lows={role:Path('/tmp')/role for role in m.SHAPES}
    assert m.patch(cpp,hpp,'combined',lows)==m.parent.patch(cpp,hpp,'combined',lows)
    c,h=m.patch(cpp,hpp,'expanded',lows)
    start=cpp.index('const float* Core::gemm_run(');end=cpp.index('\nvoid Core::gemm(',start)
    assert cpp[start:end].replace('Core::gemm_run(','Core::gemm_run_once(',1) in c
    assert 'expanded::validate_manifest(*w_)' in c and 'expanded::low_kernel' in c
    for role in m.SHAPES:assert f'/tmp/{role}/final.xclbin' in c
    with pytest.raises(ValueError):m.patch(c,h,'expanded',lows)
    with pytest.raises(ValueError):m.patch(cpp,hpp,'typo',lows)


def test_new_low_geometry_cannot_reuse_linear_entry(monkeypatch,tmp_path):
    m=load();r={'environment':dict(GQP_N='14336',GQP_K='5120',GQP_T='256',GQP_FMT='q4_1',GQP_BREUSE='0')}
    monkeypatch.setattr(m.parent.parent.kernels,'verify_build',lambda *args:r)
    assert m.verify_kernel(tmp_path,'attention')==r
    with pytest.raises(ValueError):m.verify_kernel(tmp_path,'entry')
    r['environment']['GQP_N']='5120';r['environment']['GQP_K']='17408'
    assert m.verify_kernel(tmp_path,'down')==r
    r['environment']['GQP_BREUSE']='1'
    with pytest.raises(ValueError):m.verify_kernel(tmp_path,'down')


def test_build_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):load().build(SimpleNamespace(out=tmp_path))


def test_preparation_failure_remains_incomplete(tmp_path):
    m=load('prepare');out=tmp_path/'prepared'
    with pytest.raises(FileNotFoundError):m.prepare(SimpleNamespace(out=out,builds=tmp_path/'missing'))
    r=json.loads((out/'preparation.json').read_text())
    assert not r['complete'] and not r['native_validated'] and not r['model_acceptance']
    with pytest.raises(FileExistsError):m.prepare(SimpleNamespace(out=out))


def test_replay_acceptance_requires_all_gates_and_equal_audit():
    m=load('prepare')
    r=dict(complete=True,control_valid=True,slot_identity=True,projection_passed=True,host_passed=True,conditioned_passed=True,residual_passed=True,model_acceptance=False,dispatches=64)
    m.check_replay(r,r)
    for key in ('complete','control_valid','slot_identity','projection_passed','host_passed','conditioned_passed','residual_passed'):
        bad=r|{key:False}
        with pytest.raises(ValueError):m.check_replay(bad,bad)
    with pytest.raises(ValueError):m.check_replay(r,r|{'dispatches':63})
    with pytest.raises(ValueError):m.check_replay(r|{'dispatches':63},r|{'dispatches':63})


def test_repeat_source_identity_checks_nested_and_extra_files(tmp_path):
    m=load('prepare');one=tmp_path/'one';two=tmp_path/'two'
    for p in (one,two):
        (p/'nested').mkdir(parents=True);(p/'nested/helper.hpp').write_text('same');(p/'cli.cpp').write_text(str(p))
    m.require_repeat_sources(one,two)
    (two/'nested/helper.hpp').write_text('different')
    with pytest.raises(ValueError):m.require_repeat_sources(one,two)
    (two/'nested/helper.hpp').write_text('same');(two/'extra.cpp').write_text('extra')
    with pytest.raises(ValueError):m.require_repeat_sources(one,two)
