"""Combined instrumentation must preserve arithmetic and complete causal evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT/'utilities/prefill-combined-sublayers'


def load(name):
    s=importlib.util.spec_from_file_location('combined_sublayers_'+name,TOOL/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_patch_preserves_combined_dispatch_and_host_calls():
    m=load('build');b=m.runner.prep.builder
    cpp,hpp=b.patch((ROOT/'src/open_qwen36/core.cpp').read_text(),(ROOT/'src/open_qwen36/core.hpp').read_text(),'combined',b.DEFAULTS)
    cpp,cli=b.parent.trace.patch(cpp,(ROOT/'src/open_qwen36/cli.cpp').read_text())
    result,entry=m.patch(cpp,cli)
    start=cpp.index('const float* Core::gemm_run(');end=cpp.index('\nvoid Core::gemm(',start)
    assert cpp[start:end] in result
    assert result.count('host::attention_block(')==cpp.count('host::attention_block(')
    assert result.count('host::deltanet_block(')==cpp.count('host::deltanet_block(')
    for key in ('q','k','v','gate','kv_after','qkv','state_before','state_after'):
        assert f'"{key}"' in result
    assert str(TOOL/'capture.hpp') in entry
    with pytest.raises(ValueError):m.patch(result,entry)


def test_real_prefix_capture_and_target_selection(tmp_path):
    cpp=tmp_path/'probe.cpp';cpp.write_text(r'''
#include "capture.hpp"
using namespace open_qwen36::sublayer_trace;
int main(int argc,char**argv) {
 prefix=argv[1];float x[40]={};for(int i=0;i<40;++i)x[i]=i;
 rows("q",63,0,20,2,x);row("res",63,0,20,2,x);
 row("res",42,0,20,2,x);row("res",52,0,20,2,x);
 row("res",53,0,20,2,x);
 try {rows("q",63,0,20,2,x);return 1;}catch(const std::runtime_error&){}
 try {rows("k",63,1,19,2,x);return 2;}catch(const std::runtime_error&){}
}
''')
    exe=tmp_path/'probe'
    subprocess.run(['c++','-std=c++17','-I',str(TOOL),str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe),str(tmp_path/'run')],check=True)
    assert {p.name for p in tmp_path.glob('*.bin')}=={'run_L63_P1_q.bin','run_L63_P1_res.bin','run_L42_P17_res.bin','run_L52_P17_res.bin'}
    np.testing.assert_array_equal(np.fromfile(tmp_path/'run_L63_P1_res.bin','<f4'),[2,3])


@pytest.mark.parametrize('problem',['missing','extra','old'])
def test_capture_inventory_rejects_incomplete_or_stale_data(tmp_path,problem):
    m=load('check');names=m.expected_captures();assert len(names)==83
    for n in names:(tmp_path/n).touch()
    m.inventory(tmp_path)
    if problem=='missing':(tmp_path/next(iter(names))).unlink()
    else:(tmp_path/('hybrid_L63_P1_extra.bin' if problem=='extra' else 'hybrid_L34_P17_qkv.bin')).touch()
    with pytest.raises(ValueError):m.inventory(tmp_path)


def test_head_gate_does_not_hide_small_head_failure():
    m=load('check');ref=np.array([[1000.,1000.,1.,1.]])
    got=ref.copy();got[0,2]+=.02
    result=m.head_metrics(got,ref,2)
    assert result[0]['metrics']['passed'] and not result[1]['metrics']['passed']
    with pytest.raises(ValueError):m.head_metrics(got,ref[:,:2],2)


def test_failed_build_preserves_incomplete_evidence(tmp_path):
    m=load('build');out=tmp_path/'build'
    with pytest.raises(FileNotFoundError):m.build(SimpleNamespace(out=out,parent=tmp_path/'missing'))
    assert not json.loads((out/'build-evidence.json').read_text())['complete']
    with pytest.raises(FileExistsError):m.build(SimpleNamespace(out=out))


def test_failed_audit_never_claims_acceptance(tmp_path):
    m=load('check');out=tmp_path/'audit.json'
    with pytest.raises(FileNotFoundError):m.check(SimpleNamespace(run=tmp_path/'missing',out=out))
    r=json.loads(out.read_text());assert not r['capture_valid'] and not r['model_acceptance']


def test_runner_preflight_failure_keeps_false_report(tmp_path):
    out=tmp_path/'run'
    with pytest.raises(FileNotFoundError):
        load('run').run(SimpleNamespace(out=out,baseline=tmp_path/'missing',build=tmp_path/'build'))
    saved=json.loads((out/'report.json').read_text())
    assert not saved['complete'] and not saved['passed'] and not saved['model_acceptance']


def test_attention_oracle_accepts_spec_frequency_list_and_is_causal():
    m=load('check');values={k:np.ones((3,4)) for k in ('q','k','v','gate')}
    values['gate'][:]=0
    args=(np.ones(4),np.ones(4),[1.,.1],1,1,4,1e-6)
    og,kv=m.attention_reference(values,*args)
    assert og.shape==(3,4) and kv.shape==(3,16)
    np.testing.assert_allclose(og,.5)
    changed={k:v.copy() for k,v in values.items()};changed['v'][2]*=9
    later,_=m.attention_reference(changed,*args)
    np.testing.assert_array_equal(og[:2],later[:2])
    assert np.any(og[2]!=later[2])
