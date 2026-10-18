"""All-layer separate projections need production identity and independent gates."""
import importlib.util
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
import pytest
from recipes.load import load_spec
from recipes.manifest import manifest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'utilities/ffn-front-replay'))
import front_build as build
loader=importlib.util.spec_from_file_location('front_sweep',ROOT/'utilities/ffn-front-replay/sweep.py')
sweep=importlib.util.module_from_spec(loader);loader.loader.exec_module(sweep)


def spec():return load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')


def test_production_mode_selects_consistent_direct_prep():
    assert build.prep_flags(2)==['-DFRONT_FP32=1','-DGEMV_Q4_CONSISTENT_SUM=1']
    assert build.prep_flags(0)==['-DFRONT_FP32=0']
    assert build.prep_flags(1)==['-DFRONT_FP32=1']
    build.validate_spec(spec(),2)
    with pytest.raises(ValueError):build.prep_flags(3)
    with pytest.raises(ValueError):build.validate_spec(replace(spec(),quant='q8'),2)
    with pytest.raises(ValueError):build.validate_spec(replace(spec(),hidden=4096),2)


def test_compiled_prep_and_dependency_drift_are_refused(tmp_path):
    prj=tmp_path/'final.prj';prj.mkdir()
    for n in ('front_prep','front_gms','front_emit'):
        (prj/(n+'.cc')).write_bytes((build.HERE/(n+'.cc')).read_bytes())
        (prj/(n+'.o.d')).write_text('obj: '+str(build.GEMV/'gemv_q4.h')+' '+str(build.GEMV/'gemv_tab.h'))
    (prj/'ln_nr32.cc').write_bytes((build.LN/'ln_nr32.cc').read_bytes())
    (prj/'ln_nr32.o.d').write_text('obj: '+str(build.LN/'ln.h'))
    assert build.audit_compilation(tmp_path)
    (prj/'front_prep.cc').write_text('changed')
    with pytest.raises(ValueError,match='compiled source'):build.audit_compilation(tmp_path)
    (prj/'front_prep.cc').write_bytes((build.HERE/'front_prep.cc').read_bytes())
    (prj/'front_gms.o.d').write_text('obj: /private/gemv_q4.h /private/gemv_tab.h')
    with pytest.raises(ValueError,match='header'):build.audit_compilation(tmp_path)


@pytest.mark.parametrize('field,value',[('layers',8),('tokens',1),('build_key','wrong'),('spec_hash','wrong')])
def test_partial_or_unrelated_fixture_is_refused(field,value):
    b=manifest(spec());m=dict(layers=64,tokens=3,build_key=b['build_key'],spec_hash=b['spec_hash'])
    sweep.validate_fixture(spec(),m,b)
    with pytest.raises(ValueError):sweep.validate_fixture(spec(),dict(m,**{field:value}),b)


def case():
    res=np.ones(2);norm=np.ones(2);x=res/np.sqrt(1+1e-6)
    w=np.ones((1,2));u=w@x;h=u*u/(1+np.exp(-u))
    return res,norm,w,w,x,(u,u,h),1e-6,x.copy(),h.copy()


def test_projection_failure_cannot_be_hidden_by_matching_activation():
    args=list(case());u,g,h=args[5];args[5]=(2*u,g,h)
    r=sweep.score_case(*args)
    assert r['production_identical'] and not r['passed'] and not r['up']['passed']


def test_numerical_pass_cannot_replace_production_identity():
    args=list(case());args[-1]=np.nextafter(args[-1],np.inf)
    r=sweep.score_case(*args)
    assert r['front']['passed'] and r['up']['passed'] and r['gate']['passed']
    assert not r['production_identical'] and not r['passed']
    assert sweep.score_case(*case())['passed']


def test_pool_streaming_covers_each_layer_and_position(tmp_path):
    from recipes.qwen35 import layout
    s=spec();text=sweep.config(s,layout(s,2048),range(64),3,tmp_path/'pools',tmp_path/'out',tmp_path/'build')
    assert text.count('buf pool ')==1 and text.count('load pool ')==64
    assert text.count('run front pool rn norm out')==192
    assert text.count('dump norm ')==192 and text.count('dump out ')==192
    assert 'front_L63_t2.bin' in text


def test_original_provenance_is_required(tmp_path):
    p=tmp_path/'pool';current={str(p):'abc'}
    sweep.verify_original(current,current)
    for bad in ({},{str(p):'wrong'}):
        with pytest.raises(ValueError,match='provenance'):sweep.verify_original(current,bad)


def test_source_outputs_cannot_overlap(tmp_path):
    source=tmp_path/'source'
    for out in (source,source/'child',tmp_path):
        with pytest.raises(ValueError,match='separate'):sweep.separate_output(out,[source])
    sweep.separate_output(tmp_path/'out',[source])


def test_failed_score_removes_stale_success(tmp_path,monkeypatch):
    (tmp_path/'score.json').write_text('{"passed":true}')
    def fail(_):raise ValueError('stale')
    monkeypatch.setattr(sweep,'verified',fail)
    with pytest.raises(ValueError):sweep.score(tmp_path)
    assert not (tmp_path/'score.json').exists()


def test_fp64_reference_input_is_serialized_as_fp32_then_bf16():
    from q4nx import f32_to_bf16
    res=np.array([1.25,-0.0,3.5,-9.25],np.float64);nw=np.array([1,2,3,4],np.float64)
    packed=sweep.pack_input(res,nw)
    assert len(packed)==6*len(res)
    assert packed[:4*len(res)]==res.astype('<f4').tobytes()
    assert packed[4*len(res):]==f32_to_bf16(nw.astype(np.float32)).tobytes()
    with pytest.raises(ValueError):sweep.pack_input(res,nw[:2])
    with pytest.raises(ValueError):sweep.pack_input(np.full_like(res,np.nan),nw)
