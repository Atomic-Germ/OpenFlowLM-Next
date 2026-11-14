"""Current full-ax replay must preserve native KV and exact capture provenance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('native_l3',ROOT/'utilities/prefill-sequential-drift/native_l3.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def sizes():
    return dict(pool=128,consts=32,act=64,state=240*16,hidden=4,kv_row=16,ptab_row=8,ptab=240*8)


def test_program_keeps_kv_and_patches_every_position(tmp_path):
    m=load();lines=m.program(tmp_path,tmp_path/'kernels',sizes(),240).splitlines()
    assert 'attngeom 16 8' in lines
    assert sum(s.startswith('buf state ') for s in lines)==1
    assert not any(s.startswith(('load state ','load act ')) for s in lines)
    assert lines.count('run ax pool xres consts state act ptab')==240
    for p in range(240):
        i=lines.index(f'load xres {tmp_path}/input_{p}.bin')
        assert lines[i+1:i+6]==[f'attnpos ax {p}','run ax pool xres consts state act ptab',
            f'dump xres {tmp_path}/xres_{p}.bin 16',f'dump act {tmp_path}/act_{p}.bin 64',
            f'dump state {tmp_path}/state_{p}.bin {(p+1)*16}']


@pytest.mark.parametrize('fault',['count','bool','kv_capacity','ptab_capacity','geometry'])
def test_program_refuses_invalid_extent_or_geometry(tmp_path,fault):
    m=load();s=sizes();count=240
    if fault=='count':count=241
    elif fault=='bool':count=True
    elif fault=='kv_capacity':s['state']=16
    elif fault=='ptab_capacity':s['ptab']=8
    else:s['kv_row']=0
    with pytest.raises(ValueError):m.program(tmp_path,tmp_path,s,count)


def test_reader_requires_complete_active_kv_prefix(tmp_path):
    m=load();s=sizes()
    for p in range(2):
        np.ones(4,dtype='<f4').tofile(tmp_path/f'xres_{p}.bin')
        (tmp_path/f'act_{p}.bin').write_bytes(bytes(64))
        (tmp_path/f'state_{p}.bin').write_bytes(bytes((p+1)*16))
    assert m.read_outputs(tmp_path,s,2)[0].shape==(2,4)
    (tmp_path/'state_1.bin').write_bytes(bytes(16))
    with pytest.raises(ValueError,match='capture'):m.read_outputs(tmp_path,s,2)


def test_score_uses_current_sequential_layer_three(tmp_path):
    m=load();d=tmp_path/'257/trace';d.mkdir(parents=True)
    expected=np.arange(258*4,dtype='<f4').reshape(258,4);expected.tofile(d/'sequential_res3.bin')
    out=tmp_path/'capture';out.mkdir();s=sizes()
    for p in range(m.COUNT):
        expected[p].tofile(out/f'xres_{p}.bin')
        (out/f'act_{p}.bin').write_bytes(bytes(s['act']))
        (out/f'state_{p}.bin').write_bytes(bytes((p+1)*s['kv_row']))
    proof,hashes=m.score(out,dict(sizes=s,baseline=str(tmp_path)))
    assert proof['passed'] and proof['matching_positions']==240 and len(hashes)==720
    changed=expected[125].copy();changed[0]=np.nextafter(changed[0],np.float32(np.inf))
    changed.tofile(out/'xres_125.bin')
    assert m.score(out,dict(sizes=s,baseline=str(tmp_path)))[0]['first_mismatch']==125


def test_audit_preserves_failed_native_control(tmp_path,monkeypatch):
    m=load();control=dict(passed=False,positions=240,matching_positions=239,first_mismatch=125)
    record=dict(complete=True,model_acceptance=False,preparation_sha256='hash',harness='runner',
                harness_sha256='hash',harness_evidence_sha256='hash',log_sha256='hash',identity=control,
                artifacts={'xres_0.bin':'capture'},control_valid=False,boundary_interpretation_allowed=False)
    (tmp_path/'report.json').write_text(json.dumps(record))
    monkeypatch.setattr(m,'verified',lambda out:{})
    monkeypatch.setattr(m,'digest',lambda path:'hash')
    monkeypatch.setattr(m,'verify_harness',lambda path:'hash')
    monkeypatch.setattr(m,'score',lambda *a:(control,record['artifacts']))
    assert m.audit(SimpleNamespace(out=tmp_path))==1
    assert not json.loads((tmp_path/'audit.json').read_text())['control_valid']
    record['boundary_interpretation_allowed']=True
    (tmp_path/'report.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):m.audit(SimpleNamespace(out=tmp_path))


def test_run_refuses_existing_capture(tmp_path):
    m=load();(tmp_path/'state_125.bin').write_bytes(b'partial')
    with pytest.raises(ValueError,match='overwrite'):m.run(SimpleNamespace(out=tmp_path,harness=tmp_path/'runner'))


def test_output_inventory_rejects_extra_position(tmp_path):
    m=load();s=sizes()
    np.ones(4,dtype='<f4').tofile(tmp_path/'xres_0.bin')
    (tmp_path/'act_0.bin').write_bytes(bytes(64));(tmp_path/'state_0.bin').write_bytes(bytes(16))
    (tmp_path/'state_1.bin').write_bytes(bytes(32))
    with pytest.raises(ValueError,match='inventory'):m.read_outputs(tmp_path,s,1)


@pytest.mark.parametrize('fault',['none','kv_dtype','entry_dtype','kind','row','hash','layer'])
def test_current_bundle_abi_is_required_without_relabeling_reference(tmp_path,monkeypatch,fault):
    m=load();spec=SimpleNamespace(hidden=5120,num_layers=64,layer_types=['linear_attention']*64)
    spec.layer_types[3]='full_attention'
    bundle=dict(spec_hash='spec',build_key='current',layout=dict(pool_bytes=128,kv_row=16,ptab_row=8,
        attn_kv_dtype='bfloat16_pair',attn_output_dtype='float32',attn_entry_dtype='float32',ffn_norm_dtype='float32'),
        layer_types=dict(full_attention=dict(buffers=dict(consts=32,act=64,state=dict(kind='kv',row=16)))))
    meta=dict(model=str(tmp_path),kernels=str(tmp_path),spec_hash='spec',build_key='historical',max_ctx=256)
    if fault=='kv_dtype':bundle['layout']['attn_kv_dtype']='bfloat16'
    if fault=='entry_dtype':bundle['layout']['attn_entry_dtype']='bfloat16'
    if fault=='kind':bundle['layer_types']['full_attention']['buffers']['state']['kind']='linear'
    if fault=='row':bundle['layer_types']['full_attention']['buffers']['state']['row']=8
    if fault=='hash':meta['spec_hash']='changed'
    if fault=='layer':spec.layer_types[3]='linear_attention'
    monkeypatch.setattr(m.base,'spec_from_model_dir',lambda *a:spec)
    monkeypatch.setattr(m.base,'manifest',lambda *a,**kw:{})
    monkeypatch.setattr(m.base,'load_bundle',lambda *a:bundle)
    if fault=='none':
        _,actual,s=m.context(meta)
        assert actual['build_key']=='current' and meta['build_key']=='historical'
        assert s['state']==256*16 and s['ptab']==256*8
    else:
        with pytest.raises(ValueError):m.context(meta)


def test_harness_build_evidence_is_required_and_rehashed(tmp_path):
    m=load();h=tmp_path/'run_kernel';h.write_bytes(b'runner')
    paths=[h,ROOT/'open_kernels/harness/run_kernel.cpp',ROOT/'open_kernels/harness/stream_patch.hpp']
    evidence=tmp_path/'build-evidence.json'
    evidence.write_text(json.dumps(dict(sha256={str(p):m.digest(p) for p in paths})))
    assert m.verify_harness(h)==m.digest(evidence)
    h.write_bytes(b'changed')
    with pytest.raises(ValueError):m.verify_harness(h)
    evidence.write_text(json.dumps(dict(sha256={})))
    with pytest.raises(ValueError,match='unbound'):m.verify_harness(h)
