"""Isolated recurrence keeps causal state, original geometry and strict gates."""
import sys
from pathlib import Path
import numpy as np
import pytest
from recipes.load import load_spec

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'utilities/deltanet-replay'))
import recurrence_common as mod


def test_production_slices_and_padding_are_preserved():
    g=mod.geometry(load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json'))
    assert (g['rows'],g['slices'],g['state_bytes'],g['call_bytes'])==(10,13,66560,5120)
    assert g['heads']==48 and g['scratch_floats']==1280


def test_control_uses_production_header_and_entries():
    a=mod.sources('control');b=mod.sources('triple')
    assert a['dnx.h']==(ROOT/'open_kernels/designs/layer_x/dnx.h').read_text()
    assert a['dnx_row.cc']==b['dnx_row.cc']
    assert a['dnx.h']!=b['dnx.h']
    with pytest.raises(ValueError):mod.sources('unknown')


def test_each_head_starts_zero_but_tokens_carry_candidate_state(tmp_path):
    cfg=mod.replay_cfg(tmp_path,tmp_path,tmp_path,[0,2],3,2)
    assert cfg.count('load s ')==4
    assert cfg.count('run step ')==12
    assert cfg.count('copy s 0 so 0 66560')==12
    assert 'ref' not in cfg
    assert cfg.count('load so ')==12 and cfg.count('load o ')==12


def test_control_comparison_requires_signed_zero_bits():
    s=np.zeros((130,128),np.float32);o=np.zeros(128,np.float32)
    assert mod.score(s,o,s,o,s[:128],o,control=True)['passed']
    got=s.copy();got[0,0]=-0.
    r=mod.score(got,o,s,o,s[:128],o,control=True)
    assert not r['passed'] and not r['byte_exact']


def test_candidate_retains_head_gate_and_padding_gate():
    s=np.ones((130,128),np.float32);s[128:]=0;o=np.ones(128,np.float32)
    got=o.copy();got[0]+=1e-3
    assert not mod.score(s,got,s,o,s[:128],o,control=False)['passed']
    bad=s.copy();bad[-1,-1]=1e-30
    assert not mod.score(bad,o,s,o,s[:128],o,control=False)['passed']


@pytest.mark.parametrize('bad',['short','nan','extra'])
def test_incomplete_or_nonfinite_state_cannot_pass(bad):
    s=np.zeros((130,128),np.float32);o=np.zeros(128,np.float32);got=s.copy()
    if bad=='short':got=got[:-1]
    if bad=='extra':got=np.vstack([got,got[-1:]])
    if bad=='nan':got[0,0]=np.nan
    with pytest.raises(ValueError):mod.score(got,o,s,o,s[:128],o,control=False)


def input_fixture(path):
    import json
    names={f'L0_H{h}_T{t}.bin' for h in range(48) for t in range(2)}
    names|={'zero.bin','poison-state.bin','poison-output.bin'}
    for n in names:(path/n).write_bytes(b'fixture')
    meta=dict(layers=[0],heads=48,tokens=2,build_key='key',input_sha256={n:mod.digest(path/n) for n in names})
    (path/'fixture.json').write_text(json.dumps(meta));return meta


@pytest.mark.parametrize('damage',['missing_hash','wrong_hash','missing_file','single_token'])
def test_replay_input_identity_is_complete(tmp_path,damage):
    import json
    import recurrence_replay as replay
    m=input_fixture(tmp_path)
    if damage=='missing_hash':m['input_sha256'].pop('L0_H47_T1.bin')
    if damage=='wrong_hash':(tmp_path/'L0_H47_T1.bin').write_bytes(b'changed')
    if damage=='missing_file':(tmp_path/'L0_H47_T1.bin').unlink()
    if damage=='single_token':m['tokens']=1
    (tmp_path/'fixture.json').write_text(json.dumps(m))
    with pytest.raises((ValueError,OSError)):replay.verify_inputs(tmp_path)


@pytest.mark.parametrize('damage',['short','extra','nan'])
def test_raw_output_reader_refuses_stale_or_partial_dumps(tmp_path,damage):
    import recurrence_replay as replay
    p=tmp_path/'output.bin';a=np.zeros(128,'<f4')
    if damage=='short':a=a[:-1]
    if damage=='extra':a=np.zeros(129,'<f4')
    if damage=='nan':a[-1]=np.nan
    a.tofile(p)
    with pytest.raises(ValueError):replay.read_f32(p,(128,))


@pytest.mark.parametrize("mode",["control","triple","factored"])
def test_altered_binary_or_private_header_is_refused(tmp_path,mode):
    import json
    import recurrence_replay as replay
    (tmp_path/'source').mkdir()
    for n,s in mod.sources(mode).items():(tmp_path/'source'/n).write_text(s)
    for n in ('final.xclbin','insts.bin'):(tmp_path/n).write_bytes(b'binary')
    b=dict(mode=mode,production_build_key='key',source_sha256={n:mod.digest(tmp_path/'source'/n) for n in mod.sources(mode)},
           sha256={n:mod.digest(tmp_path/n) for n in ('final.xclbin','insts.bin')})
    (tmp_path/'build.json').write_text(json.dumps(b))
    assert replay.verify_build(tmp_path,dict(build_key='key'))==b
    (tmp_path/'source/dnx.h').write_text('stale')
    with pytest.raises(ValueError,match='source'):replay.verify_build(tmp_path,dict(build_key='key'))
    (tmp_path/'source/dnx.h').write_text(mod.sources(mode)['dnx.h'])
    (tmp_path/'insts.bin').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='binary'):replay.verify_build(tmp_path,dict(build_key='key'))


def test_remaining_error_is_separated_from_captured_state_readout():
    s=np.zeros((130,128),np.float32);s[0,0]=2
    q=np.zeros(128);q[0]=1
    ref=s[:128].astype(np.float64);ref[0,0]=1
    expected=np.zeros(128);expected[0]=1/np.sqrt(128)
    got=np.zeros(128,np.float32);got[0]=2/np.sqrt(128)
    r=mod.score(s,got,s,got,ref,expected,control=False,query=q)
    assert not r['passed'] and r['captured_state_readout']['passed']
    assert not r['state_effect']['passed']
    with pytest.raises(ValueError):mod.score(s,got,s,got,ref,expected,control=False,query=q[:-1])


def test_reconfigure_removes_stale_captures(tmp_path,monkeypatch):
    import recurrence_replay as replay
    inputs=tmp_path/'inputs';inputs.mkdir();(inputs/'fixture.json').write_text('metadata')
    build=tmp_path/'build';build.mkdir();(build/'build.json').write_text('build')
    out=tmp_path/'run';out.mkdir();old=out/'L0_H47_T1-state.bin';old.write_bytes(b'stale')
    monkeypatch.setattr(replay,'verify_inputs',lambda _:dict(layers=[0],tokens=2))
    monkeypatch.setattr(replay,'verify_build',lambda *a:dict(mode='control'))
    replay.configure(inputs,build,out)
    assert not old.exists() and (out/'run.json').exists()


def test_failed_configure_invalidates_previous_run(tmp_path,monkeypatch):
    import recurrence_replay as replay
    out=tmp_path/'run';out.mkdir();(out/'run.json').write_text('old')
    def fail(*args):raise ValueError('broken inputs')
    monkeypatch.setattr(replay,'verify_inputs',fail)
    with pytest.raises(ValueError):replay.configure(tmp_path,tmp_path,out)
    assert not (out/'run.json').exists()


def test_factored_build_pins_base_header_without_changing_control():
    candidate=mod.sources('factored')
    original=mod.sources('triple')
    assert candidate['precise.h']==original['dnx.h']
    assert candidate['dnx.h']==(mod.HERE/'factored.h').read_text()
    assert candidate['dnx_row.cc']==mod.sources('control')['dnx_row.cc']
    assert mod.sources('control')['dnx.h']==(mod.ROOT/'open_kernels/designs/layer_x/dnx.h').read_text()


def test_factored_reference_matches_independent_recurrence_with_causal_carry():
    from recurrence_common import factored_reference
    from deltanet_state import step_reference
    rng=np.random.default_rng(654)
    s=np.zeros((2,128,128))
    for t in range(3):
        record=np.zeros((2,512))
        record[:,:384]=rng.normal(size=(2,384))*.1
        record[:,384:386]=[.7,.4]
        expected,y=step_reference(s,record)
        state,output=factored_reference(s,record)
        np.testing.assert_allclose(state,expected,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(output,y,rtol=1e-12,atol=1e-14)
        s=state


def test_factored_readout_avoids_cancellation_in_rounded_updated_state():
    from recurrence_common import factored_reference
    from deltanet_state import step_reference
    record=np.zeros((1,512))
    record[0,:2]=[1.,1.+2**-12]
    record[0,128:130]=[1.+2**-12,-1.]
    record[0,256:384]=np.linspace(.1,1.,128)
    record[0,384:386]=[.9,.4]
    state,expected=step_reference(np.zeros((1,128,128)),record)
    rounded=np.einsum('hij,hi->hj',state.astype(np.float32).astype(np.float64),record[:,128:256])/np.sqrt(128)
    factored_state,output=factored_reference(np.zeros_like(state),record)
    assert np.max(np.abs(rounded))>1e-10
    np.testing.assert_allclose(output,expected,rtol=0,atol=1e-16)
    assert np.max(np.abs(output))==0
