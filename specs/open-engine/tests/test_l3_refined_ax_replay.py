"""Causal integrated replay must retain own KV and reject incomplete evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('refined_ax_replay',ROOT/'utilities/l3-refined-ax/replay.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_candidate_uses_own_kv_without_external_projection_inputs(tmp_path):
    m=load();sizes=dict(pool=8,consts=8,hidden=5120,act=274432,state=2048*8192,kv_row=8192,ptab=2048*2048,ptab_row=2048)
    for role in ('control','candidate'):
        s=m.program(tmp_path,tmp_path/'native',tmp_path/'build',sizes,role).splitlines()
        assert s.count('run ax pool xres consts state act ptab')==240
        assert len([line for line in s if line.startswith('load state ')])==1
        assert f'load state {tmp_path}/zero-state.bin' in s
        assert not any('forced' in line or 'candidate-og' in line for line in s)
        assert len([line for line in s if line.startswith('attnpos ax ')])==240
        assert f'load act {tmp_path}/zero-act.bin' in s


def test_expected_scratch_changes_only_projection_and_og_regions():
    m=load();off=SimpleNamespace(AA_BYTES=90000,AA_QG=16,AA_KVN=50000,AA_OG=60000)
    x=bytes([17])*off.AA_BYTES
    proj={k:np.full(n,i,dtype='<f4') for i,(k,n) in enumerate(zip(m.tail.refine.replay.ROLES,(6144,6144,1024,1024)))}
    og=np.zeros(6144,dtype='<f4');y=m.expected_act(x,proj,og,off)
    assert len(y)==len(x) and y[:16]==x[:16] and y[84576:]==x[84576:]
    assert y[16:16+6144*4]==proj['q'].tobytes()
    assert y[60000:84576]==og.tobytes()
    with pytest.raises(ValueError):m.expected_act(x[:-1],proj,og,off)
    proj['v']=proj['v'].astype(float)
    with pytest.raises(ValueError):m.expected_act(x,proj,og,off)


def test_full_capture_inventory_refuses_extra(tmp_path):
    m=load()
    for n in m.tail.names():(tmp_path/n).write_bytes(b'')
    m.tail.inventory(tmp_path)
    (tmp_path/'candidate-act_240.bin').write_bytes(b'')
    with pytest.raises(ValueError):m.tail.inventory(tmp_path)


def test_run_refuses_partial_existing_capture(tmp_path):
    m=load();(tmp_path/'control-state_0.bin').write_bytes(b'partial')
    with pytest.raises(ValueError,match='overwrite'):m.run(SimpleNamespace(out=tmp_path))
