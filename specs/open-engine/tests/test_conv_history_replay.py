"""FP32 history must preserve low bits and causal shifts without wider FIFOs."""
from pathlib import Path
import sys
import numpy as np
import pytest
import deltanet_state as state
import deltanet_inputs as inputs
from recipes.load import load_spec
from test_linear_entry_replay import retained_linear_recipe,retained_linear_source

ROOT=Path(__file__).resolve().parents[3]


def fp32_sequence():
    from test_deltanet_state import sequence
    caps=sequence();history=np.zeros(caps[0]['conv'].shape,np.float32)
    for c in caps:
        history=np.vstack([history[1:],c['qkv'].astype(np.float32)])
        c['conv']=history.copy()
    return caps


def test_fp32_history_checks_exact_causal_shift():
    caps=fp32_sequence()
    assert all(r['passed'] for r in state.compare_sequence(caps,conv_dtype='fp32'))
    caps[1]['conv'].view(np.uint32)[-1,-1]^=1
    rows=state.compare_sequence(caps,conv_dtype='fp32')
    assert not rows[1]['conv_bit_exact'] and rows[2]['conv_bit_exact']


def test_fp32_history_preserves_signed_zero():
    caps=fp32_sequence();caps[0]['conv'][0,0]=-0.
    assert not state.compare_sequence(caps,conv_dtype='fp32')[0]['conv_bit_exact']


@pytest.mark.parametrize('dtype',['bf16','fp16','unknown'])
def test_history_storage_cannot_be_inferred_or_misdeclared(dtype):
    with pytest.raises(ValueError):state.compare_sequence(fp32_sequence(),conv_dtype=dtype)


def test_fp32_history_shared_preparation_oracle():
    from test_deltanet_inputs import spec,weights,captures
    s=spec();w=weights(s);x=[np.array([1.,-.5]),np.array([.25,2.])]
    c=captures(s,w,x);history=np.zeros(c[0]['conv'].shape,np.float32)
    for row in c:
        history=np.vstack([history[1:],row['qkv'].astype(np.float32)])
        row['conv']=history.copy()
    rows=inputs.score_sequence(s,w,x,c,conv_dtype='fp32')
    assert all(r['passed'] for r in rows)
    assert all(r['composed_state']['maxrel']==0 for r in rows)


def test_private_history_geometry_and_design(monkeypatch,retained_linear_recipe,retained_linear_source):
    monkeypatch.syspath_prepend(str(ROOT/'utilities/conv-history-replay'))
    from history_common import geometry,render_design
    s=load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')
    a=geometry(s,'fp32');b=geometry(s,'bf16')
    assert a['act_bytes']==b['act_bytes']==405504
    assert a['state_bytes']==b['state_bytes']+61440==3317760
    assert a['state_offset']==122880 and a['conv_dtype']=='fp32'
    code=render_design(s,source=retained_linear_source)
    assert 'install_history(XL)' in code and 'history_sequence(' in code
    assert 'STATE_S_OFF *= 2' in code
    with pytest.raises(ValueError):render_design(s,source='changed upstream')


def test_fp32_binary_reader_covers_expanded_state_and_finite_history():
    from types import SimpleNamespace
    spec=SimpleNamespace(lin_key_dim=128,lin_value_dim=128,lin_key_heads=1,lin_value_heads=2,
                         lin_qkv_dim=512,conv_kernel=4)
    off=SimpleNamespace(A_QKV=0,A_VEC=2048,A_O=6144,A_BYTES=7168,S_ROWS=130,
                        S_HEAD_BYTES=130*128*4,STATE_S_OFF=6144,STATE_BYTES=6144+2*130*128*4)
    buf=bytearray(off.STATE_BYTES);buf[:4]=np.float32(1.001).tobytes()
    c=state.read_boundaries(bytes(off.A_BYTES),buf,spec,off,conv_dtype='fp32')
    assert c['conv'].dtype==np.float32 and c['conv'][0,0]==np.float32(1.001)
    with pytest.raises(ValueError):state.read_boundaries(bytes(off.A_BYTES),buf,spec,off)
    buf[:4]=np.float32(np.nan).tobytes()
    with pytest.raises(ValueError):state.read_boundaries(bytes(off.A_BYTES),buf,spec,off,conv_dtype='fp32')


def test_half_tile_schedule_retains_fifo_bounds_and_all_heads(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(ROOT/'utilities/conv-history-replay'))
    from history_worker import conv_sequence
    class Stream:
        def __init__(self,limit):self.limit=limit;self.held=0;self.total=0
        def acquire(self,n):
            assert self.held==0 and n<=self.limit
            self.held=n;self.total+=n;return list(range(n)) if n>1 else 0
        def release(self,n):assert self.held==n;self.held=0
    side,act,out=Stream(2),Stream(5),Stream(3);calls=[];heads=[]
    xl=SimpleNamespace(KEY_TILES=4,VALUE_TILES=6,CONVW_ELEMS=2,range_=range,
                       X=SimpleNamespace(band_range=range),D=SimpleNamespace(HEADS_PER_TILE=8))
    conv_sequence(xl,side,act,out,None,None,None,None,
                  lambda *args:calls.append(args[-3:]),lambda *args:heads.append(args[-2:]))
    assert len(calls)==20 and len(heads)==48
    assert [(tile+base,half) for tile,base,half in calls]==[(t,h) for t in range(10) for h in range(2)]
    assert (side.total,act.total,out.total)==(20,80,108)


def test_host_rows_cover_exactly_all_fp32_history_bytes(monkeypatch):
    pytest.importorskip('aie')
    monkeypatch.syspath_prepend(str(ROOT/'utilities/conv-history-replay'))
    from history_worker import history_rows
    seen=[]
    for t in range(10):
        for h in range(2):
            tap=history_rows(10240,t,h,3317760)
            # Each half: three contiguous 2048-byte slices with full-row stride.
            assert list(tap.sizes)==[1,1,3,2048]
            assert list(tap.strides)==[0,0,40960,1]
            for row in range(3):seen.extend(range(tap.offset+row*40960,tap.offset+row*40960+2048))
    assert sorted(seen)==list(range(122880))


def test_replay_cfg_allocates_expanded_state_and_keeps_carry(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'utilities/conv-history-replay'))
    from history_check import replay_cfg
    m=dict(layout=dict(pool_bytes=128,hidden=2),layer_types={'linear_attention':{'buffers':{'consts':32,'state':{'bytes':64}}}})
    cfg=replay_cfg(m,dict(act_bytes=96,state_bytes=128),[0,2],3,tmp_path,tmp_path,tmp_path,tmp_path)
    assert 'buf state 128' in cfg and cfg.count('load state ')==2
    assert cfg.count('dump state ')==6 and cfg.count('run lx ')==6


@pytest.mark.parametrize('damage',['short','extra','nan'])
def test_fp32_history_capture_rejects_invalid_values(damage):
    caps=fp32_sequence()
    if damage=='short':caps[1]['conv']=caps[1]['conv'][:2]
    if damage=='extra':caps[1]['conv']=np.vstack([caps[1]['conv'],caps[1]['conv'][-1:]])
    if damage=='nan':caps[1]['conv'][0,0]=np.nan
    with pytest.raises(ValueError):state.compare_sequence(caps,conv_dtype='fp32')
