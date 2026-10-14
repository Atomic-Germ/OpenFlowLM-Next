"""Private linear entry replay keeps production state/weights and causal history."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'utilities/linear-entry-replay'))
from linear_common import geometry,render_design,source_hash,read_norm
from linear_check import replay_cfg
from recipes.load import load_spec
sys.path.remove(str(ROOT/'utilities/linear-entry-replay'))


@pytest.fixture
def retained_linear_recipe(monkeypatch):
    from dataclasses import replace
    import linear_common as mod
    original=mod.recipe
    def retained(s):
        r=original(s)
        if r.linear.XN_F32:
            r=replace(r,linear=replace(r.linear,XN_F32=False,CONV_F32=False,XN_SIDE_ELEMS=3),
                      layout=replace(r.layout,A_BYTES=385024,STATE_S_OFF=61440,STATE_BYTES=3256320))
        return r
    monkeypatch.setattr(mod,'recipe',retained)


@pytest.fixture
def retained_linear_source():
    # Only historical anchors are needed; never reinterpret integrated production.
    return '\n'.join(['HERE = Path(__file__).parent', 'OG_ELEMS = D.OG_ELEMS',
        'XL.ln_worker(of_lni, of_lno, tl, L)',
        'X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, QKV_PC + Z_PC, "linear")',
        'lno.drain(a_act, tap=bt(A_BYTES, A_XN, ELN), wait=True, group=tg_ln)',
        'K = X.kernels(inc, t)',
        '        pipe = Pipeline(3)', '        for tt in range(NT):', '            pass',
        '        pipe.finish()                                    # the records are in DDR'])


def test_integrated_recipe_is_refused_by_historical_replay():
    with pytest.raises(ValueError):geometry(spec(),'fp32')


def spec():return load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')


def test_fp32_entry_appends_scratch_and_streams_five_ab_blocks(retained_linear_recipe):
    g=geometry(spec(),'fp32');b=geometry(spec(),'bf16')
    assert g['act_bytes']==b['act_bytes']+20480
    assert g['norm_offset']==b['act_bytes'] and g['norm_bytes']==20480
    assert g['ab_tiles']==[32]*5 and sum(g['ab_tiles'])==160
    assert b['ab_tiles']==[64,64,32]
    assert g['state_bytes']==b['state_bytes']


def test_render_switches_all_entry_consumers_without_touching_production(retained_linear_recipe,retained_linear_source):
    p=ROOT/'open_kernels/designs/layer_x/lx.py';before=p.read_bytes()
    s=render_design(spec(),source=retained_linear_source)
    assert 'entry_fp32=True' in s and '"linear", fp32=True)' in s
    assert 'A_XN = A_BYTES' in s and 'A_XN, HID * 4)' in s
    assert 'install_glue(XL)' in s and 'AB_TILES = XL.AB_TILES' in s
    assert p.read_bytes()==before
    assert source_hash(spec(),'fp32')!=source_hash(spec(),'bf16')


def test_changed_anchor_is_refused(retained_linear_recipe):
    with pytest.raises(ValueError,match='anchor'):render_design(spec(),source='new upstream')


def test_private_translation_units_follow_recipe_and_reject_stale_files(tmp_path):
    from linear_common import generated_sources,verify_generated
    sources=generated_sources(spec())
    assert '#define GEMV_PER_CALL 1' in sources['gemv_q4_gy.cc']
    assert 'gemv_q4_prep_f32_direct_blocks(' in sources['dense_prep_f32.cc']
    for name,source in sources.items():(tmp_path/name).write_text(source)
    hashes=verify_generated(tmp_path,spec())
    assert set(hashes)==set(sources)
    (tmp_path/'gemv_q4_gy.cc').write_text('#define GEMV_PER_CALL 2')
    with pytest.raises(ValueError,match='generated'):verify_generated(tmp_path,spec())


def test_design_uses_private_kernel_entries(retained_linear_recipe,retained_linear_source):
    s=render_design(spec(),source=retained_linear_source)
    assert 'private_kernels(X, inc, t, Path(__file__).parent / "generated")' in s


@pytest.mark.parametrize('mode',['fp16','unknown'])
def test_unknown_mode(mode):
    with pytest.raises(ValueError):geometry(spec(),mode)


def test_other_geometry_is_refused():
    with pytest.raises(ValueError):geometry(load_spec(ROOT/'open_kernels/recipes/specs/qwen35-9b.json'),'fp32')


def test_norm_read_keeps_low_bits_and_checks_bytes(tmp_path):
    g=dict(mode='fp32',act_bytes=24,norm_offset=8,norm_bytes=16,hidden=4)
    x=np.array([1.001,-2.001,3.003,4.004],np.float32);p=tmp_path/'act'
    p.write_bytes(bytes(8)+x.tobytes());np.testing.assert_array_equal(read_norm(p,g),x.astype(np.float64))
    p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError):read_norm(p,g)
    x[-1]=np.nan;p.write_bytes(bytes(8)+x.tobytes())
    with pytest.raises(ValueError):read_norm(p,g)


def test_cfg_keeps_state_between_tokens_and_resets_between_layers(tmp_path):
    m=dict(layout=dict(pool_bytes=128,hidden=2),layer_types={'linear_attention':{'buffers':{'consts':32,'state':{'bytes':64}}}})
    cfg=replay_cfg(m,dict(act_bytes=96),[0,2],3,tmp_path/'fixture',tmp_path/'pools',tmp_path/'out',tmp_path/'build')
    assert cfg.count('run lx pool xres consts state act')==6
    assert cfg.count('load state ')==2 and cfg.count('load act ')==2
    assert cfg.count('dump state ')==6
    assert 'xres0.bin' in cfg and 'y_res1_t2.bin' in cfg
    assert 'buf act 96' in cfg


def test_rolled_ab_schedule_preserves_all_rows_and_first_block_reset(monkeypatch):
    pytest.importorskip('aie')
    monkeypatch.syspath_prepend(str(ROOT/'utilities/linear-entry-replay'))
    from linear_worker import ab_sequence
    class Stream:
        def __init__(self):self.n=0;self.pending=0
        def acquire(self,n):
            assert n==1 and self.pending==0
            self.pending=1;self.n+=1;return self.n
        def release(self,n):assert self.pending==n;self.pending=0
    stream=Stream();calls=[];copies=[]
    ab_sequence(stream,None,'a','b',lambda w,x,acc,tile,block:calls.append((acc,block*32+tile,block)),
                lambda e,x,i:copies.append(e),range)
    assert len(copies)==5 and stream.n==325 and stream.pending==0
    for acc in ('a','b'):
        assert [row for a,row,b in calls if a==acc]==list(range(160))
        assert sum(row==0 and block==0 for a,row,block in calls if a==acc)==1


def test_private_kernels_keep_shared_headers_and_restore_source_root(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'utilities/linear-entry-replay'))
    from linear_worker import private_kernels
    from types import SimpleNamespace
    original=ROOT/'open_kernels/designs/layer_x'
    x=SimpleNamespace(HERE=original)
    x.kernels=lambda inc,t: (x.HERE,inc,t)
    includes=['other'];types={}
    actual=private_kernels(x,includes,types,tmp_path)
    assert actual==(tmp_path,['other',str(original)],types)
    assert x.HERE==original and includes==['other']
    def fail(inc,t):raise RuntimeError('compile setup')
    x.kernels=fail
    with pytest.raises(RuntimeError):private_kernels(x,includes,types,tmp_path)
    assert x.HERE==original
