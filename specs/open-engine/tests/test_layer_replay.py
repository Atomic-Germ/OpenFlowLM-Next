"""An isolated linear layer must preserve token inputs and recurrent state."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'utilities'))
from qwen35_layer_replay import replay_cfg


def fixture():
    m = dict(layers=['linear_attention'] * 3, layout=dict(hidden=2, pool_bytes=32),
             layer_types={'linear_attention': dict(buffers=dict(consts=8, act=16,
                 state=dict(kind='linear', bytes=12)),
                 program=[dict(op='run', kernel='lx', args=['pool','xres','consts','state','act'])])})
    return m


def test_one_pool_and_state_reused_across_tokens():
    cfg = replay_cfg(fixture(), 2, 3, Path('/fixture'), Path('/pools'), Path('/out'), Path('/kernel'))
    assert cfg.count('buf pool ') == 1 and cfg.count('buf state ') == 1
    assert cfg.count('run lx pool xres consts state act') == 3
    assert 'load state' not in cfg
    for suffix in ('', '_t1', '_t2'):
        assert f'load xres /fixture/y_res1{suffix}.bin' in cfg
        assert f'dump act /out/y_act2{suffix}.bin 16' in cfg


@pytest.mark.parametrize('layer,tokens', [(-1,3),(3,3),(2,0)])
def test_bad_coverage_refused(layer,tokens):
    with pytest.raises(ValueError):
        replay_cfg(fixture(),layer,tokens,Path('/f'),Path('/p'),Path('/o'),Path('/k'))


def test_attention_replay_patches_positions_and_carries_kv():
    m=fixture();m['layers'][2]='full_attention'
    m['max_ctx_default']=8;m['layout'].update(kv_row=16,ptab_row=4)
    m['layer_types']['full_attention']=dict(
        buffers=dict(consts=8,act=16,state=dict(kind='kv',row=16)),
        program=[dict(op='run',kernel='ax',args=['pool','xres','consts','state','act','ptab'])])
    cfg=replay_cfg(m,2,3,Path('/f'),Path('/p'),Path('/o'),Path('/k'))
    assert 'buf state 128' in cfg and 'buf ptab 32 /f/ptab.bin' in cfg
    assert cfg.count('buf state ') == 1
    for token in range(3):assert f'attnpos ax {token}' in cfg
    assert cfg.count('run ax pool xres consts state act ptab') == 3


def test_generated_dense_prep_preserves_fp32_input():
    import importlib.util
    from recipes.load import load_spec
    from recipes.qwen35 import recipe
    root=Path(__file__).resolve().parents[3]
    path=root/'open_kernels/designs/layer_x/gen_kernels.py'
    modspec=importlib.util.spec_from_file_location('dense_prep_generator',path)
    module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
    spec=load_spec(root/'open_kernels/recipes/specs/qwen35-27b.json')
    code=module.files(recipe(spec))['dense_prep_f32.cc']
    assert 'gemv_q4_prep_f32_direct_blocks(' in code
    assert 'gemv_q4_prep_f32_blocks(' not in code


def test_scoring_refuses_changed_replay_inputs(tmp_path):
    import hashlib, json
    from qwen35_layer_replay import verify_replay
    inp=tmp_path/'input';inp.write_bytes(b'original')
    (tmp_path/'run.cfg').write_text('run')
    (tmp_path/'replay.json').write_text(json.dumps({'sha256':{str(inp):hashlib.sha256(inp.read_bytes()).hexdigest()}}))
    verify_replay(tmp_path,'run')
    inp.write_bytes(b'modified')
    with pytest.raises(ValueError,match='hash'):
        verify_replay(tmp_path,'run')
    with pytest.raises(ValueError,match='config'):
        verify_replay(tmp_path,'different')


def test_unsplit_dense_prep_keeps_legacy_route():
    import importlib.util
    from recipes.load import load_spec
    from recipes.qwen35 import recipe
    root=Path(__file__).resolve().parents[3]
    path=root/'open_kernels/designs/layer_x/gen_kernels.py'
    modspec=importlib.util.spec_from_file_location('unsplit_prep_generator',path)
    module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
    spec=load_spec(root/'open_kernels/recipes/specs/qwen35-9b.json')
    code=module.files(recipe(spec))['dense_prep_f32.cc']
    assert 'gemv_q4_prep_f32_blocks(' in code
    assert 'gemv_q4_prep_f32_direct_blocks(' not in code


@pytest.mark.parametrize('previous', [False, True])
def test_empty_sweep_cannot_publish_success(tmp_path, previous):
    import json, subprocess
    root=Path(__file__).resolve().parents[3]
    (tmp_path/'manifest.json').write_text(json.dumps({'layers':[]}))
    (tmp_path/'decode_reference.json').write_text(json.dumps({'layers':0,'tokens':3}))
    out=tmp_path/'result'
    if previous:
        out.mkdir();(out/'results.json').write_text('{"passed":true}')
    command=[sys.executable,str(root/'utilities/gemv-fp32-prep/sweep.py')]
    for name in ('model-dir','kernel-dir','fixture','pools','candidate-root'):
        command += ['--'+name,str(tmp_path)]
    result=subprocess.run(command+['--out',str(out)],capture_output=True,text=True)
    assert result.returncode != 0
    assert not (out/'results.json').exists()
