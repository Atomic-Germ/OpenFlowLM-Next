"""Full-model scheduling and Q4NX preflight without allocating model weights."""
from dataclasses import replace
import json
from pathlib import Path
import struct
import hashlib
import importlib.util

import pytest
from recipes.spec import ModelSpec, LINEAR, FULL
from recipes import qwen35

FIXTURE = Path(__file__).parent/'fixtures/config_qwen38_27b.json'


@pytest.fixture
def geometry(monkeypatch):
    monkeypatch.setenv('OPEN_KERNELS_UNVALIDATED','1')
    s = ModelSpec.from_hf_config(json.loads(FIXTURE.read_text()))
    return s,qwen35.layout(s,max_ctx=257)


def test_full_schedule_keeps_all_64_states_and_streams_weights(geometry):
    from recipes.wide_slice import layer_types,layer_commands
    s,l = geometry
    kinds = layer_types(s,layers=64)
    assert (len(kinds),kinds.count(LINEAR),kinds.count(FULL)) == (64,48,16)
    count = 0
    for i,kind in enumerate(kinds):
        commands = layer_commands(s,l,i,256,257,layers=64)
        assert f'load pool layer{i}/pool.bin' in commands
        assert f'load const layer{i}/const.bin' in commands
        state = f'state{i}' if kind==LINEAR else f'cache{i}'
        assert any(state in c.split() for c in commands)
        assert not any(f'state{j}' in c.split() or f'cache{j}' in c.split()
                       for c in commands for j in range(64) if j!=i)
        count += sum(c.startswith('run ') for c in commands)
    assert count == 592
    assert layer_commands(s,l,7,0,257) == layer_commands(s,l,7,0,257,layers=64)


def test_full_schedule_rejects_truncated_or_wrong_geometry(geometry):
    from recipes.wide_slice import layer_types,layer_commands
    s,l = geometry
    for bad in (replace(s,num_layers=63),replace(s,layer_types=s.layer_types[:8]),
                replace(s,layer_types=(LINEAR,)*64)):
        with pytest.raises(ValueError): layer_types(bad,layers=64)
    for count in (0,7,32,65):
        with pytest.raises(ValueError): layer_types(s,layers=count)
    with pytest.raises(ValueError): layer_commands(s,l,64,0,257,layers=64)


def model_header(s):
    from recipes.wide_model import tensor_requirements
    header,offset = {},0
    for name,r in tensor_requirements(s).items():
        size = r['bytes']
        header[name] = dict(dtype=r['dtype'],shape=r['shape'],data_offsets=[offset,offset+size])
        offset += size
    return header,offset


def test_model_contract_includes_last_layer_and_both_q_gate_halves(geometry):
    from recipes.wide_model import tensor_requirements,validate_header
    s,_ = geometry
    header,size = model_header(s)
    r = tensor_requirements(s)
    assert r['model.layers.63.self_attn.q_proj.weight']['shape'] == [7680,5120]
    assert r['model.layers.62.linear_attn.ssm_alpha_proj.bf16.weight']['shape'] == [48,5120]
    assert r['model.embed_tokens.weight']['shape'] == [248320,5120]
    assert r['lm_head.weight']['shape'] == [155200,8704]
    assert validate_header(s,header,size)['ready_for_packing']
    del header['model.layers.63.self_attn.k_proj.weight']
    with pytest.raises(ValueError,match='63.*k_proj'): validate_header(s,header,size)


def test_model_preflight_rejects_bad_dtype_shape_truncation_and_overlap(geometry):
    from recipes.wide_model import validate_header
    s,_ = geometry
    for field,value in [('dtype','F16'),('shape',[248319,5120]),('data_offsets',[0,1])]:
        header,size = model_header(s)
        header['model.embed_tokens.weight'][field] = value
        with pytest.raises(ValueError): validate_header(s,header,size)
    header,size = model_header(s)
    with pytest.raises(ValueError): validate_header(s,header,size-1)
    names=list(header)
    header[names[1]]['data_offsets'] = header[names[0]]['data_offsets']
    with pytest.raises(ValueError): validate_header(s,header,size)


def test_header_reader_rejects_missing_or_malformed_container(tmp_path):
    from recipes.wide_model import read_header
    with pytest.raises(FileNotFoundError): read_header(tmp_path/'absent.q4nx')
    file=tmp_path/'model.q4nx'
    for raw in (b'',struct.pack('<Q',2**40),struct.pack('<Q',20)+b'{}'):
        file.write_bytes(raw)
        with pytest.raises(ValueError): read_header(file)
    raw=json.dumps({'__metadata__':{'format':'test'}}).encode()
    file.write_bytes(struct.pack('<Q',len(raw))+raw+b'1234')
    header,size=read_header(file)
    assert size==4 and header['__metadata__']['format']=='test'


def test_preflight_accepts_native_block_grid_and_rejects_transposed_grid(geometry):
    from recipes.wide_model import validate_header
    s,_=geometry
    header,size=model_header(s)
    header['model.layers.0.mlp.up_proj.weight']['shape']=[544,20,5120]
    header['lm_head.weight']['shape']=[7760,20,8704]
    assert validate_header(s,header,size)['ready_for_packing']
    header['model.layers.0.mlp.up_proj.weight']['shape']=[20,544,5120]
    with pytest.raises(ValueError): validate_header(s,header,size)


@pytest.fixture
def replay_fixture(tmp_path):
    path = Path(__file__).resolve().parents[3]/'utilities/replay-wide-model.py'
    spec = importlib.util.spec_from_file_location('replay_wide_model',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, kernel = tmp_path/'source', tmp_path/'kernel'
    source.mkdir(); kernel.mkdir()
    for name in ('final.xclbin','insts.bin'):
        (source/name).write_bytes(b'old')
        (kernel/name).write_bytes(b'new')
    (source/'decode.cfg').write_text(f'xclbin out {source}/final.xclbin\nkernelx out out {source}/insts.bin\nrun out w x y\n')
    (source/'weights.bin').write_bytes(b'weights')
    (source/'cold-0-logits.bin').write_bytes(b'old capture')
    meta = dict(source='real-q4nx', kernels={str(source/n):module.sha(source/n) for n in ('final.xclbin','insts.bin')},
                fixtures={n:module.sha(source/n) for n in ('weights.bin','decode.cfg')})
    (source/'slice-fixture.json').write_text(json.dumps(meta))
    (kernel/'projection-fixture.json').write_text(json.dumps(dict(k=6144,n=5120,sha256={n:module.sha(kernel/n) for n in ('final.xclbin','insts.bin')})))
    (kernel/'projection-results.json').write_text(json.dumps(dict(passed=True)))
    return module,source,kernel,tmp_path/'trial'


def test_model_replay_keeps_inputs_and_reference_separate_from_new_captures(replay_fixture):
    module,source,kernel,out = replay_fixture
    before = (source/'decode.cfg').read_bytes()
    module.replay(source,out,kernel)
    assert (source/'decode.cfg').read_bytes() == before
    assert (out/'weights.bin').read_bytes() == b'weights'
    assert not (out/'cold-0-logits.bin').exists()
    assert str(kernel/'final.xclbin') in (out/'decode.cfg').read_text()
    meta = json.loads((out/'slice-fixture.json').read_text())
    assert meta['fixtures']['weights.bin'] == hashlib.sha256(b'weights').hexdigest()
    assert meta['fixtures']['decode.cfg'] == module.sha(out/'decode.cfg')
    assert str(source/'final.xclbin') not in meta['kernels']
    assert meta['kernels'][str(kernel/'final.xclbin')] == module.sha(kernel/'final.xclbin')


def test_model_replay_rejects_stale_or_wrong_geometry_kernel(replay_fixture):
    module,source,kernel,out = replay_fixture
    fixture = json.loads((kernel/'projection-fixture.json').read_text())
    fixture['n'] = 1024
    (kernel/'projection-fixture.json').write_text(json.dumps(fixture))
    with pytest.raises(ValueError,match='geometry'): module.replay(source,out,kernel)
    fixture['n'] = 5120
    (kernel/'projection-fixture.json').write_text(json.dumps(fixture))
    (kernel/'insts.bin').write_bytes(b'stale')
    with pytest.raises(ValueError,match='changed'): module.replay(source,out,kernel)
    assert not out.exists()


def test_model_replay_can_replace_validated_ffn_without_replacing_reference(replay_fixture):
    module,source,kernel,out = replay_fixture
    ffn = kernel.parent/'ffn'; ffn.mkdir()
    for name in ('final.xclbin','insts.bin'):
        (ffn/name).write_bytes(b'ffn')
    (ffn/'segmented-fixture.json').write_text(json.dumps(dict(k=17408,n=5120,full=True,trace=False,
        sha256={n:module.sha(ffn/n) for n in ('final.xclbin','insts.bin')})))
    (ffn/'segmented-results.json').write_text(json.dumps(dict(passed=True)))
    cfg = source/'decode.cfg'
    cfg.write_text(cfg.read_text()+f'xclbin ffn {ffn}/final.xclbin\nkernelx ffn ffn {ffn}/insts.bin\n')
    meta = json.loads((source/'slice-fixture.json').read_text())
    meta['kernels'].update({str(ffn/n):module.sha(ffn/n) for n in ('final.xclbin','insts.bin')})
    meta['fixtures']['decode.cfg'] = module.sha(cfg)
    (source/'slice-fixture.json').write_text(json.dumps(meta))
    module.replay(source,out,kernel,ffn=ffn)
    copied = json.loads((out/'slice-fixture.json').read_text())
    assert copied['kernel_overrides']['ffn'] == str(ffn)
    assert copied['fixtures']['weights.bin'] == meta['fixtures']['weights.bin']


def test_model_replay_rejects_attention_with_different_static_dma_rows(replay_fixture):
    module,source,kernel,out = replay_fixture
    attn = kernel.parent/'attn'; attn.mkdir()
    (attn/'attention-fixture.json').write_text(json.dumps(dict(nh=24,kvh=4,hd=256,rot=64,rows=8)))
    with pytest.raises(ValueError,match='attention geometry'):
        module.replay(source,out,kernel,attention=attn)
    assert not out.exists()


def test_model_replay_rejects_ln_with_different_epsilon(replay_fixture):
    module,source,kernel,out = replay_fixture
    ln = kernel.parent/'ln'; ln.mkdir()
    (ln/'ln-fixture.json').write_text(json.dumps(dict(n=5120,eps=1e-5)))
    with pytest.raises(ValueError,match='LN geometry'):
        module.replay(source,out,kernel,ln=ln)
    assert not out.exists()


def test_model_replay_validates_and_replaces_attention_projection(replay_fixture):
    m,source,kernel,out = replay_fixture
    import json
    proj=kernel.parent/'qkvg';proj.mkdir()
    for n in ('final.xclbin','insts.bin'):(proj/n).write_bytes(b'qkvg')
    fixture=dict(k=5120,n=14336,sha256={n:m.sha(proj/n) for n in ('final.xclbin','insts.bin')})
    (proj/'projection-fixture.json').write_text(json.dumps(fixture))
    (proj/'projection-results.json').write_text(json.dumps(dict(passed=True)))
    old=source/'old-qkvg';old.write_bytes(b'old qkvg')
    inst=source/'old-qkvg-insts';inst.write_bytes(b'old instructions')
    cfg=source/'decode.cfg';cfg.write_text(cfg.read_text()+f'xclbin a_qkvg {old}\nkernelx a_qkvg a_qkvg {inst}\n')
    meta=json.loads((source/'slice-fixture.json').read_text());meta['fixtures']['decode.cfg']=m.sha(cfg)
    meta['kernels'].update({str(p):m.sha(p) for p in (old,inst)})
    (source/'slice-fixture.json').write_text(json.dumps(meta))
    m.replay(source,out,kernel,attention_projection=proj)
    assert f'xclbin a_qkvg {proj}/final.xclbin' in (out/'decode.cfg').read_text()
    fixture['n']=5120;(proj/'projection-fixture.json').write_text(json.dumps(fixture))
    with pytest.raises(ValueError,match='attention projection geometry'):
        m.replay(source,out.parent/'invalid',kernel,attention_projection=proj)
