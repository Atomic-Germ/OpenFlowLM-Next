"""Wide head order and bounded-memory container output."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_tiled_48_heads_restore_grouped_hf_order_on_rows_and_columns():
    from q4nx.models.qwen35_wide import grouped_heads
    # Independently construct ggml's [slot, key-head, element] raster.
    hf=torch.arange(48*128).reshape(48,128)
    tiled=torch.stack([hf[g*3+r] for r in range(3) for g in range(16)]).reshape(-1)
    assert torch.equal(grouped_heads(tiled,0,128),hf.reshape(-1))
    columns=tiled.repeat(2,1)
    assert torch.equal(grouped_heads(columns,1,128),hf.reshape(-1).repeat(2,1))
    with pytest.raises(ValueError): grouped_heads(torch.zeros(32),0,1)


def test_qkv_only_reorders_value_rows_and_q_gate_is_split():
    from q4nx.models.qwen35_wide import restore_order
    v=torch.arange(6144).reshape(48,128)
    tiled=torch.stack([v[g*3+r] for r in range(3) for g in range(16)]).reshape(-1,1)
    qk=torch.arange(4096).reshape(-1,1)+100000
    x=torch.cat([qk,tiled])
    assert torch.equal(restore_order('model.layers.0.linear_attn.qkv_proj.weight',x),torch.cat([qk,v.reshape(-1,1)]))
    conv=x.repeat(1,4)
    assert torch.equal(restore_order('model.layers.0.linear_attn.ssm_conv1d.weight',conv),torch.cat([qk,v.reshape(-1,1)]).repeat(1,4).T)
    q=torch.arange(24*2*256).reshape(24,2,256)
    expected=torch.cat([q[:,0].reshape(-1),q[:,1].reshape(-1)])
    assert torch.equal(restore_order('model.layers.3.self_attn.q_proj.weight',q.reshape(-1,1)).ravel(),expected)


@pytest.mark.parametrize('suffix,shape,axis,unit,block', [
    ('self_attn.gate_proj.weight', (6144, 3), 0, 128, 1),
    ('linear_attn.ssm_alpha_proj.weight', (48, 3), 0, 1, 1),
    ('linear_attn.ssm_beta_proj.weight', (48, 3), 0, 1, 1),
    ('linear_attn.ssm_a', (48,), 0, 1, 1),
    ('linear_attn.ssm_dt.bias', (48,), 0, 1, 1),
    ('linear_attn.ssm_out_proj.weight', (3, 6144), 1, 128, 1),
    ('linear_attn.ssm_out_proj.weight', (3, 192), 1, 4, 32),
])
def test_wide_order_agrees_with_upstream_general_head_fix(suffix, shape, axis, unit, block):
    from q4nx.models.qwen35 import v_untile
    from q4nx.models.qwen35_wide import restore_order
    x = torch.arange(int(np.prod(shape))).reshape(shape)
    assert torch.equal(restore_order('model.layers.0.'+suffix, x, block),
                       v_untile(x, 3, unit, axis))


def test_wide_qkv_and_conv_agree_with_upstream_explicit_qk_split():
    from q4nx.models.qwen35 import untile_qkv
    from q4nx.models.qwen35_wide import restore_order
    x = torch.arange(10240*4).reshape(10240, 4)
    expected = untile_qkv(x, 4096, 3, 128)
    assert torch.equal(restore_order('linear_attn.qkv_proj.weight', x), expected)
    assert torch.equal(restore_order('linear_attn.ssm_conv1d.weight', x), expected.T)


def test_streamed_safetensors_roundtrip_and_incomplete_write(tmp_path):
    from q4nx.streaming import TensorWriter
    out=tmp_path/'model.q4nx'
    with TensorWriter(out) as writer:
        writer.add('embed',(torch.arange(12).reshape(3,4).to(torch.bfloat16),torch.ones(2,4,dtype=torch.bfloat16)))
        writer.add('scale',[torch.tensor([1.,-2.])])
    data=load_file(out)
    assert data['embed'].shape==(5,4)
    assert torch.equal(data['embed'][:3],torch.arange(12).reshape(3,4).to(torch.bfloat16))
    assert data['scale'].tolist()==[1.,-2.]
    bad=tmp_path/'bad.q4nx'
    with pytest.raises(ValueError):
        with TensorWriter(bad) as writer:
            writer.add('x',[torch.ones(1,2),torch.ones(1,3)])
    assert not bad.exists()
