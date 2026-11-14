"""Wide prefill references must preserve FP32 history and paired KV across blocks."""
import numpy as np
import pytest
import replica_block as rb


def delta(c,cs,state,**kw):
    return rb.deltanet_block(c['qkv'],c['z'],c['xn'],c['convw'],c['Wa'],c['Wb'],c['A'],c['dtb'],c['nw'],cs,state,
                            key_heads=c['key_heads'],value_heads=c['value_heads'],head_dim=c['head_dim'],**kw)


def test_fp32_history_preserves_non_bf16_low_bits_and_short_carry():
    c=rb.random_case(12);cs=c['conv_state'].copy()+.00013;state=c['S'].copy();old=cs.copy()
    delta(c,cs,state,t_real=1,conv_f32=True)
    np.testing.assert_array_equal(cs[:-1],old[1:])
    np.testing.assert_array_equal(cs[-1],c['qkv'][0])
    assert np.any(cs[-1]!=rb.bf16_round(cs[-1]))


def test_fp32_block_matches_partitioned_reference():
    c=rb.random_case(7);a=c['conv_state'].copy();b=a.copy();sa=c['S'].copy();sb=sa.copy()
    full=delta(c,a,sa,conv_f32=True)
    parts=[]
    for t in range(c['T']):
        part=dict(c,**{k:c[k][t:t+1] for k in ('qkv','z','xn')})
        parts.append(delta(part,b,sb,conv_f32=True))
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(sa,sb,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(full,np.vstack(parts),rtol=1e-12,atol=1e-12)


def attn(c,kv,pos=None,**kw):
    return rb.attention_block(*(c[k] for k in ('q','k','v','gate','qn','kn','inv_freq')),kv,
                              c['pos0'] if pos is None else pos,nh=c['nh'],kvh=c['kvh'],hd=c['hd'],**kw)


def test_paired_cache_layout_tail_and_partition_identity():
    c=rb.random_case(5);w=c['kvh']*c['hd']
    kv=np.full((c['pos0']+c['T']+1,4*w),.125)
    a=kv.copy();b=kv.copy();out=attn(c,a,kv_pair=True)
    parts=[]
    for t in range(c['T']):
        one=dict(c,**{k:c[k][t:t+1] for k in ('q','k','v','gate')})
        parts.append(attn(one,b,c['pos0']+t,kv_pair=True))
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(out,np.vstack(parts),rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(a[:c['pos0']],kv[:c['pos0']]);np.testing.assert_array_equal(a[-1],kv[-1])
    values=a[c['pos0']:c['pos0']+c['T'],2*w:3*w]+a[c['pos0']:c['pos0']+c['T'],3*w:]
    np.testing.assert_allclose(values,c['v'],rtol=2e-5,atol=2e-5)
    assert np.any(a[c['pos0']:,3*w:]!=0)


def test_new_modes_ignore_poisoned_padding():
    c=rb.random_case(6);real=2
    for k in ('qkv','z','xn','q','k','v','gate'):c[k][real:]=np.nan
    cs=c['conv_state'].copy();state=c['S'].copy()
    out=delta(c,cs,state,t_real=real,conv_f32=True)
    assert np.isfinite(out).all() and np.all(out[real:]==0)
    kv=np.zeros((c['pos0']+c['T'],4*c['kvh']*c['hd']))
    out=attn(c,kv,t_real=real,kv_pair=True)
    assert np.isfinite(out).all() and np.all(out[real:]==0)
    assert np.all(kv[c['pos0']+real:]==0)


@pytest.mark.parametrize('bad',[-1,7])
def test_invalid_real_count_is_rejected(bad):
    c=rb.random_case()
    with pytest.raises(ValueError):delta(c,c['conv_state'].copy(),c['S'].copy(),t_real=bad,conv_f32=True)
    with pytest.raises(ValueError):attn(c,np.zeros((10,4*c['kvh']*c['hd'])),t_real=bad,kv_pair=True)
