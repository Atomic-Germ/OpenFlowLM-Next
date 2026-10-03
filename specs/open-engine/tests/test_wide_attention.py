"""Probe the actual ax attention worker, including all broadcast gate slices."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from recipes.qwen36moe import attn
from recipes.spec import ModelSpec

ROOT = Path(__file__).resolve().parents[3]


def support():
    spec = importlib.util.spec_from_file_location('attn_probe_support', ROOT / 'open_kernels/designs/attn/probe_support.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference():
    spec = importlib.util.spec_from_file_location('wide_attn_ref', ROOT / 'utilities/wide_attention_reference.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('fixture', ['config_qwen38_27b.json', 'config_qwen35_9b.json'])
def test_production_worker_consumes_broadcast_and_owns_correct_gate_heads(fixture):
    s = ModelSpec.from_hf_config(json.loads((Path(__file__).parent / 'fixtures' / fixture).read_text()))
    d = attn(s)
    fn = support().worker(d, range)
    for c in range(d.ACORES):
        events = []
        stream = ['meta', 'pos'] + ['q']*d.Q_AIN_ELEMS + ['k']*d.K_AIN_ELEMS + ['v']*d.K_AIN_ELEMS
        stream += ['cached_k', 'cached_v'] * 3
        stream += list(range(d.NH // d.HPE))
        class Input:
            def acquire(self, n): return stream[0] if n == 1 else stream[:n]
            def release(self, n): del stream[:n]
        class Output:
            def acquire(self, n): return np.zeros(d.KVW)
            def release(self, n): events.append('output')
        def meta(a, b, qn, kn, cs, pb): pb[1] = 3
        def noop(*args): pass
        def step(*args): events.append(('step', args[-1]))
        def finish(acc, ml, g0, g1, out, hp): events.append(('gate', g0, g1, hp))
        out = Output()
        fn(Input(), out if c == 0 else None, out, None, None, None, None, None,
           np.zeros(d.KVW), np.zeros(d.KVW), None, None, np.zeros(4, int),
           meta, noop, noop, noop, noop, step, step, finish, None, c)
        assert not stream
        assert [e for e in events if isinstance(e, tuple) and e[0] == 'gate'] == [('gate', 2*c, 2*c+1, 0)]
        assert [e for e in events if isinstance(e, tuple) and e[0] == 'step'] == [('step', c*d.NHL)]*4
        assert events.count('output') == (3 if c == 0 else 1)


def test_probe_schedule_drains_all_heads_and_streams_interleaved_cache():
    s = ModelSpec.from_hf_config(json.loads((Path(__file__).parent / 'fixtures/config_qwen38_27b.json').read_text()))
    d = attn(s)
    events = []
    class Pipe:
        def drain(self, ep, tensor, tap): events.append(('drain', ep, tensor, tap))
        def fill(self, ep, tensor, tap): events.append(('fill', ep, tensor, tap))
        def finish(self): events.append(('finish',))
    support().sequence(d, 17, Pipe(), lambda n,o,l:(n,o,l), 'meta','qg','kvn','cache','new','og','in',list(range(d.ACORES)))
    drains = [e for e in events if e[0] == 'drain']
    assert drains[0] == ('drain', 0, 'new', (2048, 0, 2048))
    assert [e[3] for e in drains[1:]] == [(6144, c*1024, 1024) for c in range(6)]
    fills = [e for e in events if e[0] == 'fill']
    assert [e[2] for e in fills] == ['meta','qg','kvn','kvn','cache','qg']
    assert fills[-2][3] == (17*2048, 0, 17*2048)
    assert fills[-1][3] == (12288, 6144, 6144)


def test_reference_group_mapping_mask_and_gate_at_zero_scores():
    from ml_dtypes import bfloat16
    nh, kvh, hd = 24, 4, 256
    qg = np.zeros((2,nh,hd), np.float32)
    kvn = np.zeros((2,kvh,hd), np.float32)
    kvn[1] = np.arange(1,5)[:,None]
    norms = np.ones((2,hd), bfloat16)
    cs = np.r_[np.ones(32),np.zeros(32)].astype(np.float32)
    cache = np.full((3,2,kvh,hd), np.nan, bfloat16)
    new, og = reference().decode(qg,kvn,norms,cs,cache,0)
    np.testing.assert_array_equal(og, np.repeat(kvn[1],6,axis=0)*.5)
    cache[0,0] = 0
    cache[0,1] = np.arange(5,9)[:,None]
    qg[1] = np.log(3)
    _, og = reference().decode(qg,kvn,norms,cs,cache,1)
    expected = np.repeat((kvn[1]+cache[0,1].astype(np.float64))*.5,6,axis=0)
    np.testing.assert_allclose(og,expected/(1+np.exp(-qg[1].astype(np.float64))),rtol=1e-14)


def test_reference_partial_rope_rotates_only_first_64_dimensions():
    x = np.arange(1,257,dtype=np.float64)[None,:]
    weights = np.ones(256)
    cs = np.r_[np.zeros(32),np.ones(32)]
    got = reference().norm_rope(x,weights,cs)
    norm = x/np.sqrt(np.mean(x*x)+1e-6)
    np.testing.assert_array_equal(got[:,:32],-norm[:,32:64])
    np.testing.assert_array_equal(got[:,32:64],norm[:,:32])
    np.testing.assert_array_equal(got[:,64:],norm[:,64:])


def test_fixture_preserves_device_cache_between_warm_tokens(tmp_path):
    spec = importlib.util.spec_from_file_location('attn_fixture', ROOT / 'utilities/test-wide-attention.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ('final.xclbin','insts.bin'):
        (tmp_path/name).write_bytes(b'test artifact')
    mod.prepare(tmp_path, 'config_qwen38_27b.json', 17)
    cfg = (tmp_path/'attention.cfg').read_text().splitlines()
    copies = [line for line in cfg if line.startswith('copy cache ')]
    assert copies == [f'copy cache {i*4096} new 0 4096' for i in range(8)]
    assert not any('ref' in line for line in cfg)
    assert cfg.count('load cache cache-warm.bin') == 1


def test_metric_serializes_failed_and_zero_gates():
    mod=reference()
    for got,ref,passed in [(np.zeros(32),np.zeros(32),True),
                           (np.ones(32)*2,np.ones(32),False),
                           (np.full(32,np.nan),np.ones(32),False)]:
        result=mod.metric(got,ref,1e-2)
        assert json.loads(json.dumps(result))['passed'] is passed
