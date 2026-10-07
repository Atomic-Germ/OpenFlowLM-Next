"""Block trace lane order and localization against original Q4 bytes."""
import sys
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'utilities'))


def test_blocks_restore_even_odd_lanes_and_final_interface():
    from wide_down_blocks import decode
    raw=np.arange(16*1120,dtype=np.float32).reshape(16,1120)
    blocks,high,low=decode(raw.ravel())
    assert blocks.shape==(128,4,32)
    assert high.shape==low.shape==(16,32)
    for lane in range(32):
        perm=lane//2+16*(lane%2)
        assert blocks[9,2,lane]==raw[1,128+64+perm]
        assert high[0,lane]==raw[0,1024+perm]
        assert high[-1,lane]==raw[-1,1024+lane]
        assert low[-1,lane]==raw[-1,1056+perm]
    with pytest.raises(ValueError): decode(raw.ravel()[:-1])


def test_activation_sum_components_follow_k_blocks_without_lane_permutation():
    from wide_down_blocks import block_sums
    raw=np.arange(16*1120,dtype=np.float32).reshape(16,1120)
    got=block_sums(raw.ravel())
    assert got.shape==(128,3)
    np.testing.assert_array_equal(got[53],raw[6,[1088+5,1096+5,1104+5]])


def test_block_reference_uses_original_packed_weights_and_bf16_activation():
    from wide_down_blocks import reference
    from q4_1_pack import random_q4_1_blocks,pack_q4_1_pool,dequant_chunk
    from ml_dtypes import bfloat16
    rng=np.random.default_rng(4872)
    pool=pack_q4_1_pool(random_q4_1_blocks(64,4096,rng),2)
    tiles=np.concatenate([pool[i*10240:i*10240+5120] for i in range(16)])
    x=rng.normal(size=4096).astype(np.float32)
    got=reference(tiles,x)
    w=np.concatenate([dequant_chunk(tiles[i*5120:(i+1)*5120]) for i in range(16)],axis=1).astype(np.float64)
    want=(w.reshape(32,128,32)*x.astype(bfloat16).astype(np.float64).reshape(1,128,32)).sum(axis=2).T
    np.testing.assert_array_equal(got,want)


def test_block_replay_rejects_tampering_guards_and_stale_zero_state(tmp_path):
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('block_diagnosis',root/'utilities/diagnose-wide-down-blocks.py')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    np.zeros(4096,np.float32).tofile(tmp_path/'x.bin')
    np.zeros(16*5120,np.uint8).tofile(tmp_path/'weights.bin')
    np.save(tmp_path/'reference.npy',np.zeros((128,32)))
    np.save(tmp_path/'segment.npy',np.zeros((2,32),np.float32))
    meta=dict(segment=3,channel=4872,sha256={p.name:m.down.ffn.sha(p) for p in tmp_path.iterdir()})
    (tmp_path/'blocks-fixture.json').write_text(json.dumps(meta))
    raw=bytes(m.TRACE_BYTES)+m.GUARD
    for i in range(3): (tmp_path/f'trace{i}.bin').write_bytes(raw)
    assert m.compare(tmp_path)==0
    wrong=bytearray(raw);wrong[:4]=np.array([1],np.float32).tobytes()
    (tmp_path/'trace1.bin').write_bytes(wrong)
    assert m.compare(tmp_path)==1
    (tmp_path/'trace1.bin').write_bytes(raw)
    (tmp_path/'trace0.bin').write_bytes(raw[:-64]+bytes(64))
    with pytest.raises(ValueError,match='canary'): m.compare(tmp_path)
    (tmp_path/'trace0.bin').write_bytes(raw)
    (tmp_path/'weights.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='fixture changed'): m.compare(tmp_path)
