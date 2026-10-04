"""Segment traces must distinguish GEMV loss from cross-segment reduction loss."""
import sys
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'utilities'))


def test_trace_restores_core_segment_band_plane_order():
    from wide_down_trace import decode_trace
    # Two cores, three segments, two bands/core, four planes, 64 lanes.
    raw = np.arange(2*3*2*4*64, dtype=np.float32)
    got = decode_trace(raw, 256, 3, cores=2)
    assert got.shape == (3, 4, 256)
    for core in range(2):
        for seg in range(3):
            for band in range(2):
                for plane in range(4):
                    start = ((((core*3+seg)*2+band)*4+plane)*64)
                    np.testing.assert_array_equal(
                        got[seg, plane, core*128+band*64:core*128+(band+1)*64],
                        raw[start:start+64])
    with pytest.raises(ValueError, match='size'):
        decode_trace(raw[:-1], 256, 3, cores=2)
    raw[0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        decode_trace(raw, 256, 3, cores=2)


def test_diagnosis_separates_lost_segment_tail_from_lost_reduction():
    from wide_down_trace import analyze
    parts = np.array([[2**24+1], [-2**24], [3]], dtype=np.float64)
    trace = np.zeros((3, 4, 1), np.float32)
    trace[:, 0, 0] = [2**24, -2**24, 3]
    trace[:, 1, 0] = [1, 0, 0]
    trace[:, 2, 0] = [2**24, 0, 3]  # reducer incorrectly drops the tail
    r = analyze(trace, parts, 0)
    assert r['diagnostic_only']
    assert r['segment_error'] == [0, 0, 0]
    assert r['reduction_error'] == [-1, -1, -1]
    assert r['fp64_reduce_device_parts'] == 4
    assert r['device_final'] == 3
    trace[0, 1, 0] = 0
    r = analyze(trace, parts, 0)
    assert r['segment_error'] == [-1, 0, 0]
    assert r['reduction_error'] == [0, 0, 0]
    assert r['fp64_reduce_device_parts'] == 3
    with pytest.raises(ValueError, match='channel'):
        analyze(trace, parts, 1)


def test_replay_enforces_hashes_canaries_repeat_and_full_ffn_equivalence(tmp_path):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location('down_diagnosis',root/'utilities/diagnose-wide-down.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    n, k = 512, 64
    traces = np.zeros((8,1,1,4,64),np.float32)
    traces[:,:,:,0,:] = traces[:,:,:,2,:] = 1
    raw = np.concatenate([np.zeros(k,np.float32),np.ones(n,np.float32)]).tobytes()+m.ffn.GUARD
    for name in ('act','got0','got1'): (tmp_path/f'{name}.bin').write_bytes(raw)
    traw = traces.tobytes()+m.ffn.GUARD
    for name in ('trace0','trace1'): (tmp_path/f'{name}.bin').write_bytes(traw)
    np.ones(n,np.float32).tofile(tmp_path/'source-fo.bin')
    np.save(tmp_path/'parts.npy',np.ones((1,n),np.float64))
    meta=dict(n=n,k=k,segments=1,snapshots=1,act_bytes=(n+k)*4,
              trace_bytes=traces.nbytes,h_offset=0,out_offset=k*4,
              sha256={name:m.ffn.sha(tmp_path/name) for name in ('act.bin','parts.npy','source-fo.bin')})
    (tmp_path/'down-fixture.json').write_text(json.dumps(meta))
    assert m.compare(tmp_path,[0])==0
    (tmp_path/'trace1.bin').write_bytes(bytes(len(traw)))
    assert m.compare(tmp_path,[0])==1
    (tmp_path/'trace1.bin').write_bytes(traw)
    (tmp_path/'trace0.bin').write_bytes(traw[:-64]+bytes(64))
    with pytest.raises(ValueError,match='canary'): m.compare(tmp_path,[0])
    (tmp_path/'trace0.bin').write_bytes(traw)
    np.save(tmp_path/'parts.npy',np.zeros((1,n)))
    with pytest.raises(ValueError,match='fixture changed'): m.compare(tmp_path,[0])


def test_down_trace_rejects_wrong_scope_before_creating_output(tmp_path):
    import subprocess
    root = Path(__file__).resolve().parents[3]
    p=subprocess.run([sys.executable,str(root/'utilities/probe-qwen35-wide.py'),
                      '--scope','ffn','--down-trace','--out',str(tmp_path/'absent')],
                     capture_output=True,text=True)
    assert p.returncode==2 and '--down-trace requires' in p.stderr
    assert not (tmp_path/'absent').exists()
