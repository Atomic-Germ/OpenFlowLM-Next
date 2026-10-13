"""Optional boundary captures preserve the original dispatch sequence and state."""
from pathlib import Path
import pytest
from make_decode import build_cfg
from recipes.load import load_spec
from recipes.manifest import manifest
ROOT=Path(__file__).resolve().parents[3]


def test_boundary_capture_is_opt_in_and_preserves_program(tmp_path):
    s=load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json');m=manifest(s,max_ctx=8)
    base=build_cfg(m,4,3,tmp_path,tmp_path,8)
    new=build_cfg(m,4,3,tmp_path,tmp_path,8,capture_boundaries=True)
    additions=[x for x in new.splitlines() if x not in base.splitlines()]
    assert len(additions)==12  # 3 linear states + final norm, at all 3 positions
    assert all(x.startswith(('dump state','dump hn')) for x in additions)
    assert '\n'.join(x for x in new.splitlines() if x not in additions)+'\n'==base
    for t in range(3):
        suffix='' if not t else f'_t{t}'
        assert f'dump state0 {tmp_path}/y_state0{suffix}.bin' in new
        assert f'dump hn {tmp_path}/y_norm{suffix}.bin 10240' in new
    assert 'dump state3 ' not in new  # full-attention KV is not recurrent DeltaNet state
    assert new.index('dump hn ')>new.index('run ln ')
    assert new.index('dump hn ')<new.index('run lm ')


def test_boundary_capture_refuses_an_unrecognized_tail(tmp_path):
    s=load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json');m=manifest(s,max_ctx=8)
    m['tail']=[]
    with pytest.raises(ValueError,match='tail'):build_cfg(m,4,1,tmp_path,tmp_path,8,capture_boundaries=True)
