"""Independent packed-weight partials expose rounding before segment addition."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'utilities'))


def test_segment_reference_matches_dense_float64_without_partial_rounding():
    from wide_down_reference import partial_reference
    from q4_1_pack import random_q4_1_blocks, pack_q4_1_pool, dequant_pool
    from ml_dtypes import bfloat16
    rng = np.random.default_rng(381)
    pool = pack_q4_1_pool(random_q4_1_blocks(64,2304,rng),2)
    x = rng.normal(size=2304).astype(bfloat16).astype(np.float64)
    got = partial_reference(pool,x,64,2304,1024)
    w = dequant_pool(pool,64,2304,2).astype(np.float64)
    expected = np.stack([w[:,s:min(s+1024,2304)]@x[s:min(s+1024,2304)] for s in (0,1024,2048)])
    assert got.dtype==np.float64
    np.testing.assert_allclose(got,expected,rtol=0,atol=1e-13)


def test_rounded_partials_cannot_recover_discarded_low_bits():
    from wide_down_reference import reduction_diagnosis
    parts = np.array([[2.0**24+1],[-2.0**24],[3]],np.float64)
    d = reduction_diagnosis(parts)
    assert d['exact'][0] == 4
    assert d['rounded_partials'][0] == 3
    assert d['sequential'][0] == 3


def test_segment_carry_rejects_missing_prerequisite_before_build(tmp_path):
    import subprocess
    root = Path(__file__).resolve().parents[3]
    for args, message in [(['--scope','ffn','--segment-carry'], 'requires --block-carry'),
                          (['--scope','projection','--projection-correction','--product-correction',
                            '--block-carry','--segment-carry'], 'requires --scope ffn')]:
        p = subprocess.run([sys.executable,str(root/'utilities/probe-qwen35-wide.py'),
                            *args,'--out',str(tmp_path/'absent')],capture_output=True,text=True)
        assert p.returncode==2 and message in p.stderr
        assert not (tmp_path/'absent').exists()
