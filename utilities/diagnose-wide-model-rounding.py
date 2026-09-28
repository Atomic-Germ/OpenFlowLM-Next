#!/usr/bin/env python3
"""Separate local RMSNorm rounding from propagated input error, offline only."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_model',HERE/'test-wide-full-model.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from wide_slice_reference import norm


def diagnose(out):
    meta = json.loads((out/'slice-fixture.json').read_text())
    params = {}
    def checked(name):
        if M.sha(out/name) != meta['fixtures'][name]:
            raise ValueError(f'fixture changed: {name}')
        return np.load(out/name)
    for name, digest in meta['kernels'].items():
        if M.sha(name) != digest:
            raise ValueError(f'kernel changed: {name}')
    def capture(name, dtype, size):
        raw = (out/name).read_bytes()
        if len(raw) != size+64 or raw[-64:] != M.GUARD:
            raise ValueError(f'{name}: size/canary')
        a = np.frombuffer(raw[:size],dtype)[:M.H]
        if not np.isfinite(a.astype(np.float32)).all():
            raise ValueError(f'{name}: nonfinite logical input')
        return a
    rows = []
    for case in meta['cases']:
        tag, i = case['tag'], case['layer']
        if i not in params: params[i] = dict(checked(f'layer{i}/params.npz'))
        ref = checked(f'{tag}-ref.npz')
        sizes = meta['outputs']['d' if case['kind']==M.LINEAR else 'a']
        for field, input_name, weight in (('xn',f'{tag}-input-x.bin','lnw'),
                                          ('xm',f'{tag}-got-res1.bin','postw')):
            x = capture(input_name,np.float32,M.H*4)
            got = capture(f'{tag}-got-{field}.bin',bfloat16,sizes[field])
            conditional = norm(x,params[i][weight]).view(np.uint16)
            expected = ref[field].astype(bfloat16).view(np.uint16)
            bits = got.view(np.uint16)
            rows.append(dict(tag=tag,field=field,
                device_vs_reference=int(np.count_nonzero(bits!=expected)),
                local_norm=int(np.count_nonzero(bits!=conditional)),
                propagated_input=int(np.count_nonzero(conditional!=expected))))
    result = dict(diagnostic_only=True,boundaries=rows,
                  totals={key:sum(row[key] for row in rows)
                          for key in ('device_vs_reference','local_norm','propagated_input')})
    print(json.dumps(result['totals']),flush=True)
    print('First changed boundaries:',rows[:6],flush=True)
    (out/'rounding-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    diagnose(p.parse_args().out)
