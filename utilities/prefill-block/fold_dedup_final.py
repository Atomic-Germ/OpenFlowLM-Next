"""Fresh PR4B numerical gates; archived control, no timing or recovery actions."""
import argparse
import os
from pathlib import Path

import fold_dedup_continue as L
import fold_dedup_experiment as X
import pipeline_capture as C
import pipeline_evidence as E


def numerical(out):
    out.mkdir(parents=True, exist_ok=False)
    L.OUT = out / 'isolation'
    L.OUT.mkdir()
    a, b = C.fixture(20), C.fixture(13, seed=26027)
    records = {}
    for arm in ('original', 'on'):
        fresh = L.list_capture(arm, [a], f'{arm}-fresh-a')
        aba = L.list_capture(arm, [a, b, a], f'{arm}-aba')
        L.compare_isolation(fresh, aba)
        records[arm] = aba
    for name in records['original']['outputs']:
        E.require(records['original']['outputs'][name]['sha256'] ==
                  records['on']['outputs'][name]['sha256'], 'control/candidate list mismatch: ' + name)
    E.require(records['original']['runtime']['before'] == records['on']['runtime']['before'],
              'control/candidate runtime mismatch')
    C.write_json(out / 'isolation-passed.json', dict(passed=True,
        scope='layer-major list route; full real-vocabulary prompt logits, 17 IDs, fresh A/A-B-A',
        state_scope='No per-request state dumps; not all-layer state isolation'))
    work = out / 'boundary'
    work.mkdir()
    for n in (95, 96, 97, 255, 256, 257, 511, 512, 513):
        control = X.correctness_capture(n, 'original', output_root=work)
        candidate = X.correctness_capture(n, 'on', output_root=work)
        X.equal(control, candidate)
        C.write_json(out / f'boundary-{n}-passed.json', dict(passed=True, tokens=n,
            scope='64 layers; final prompt and 17 continuation logits; sampled state layers 0/3/62/63'))
    C.write_json(out / 'numerical-passed.json', dict(passed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib'
    os.environ['PATH'] = '/opt/xilinx/xrt/bin:' + os.environ['PATH']
    numerical(args.out.absolute())
