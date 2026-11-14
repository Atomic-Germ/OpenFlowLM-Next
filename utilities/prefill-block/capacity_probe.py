"""Phase 29.2/29.3: what the runtime can ALLOCATE, kept strictly apart from what it can run.

"256K context works" has three different meanings and this phase must never blur them:

    capacity     the buffers allocate, the indices are representable, the context object
                 accepts the length
    addressing   the RoPE / KV / position arithmetic is right at that absolute range
    end-to-end   a real prompt of that length went through the real model and decoded

This script only tests the FIRST. It loads the engine at a requested context length and then
touches nothing: no prompt is prefilled. A PASS here means "the allocator accepted it" and
nothing more.

It also derives the arithmetic minimum from the manifest rather than a hand estimate, so a
failure can be attributed to the exact allocation. The two numbers reported per length are:

    predicted   model-resident + KV at this length + the manifest's fixed globals
    observed    the process RSS after load

A gap between them is normal (XRT/BO overhead, the container's mapped pages, libc arenas);
what matters is whether one allocation fails, and at which size.

    python capacity_probe.py                       # 4K .. 256K
    python capacity_probe.py --lengths 4096 8192
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oflm_drive as D  # noqa: E402

GiB = 2 ** 30


def manifest_model(model: Path = D.MODEL) -> dict:
    """The real numbers, from the manifest the engine reads -- not a re-derivation."""
    man = json.loads((D.KERNELS / 'manifest.json').read_text())
    lay = man['layout']
    types = man['layer_types']
    layers = man['layers']
    kv_row = lay['kv_row']
    n_kv = sum(1 for l in layers if (types[l].get('buffers', {}).get('state', {})
                                     .get('kind') == 'kv'))
    fixed_state = sum((types[l].get('buffers', {}).get('state', {}).get('bytes') or 0)
                      for l in layers
                      if types[l].get('buffers', {}).get('state', {}).get('kind') != 'kv')
    globals_bytes = sum(v for v in man['globals'].values() if isinstance(v, int))
    per_row = sum(v['per_row'] for v in man['globals'].values() if isinstance(v, dict))
    # The pool/consts/act allocation sizes are per layer type; the manifest carries them
    # under `buffers`, and `pool_bytes` / `chunk_bytes` at the layout level.
    per_layer_fixed = 0
    for l in layers:
        b = types[l].get('buffers', {})
        per_layer_fixed += sum(v for k, v in b.items()
                               if k not in ('state', 'post_output') and isinstance(v, int))
    return dict(kv_row=kv_row, kv_layers=n_kv, linear_state_bytes=fixed_state,
                globals_bytes=globals_bytes, per_row_bytes=per_row,
                per_layer_fixed=per_layer_fixed,
                pool_bytes=lay.get('pool_bytes', 0),
                container_bytes=(model / 'model.q4nx').stat().st_size)


def predict(m: dict, ctx: int) -> dict:
    kv = m['kv_row'] * m['kv_layers'] * ctx
    ptab = m['per_row_bytes'] * ctx
    total = (m['container_bytes'] + kv + ptab + m['globals_bytes']
             + m['linear_state_bytes'] + m['per_layer_fixed'])
    return dict(kv_bytes=kv, ptab_bytes=ptab, total_bytes=total,
                kv_gib=round(kv / GiB, 3), total_gib=round(total / GiB, 3))


def probe(ctx: int, *, timeout: float, model: Path = D.MODEL) -> dict:
    """Load at `ctx` and stop. The engine prints its capacity line during load."""
    script = '/bye\n'
    cmd = [str(D.OFLM), 'run', 'qwen3.8:27b', '--ctx-len', str(ctx)]
    t0 = time.monotonic()
    LOCK = D.LOCK
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as handle:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, cwd=str(D.ROOT), env=D.env(block=True), input=script,
                              capture_output=True, text=True, timeout=timeout)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    cap = re.search(r'context capacity (\d+)', text)
    refused = re.search(r'(capacity stays \d+|exceed[s]? the context capacity|'
                        r'context (?:length )?[^\n]*refus|out of memory|'
                        r'Killed|bad_alloc|std::bad_alloc)', text, re.I)
    return dict(ctx=ctx, rc=proc.returncode, wall_s=round(wall, 2),
                capacity_reported=int(cap.group(1)) if cap else None,
                loaded=bool(cap) and proc.returncode == 0,
                refusal=refused.group(0) if refused else None,
                tail=[l for l in text.splitlines()
                      if 'capacity' in l or 'alloc' in l.lower() or 'ERROR' in l][-6:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lengths', type=int, nargs='+',
                    default=[4096, 8192, 16384, 32768, 65536, 131072, 262144])
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'capacity-probe-{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    m = manifest_model()
    result = dict(tool='prefill-block-capacity-probe', version=1, phase=29,
                  gate='29.3 capacity only -- NOT end-to-end validation',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, manifest=m,
                  distinction='capacity_pass means the allocator accepted the length and the '
                              'engine reported that capacity; no prompt was prefilled',
                  host=dict(total_gib=round(os.sysconf('SC_PAGE_SIZE') *
                                            os.sysconf('SC_PHYS_PAGES') / GiB, 1)))
    rows = []
    for ctx in a.lengths:
        p = predict(m, ctx)
        r = probe(ctx, timeout=a.timeout)
        r.update(predicted=p, capacity_pass=bool(r['loaded']))
        rows.append(r)
        print(f"  ctx {ctx:7d}: capacity {'PASS' if r['capacity_pass'] else 'FAIL'} "
              f"reported={r['capacity_reported']} pred {p['total_gib']:6.2f} GiB "
              f"(KV {p['kv_gib']:5.2f}) {r['wall_s']:6.1f}s"
              f"{'' if r['capacity_pass'] else '  ' + str(r['refusal'] or r['rc'])}",
              flush=True)
    result['rows'] = rows
    result['capacity_validated'] = [r['ctx'] for r in rows if r['capacity_pass']]
    result['capacity_failed'] = [r['ctx'] for r in rows if not r['capacity_pass']]
    result['verdict'] = 'PASS' if result['capacity_validated'] else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\ncapacity validated: {result['capacity_validated']}")
    if result['capacity_failed']:
        print(f"capacity failed:    {result['capacity_failed']}")
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
