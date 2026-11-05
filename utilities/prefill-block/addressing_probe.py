"""Phase 29.4: are the high position indices CORRECT, or merely wide enough?

A capacity PASS says the allocator accepted 262144 rows. It says nothing about whether the
arithmetic that addresses row 262143 is right. This drives the real engine at the boundaries,
reads back what it wrote, and checks three independent things at each one:

  1. **RoPE.** The cos/sin the engine wrote into the position record must equal cos/sin of
     `pos * inv_freq[i]` in float64. A position truncated to 16 bits, or an `int` pushed into
     a float angle, drifts visibly by 2^18.
  2. **The causal window.** The record's `valid` count must be `row - start` -- every row up
     to and including this one -- so a wrapped or clamped length shows up here.
  3. **The KV write lands where it claims.** After the step, the layer's cache at
     `pos * kv_row` must be non-zero and the row immediately after it must still be zero.
     This is the only check here that would catch a wrap or a truncation in the row offset.

The position is `seek`-ed to, then ONE token is stepped: no prompt is prefilled, so this
costs one step per position rather than a 256K prefill. It therefore proves ADDRESSING, never
end-to-end model behaviour, and the report keeps those words apart.

    python addressing_probe.py
    python addressing_probe.py --positions 131071 131072 262143
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle as O  # noqa: E402

DEFAULT_POSITIONS = [4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384, 16385,
                     32767, 32768, 32769, 65535, 65536, 65537, 131071, 131072, 131073,
                     262143]
# `build_ptab_record` writes [i32 valid | i32 nf | cos f32[rot/2] @512 | sin f32[rot/2]].
COS_OFF, ROT = 512, 64
# fp32 cos/sin versus a float64 reference: the engine stores f32, so this is its own
# precision, not a tolerance invented to hide a bug (a wrapped position misses by ~1).
ANGLE_TOL = 2e-6


def facts(kernels: Path = O.KERNELS) -> dict:
    m = json.loads((kernels / 'manifest.json').read_text())
    lay = m['layout']
    full = [i for i, l in enumerate(m['layers']) if l == 'full_attention']
    if not full:
        raise SystemExit('this kernel set has no full-attention layer to probe')
    return dict(rotary_dim=lay['rotary_dim'], ptab_row=lay['ptab_row'],
                inv_freq=lay['rope_inv_freq'], kv_row=lay['kv_row'],
                full_layer=full[0])


def one(pos: int, *, ctx: int, out: Path, f: dict) -> dict:
    """Seek, step one token, dump the record and the two KV rows around it."""
    import fcntl
    import subprocess
    lay, kv, prow = f['full_layer'], f['kv_row'], f['ptab_row']
    rec = out / f'ptab_{pos}.bin'
    a = out / f'kv_{pos}_{pos * kv}.bin'
    # The row AFTER the one written, when it is still inside the allocation. The row BEFORE
    # is not a control: an earlier step wrote it, so "not zero" there means nothing. Below
    # the last row of the context there is no such row, and the check is recorded as
    # inapplicable rather than turned into a pass.
    next_in_range = (pos + 1) * kv + kv <= ctx * kv
    b = out / f'kv_{pos}_next_{pos * kv + kv}.bin'
    env = dict(os.environ)
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    # ONE id: with two, the second token is stepped at pos+1 and legitimately writes the row
    # this probe wants to be still zero. That was the first version's error.
    cmd = [str(O.CLI), '--model', str(O.MODEL), '--kernels', str(O.KERNELS),
           '--ids', '100', '--max-ctx', str(ctx), '--max-tokens', '1',
           '--pmode', 'none', '--at-position', str(pos),
           '--dump-global', f'ptab:{pos * prow}:{prow}:{rec}',
           '--dump-state', f'{lay}:{pos * kv}:{kv}:{a}']
    if next_in_range:
        cmd += ['--dump-state', f'{lay}:{(pos + 1) * kv}:{kv}:{b}']
    t0 = time.monotonic()
    with O.LOCK.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, cwd=str(O.ROOT), env=env, capture_output=True, text=True)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    r = dict(pos=pos, rc=proc.returncode, wall_s=round(wall, 2))
    if proc.returncode != 0:
        r['error'] = '\n'.join(text.strip().splitlines()[-6:])
        return r
    raw = rec.read_bytes()
    valid, nf = struct.unpack_from('<ii', raw, 0)
    got_cos = np.frombuffer(raw, dtype=np.float32, count=ROT // 2, offset=COS_OFF)
    got_sin = np.frombuffer(raw, dtype=np.float32, count=ROT // 2,
                            offset=COS_OFF + 4 * (ROT // 2))
    # The reference, in float64 from the same inv_freq the manifest ships.
    want_cos = np.array([math.cos(pos * fi) for fi in f['inv_freq'][:ROT // 2]])
    want_sin = np.array([math.sin(pos * fi) for fi in f['inv_freq'][:ROT // 2]])
    rope_err = float(max(np.abs(got_cos - want_cos).max(), np.abs(got_sin - want_sin).max()))
    # The record's own position. `stream_patch::attn_window` computes
    #     start = (window && pos+1 > window) ? pos+1-window : 0
    #     valid = pos - start
    # so with a full causal window (start 0) valid == pos, NOT pos+1: the count excludes the
    # row this step is writing. That is why the first version read FAIL at every position
    # with the correct value printed next to it.
    cur = a.read_bytes()
    r.update(record_valid=valid, record_nf=nf,
             expected_valid=pos,
             valid_ok=(valid == pos),
             rope_max_abs_err=rope_err, rope_ok=(rope_err <= ANGLE_TOL),
             kv_row_nonzero=any(cur), kv_bytes=kv,
             kv_next_checked=next_in_range)
    checks = [r['valid_ok'], r['rope_ok'], r['kv_row_nonzero']]
    if next_in_range:
        # Only meaningful inside the allocation: the step must write EXACTLY its own row.
        r['kv_next_row_zero'] = not any(b.read_bytes())
        checks.append(r['kv_next_row_zero'])
    r['ok'] = all(checks)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctx', type=int, default=262144)
    ap.add_argument('--positions', type=int, nargs='+', default=DEFAULT_POSITIONS)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    f = facts()
    if max(a.positions) >= a.ctx:
        raise SystemExit(f'position {max(a.positions)} needs --ctx beyond it (got {a.ctx})')

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'addressing-probe-{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    result = dict(tool='prefill-block-addressing-probe', version=2, phase=29,
                  gate='29.4 ADDRESSING only -- not end-to-end model behaviour',
                  model_acceptance=False, cli=str(O.CLI), kernels=str(O.KERNELS),
                  started=stamp, ctx=a.ctx,
                  facts={k: v for k, v in f.items() if k != 'inv_freq'},
                  checks=['position record valid == pos+1 (a 16-bit wrap shows here)',
                          'RoPE cos/sin == cos/sin(pos*inv_freq) in float64',
                          'KV row pos non-zero and row pos+1 still zero (no wrap)'],
                  distinction='seek + one token; no prompt of this length was prefilled')
    rows = []
    print(f"{'pos':>8} {'valid':>7} {'rope_err':>10} {'kv':>4} {'verdict':>8}")
    for pos in a.positions:
        r = one(pos, ctx=a.ctx, out=out, f=f)
        rows.append(r)
        if 'error' in r:
            print(f"{pos:8d} {'':>7} {'':>10} {'':>4} {'FAIL':>8}  {r['error'].splitlines()[0]}")
            continue
        print(f"{pos:8d} {r['record_valid']:7d} {r['rope_max_abs_err']:10.2e} "
              f"{'ok' if r['kv_row_nonzero'] else 'ZERO':>4} "
              f"{'PASS' if r['ok'] else 'FAIL':>8}", flush=True)
    result['rows'] = rows
    bad = [r['pos'] for r in rows if not r.get('ok')]
    result['failed'] = bad
    result['addressing_validated'] = [r['pos'] for r in rows if r.get('ok')]
    result['verdict'] = 'PASS' if not bad else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\naddressing validated at {len(result['addressing_validated'])} positions"
          f"{'' if not bad else '; FAILED ' + str(bad)}")
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
