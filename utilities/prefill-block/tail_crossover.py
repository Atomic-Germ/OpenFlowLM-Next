"""Phase 27.4: measure the block-prefill tail crossover on the PRODUCTION runtime path.

A logical tail smaller than the physical block still costs a full-width dispatch, so a
short prompt can be cheaper one token at a time. That crossover is currently a constant
(64) inherited from the Qwen3.6-35B MoE work, where the reason was expert-weight streaming
-- a dense model has no such reason, so the number must be re-measured rather than reused.

Method: the same prompt length, twice, on the same hardware session, with the NPU
quiesced:

    A  sequential      OFLM_OPEN_GEMM_BLOCK=0        one core.step() per token
    B  padded block    OFLM_OPEN_GEMM_BLOCK=1        one full 256-row dispatch

Both go through `oflm`'s own Engine::prefill -- the production path, not the standalone
harness -- so the number applies to what ships.

    python tail_crossover.py --lengths 1 2 4 8 16 32 64 96 128 160 192 224 255

Records wall time, per-prompt dispatch count and the first generated token, so a speed
claim cannot be made against a run that computed something else.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFLM = ROOT / 'build/src/oflm'
LOCK = ROOT / 'Models/qwen38-27b/.npu.lock'
REAL_VOCAB = 248077


def fixture(n: int, seed: int = 26026, pool: int = 512) -> list[int]:
    """The same distinct deterministic draw the Phase 26 oracle uses."""
    import numpy as np
    if n > pool:
        raise SystemExit(f'{n} exceeds the {pool}-id pool')
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(2, REAL_VOCAB), size=pool, replace=False)
    return [int(i) for i in ids[:n]]


def run(n: int, block: bool, model: str, kernels: str, max_tokens: int) -> dict:
    ids = fixture(n)
    env = dict(os.environ)
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    # =0 is the only way to force sequential; anything else leaves the default on.
    env['OFLM_OPEN_GEMM_BLOCK'] = '1' if block else '0'
    env.pop('OFLM_OPEN_GEMM_BLOCK_MIN', None)      # the policy decides, not an override
    env['OFLM_OPEN_DISPATCH_LOG'] = '1'
    cmd = [str(OFLM), 'run', '--model', model, '--kernels', kernels,
           '--ids', ','.join(str(i) for i in ids),
           '--max-tokens', str(max_tokens), '--pmode', 'none']
    t0 = time.monotonic()
    with LOCK.open('a') as handle:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    if proc.returncode != 0:
        tail = '\n'.join(text.strip().splitlines()[-10:])
        raise SystemExit(f'n={n} block={block} failed (rc {proc.returncode}):\n{tail}')
    tokens = [int(x) for x in re.findall(r'^token (\d+)$', text, re.M)]
    calls = sum(int(c) for c in re.findall(r'(\d+) calls', text))
    # The engine's own stage line: its total is the request, not the process.
    totals = [float(x) for x in re.findall(r't_real=\d+: ([\d.]+) ms', text)]
    return dict(n=n, block=block, wall_s=round(wall, 3),
                request_ms=round(sum(totals), 1) if totals else None,
                dispatches=calls, first_token=tokens[0] if tokens else None,
                tokens=tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lengths', nargs='+', type=int,
                    default=[1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 255])
    ap.add_argument('--reps', type=int, default=2,
                    help='two runs per point; one cannot tell a real difference from noise')
    ap.add_argument('--max-tokens', type=int, default=1)
    ap.add_argument('--model', default=str(ROOT / 'Models/qwen38-27b/Qwen3.8-27B-NPU2'))
    ap.add_argument('--kernels', default=str(ROOT / 'Models/qwen38-27b/production-v2/kernels-model'))
    ap.add_argument('--out', type=Path, default=HERE / 'tail-crossover.json')
    a = ap.parse_args()
    if not OFLM.is_file():
        raise SystemExit(f'missing {OFLM}; build oflm first')

    rows = []
    print(f'{"n":>4s} {"seq ms":>9s} {"blk ms":>9s} {"blk/seq":>8s} '
          f'{"seq disp":>9s} {"blk disp":>9s}  first token')
    for n in a.lengths:
        seq = [run(n, False, a.model, a.kernels, a.max_tokens) for _ in range(a.reps)]
        blk = [run(n, True, a.model, a.kernels, a.max_tokens) for _ in range(a.reps)]
        # Median of the request time; fall back to wall only if the stage line was absent.
        ts = [r['request_ms'] if r['request_ms'] is not None else r['wall_s'] * 1000 for r in seq]
        tb = [r['request_ms'] if r['request_ms'] is not None else r['wall_s'] * 1000 for r in blk]
        ms, mb = statistics.median(ts), statistics.median(tb)
        same = seq[0]['first_token'] == blk[0]['first_token']
        row = dict(n=n, seq_ms=round(ms, 1), block_ms=round(mb, 1),
                   ratio=round(mb / ms, 3) if ms else None,
                   seq_dispatch=seq[0]['dispatches'], block_dispatch=blk[0]['dispatches'],
                   seq_first_token=seq[0]['first_token'],
                   block_first_token=blk[0]['first_token'],
                   first_token_agrees=same,
                   seq_runs=[r['request_ms'] for r in seq],
                   block_runs=[r['request_ms'] for r in blk])
        rows.append(row)
        print(f'{n:4d} {ms:9.1f} {mb:9.1f} {row["ratio"]:8.2f} '
              f'{row["seq_dispatch"]:9d} {row["block_dispatch"]:9d}  '
              f'{"same" if same else "DIFFER"}')

    # The crossover is the smallest n where the block route is no slower than sequential.
    cross = None
    for r in rows:
        if r['ratio'] is not None and r['ratio'] <= 1.0:
            cross = r['n']
            break
    result = dict(tool='tail_crossover', version=1, phase=27,
                  scope='production Engine::prefill, one pass per point, median of reps',
                  model=a.model, kernels=a.kernels, reps=a.reps,
                  max_tokens=a.max_tokens,
                  physical_block=256, lengths=a.lengths, rows=rows,
                  crossover_n=cross,
                  crossover_note='smallest measured n where padded-block cost <= sequential',
                  all_first_tokens_agree=all(r['first_token_agrees'] for r in rows))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + '\n')
    print(f'\ncrossover: n = {cross}  (block no slower than sequential from here up)')
    print(f'first tokens agree at every length: {result["all_first_tokens_agree"]}')
    print(f'wrote {a.out}')
    return 0 if result['all_first_tokens_agree'] else 1


if __name__ == '__main__':
    sys.exit(main())
