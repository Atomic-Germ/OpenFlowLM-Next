"""Phase 27.15: the block route's cost through the shipped `oflm`, by prompt length.

This is a *route* benchmark, not a throughput claim for the product: it drives `oflm run`
at a set of exact prompt lengths and reports wall time and the per-request prefill/decode
split the runner prints. The sequential side of the comparison is the 27.6 matrix, which
measured both routes at the boundary; this fills in lengths the matrix does not cover.

The physical block is 256, so the shape to expect is a staircase: cost rises when another
block is needed and is roughly flat between those points. A linear ramp instead would mean
the block route was not the one taken.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oflm_drive as D  # noqa: E402

TEMPLATE_OFFSET = 11


def blocks_for(prefilled: int, physical: int = 256) -> int:
    """How many full-width passes a prompt of this length needs."""
    return max(1, -(-prefilled // physical))


def measure(n: int, *, gen_lim: int, ctx: int, block: bool) -> dict:
    tok = D.Tokenizer()
    text = tok.exact_text(n)
    rec = D.run([text], block=block, gen_lim=gen_lim, ctx=ctx, tag=f'b{n}')
    prefilled = D.prefilled_tokens(rec)
    return dict(text_tokens=n, prefilled=prefilled, expected=n + TEMPLATE_OFFSET,
                blocks=blocks_for(prefilled), rc=rec['rc'], wall_s=rec['wall_s'],
                output_len=len(rec['output']), reached_model=prefilled >= n,
                matches_expected=prefilled == n + TEMPLATE_OFFSET)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lengths', type=int, nargs='+', default=[128, 1024, 2048])
    ap.add_argument('--gen-lim', type=int, default=8)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--sequential', action='store_true',
                    help='also time the sequential route at each length (expensive)')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'oflm-bench-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-oflm-bench', version=1, phase=27,
                  gate='27.15 production benchmark', model_acceptance=False,
                  binary=str(D.OFLM), kernels=str(D.KERNELS), started=stamp,
                  template_offset=TEMPLATE_OFFSET, gen_lim=a.gen_lim, ctx=a.ctx,
                  note='route cost through oflm run; the paired A/B is the 27.6 matrix')
    rows = []
    for n in a.lengths:
        b = measure(n, gen_lim=a.gen_lim, ctx=a.ctx, block=True)
        row = dict(block=b)
        if a.sequential:
            row['sequential'] = measure(n, gen_lim=a.gen_lim, ctx=a.ctx, block=False)
            s = row['sequential']
            row['ratio'] = round(b['wall_s'] / s['wall_s'], 3) if s['wall_s'] else None
        rows.append(row)
        line = (f"  n={n:5d} {b['prefilled']:5d}tok {b['blocks']}blk "
                f"{b['wall_s']:8.2f}s")
        if a.sequential:
            line += f"   seq {row['sequential']['wall_s']:8.2f}s  ratio {row['ratio']}"
        print(line, flush=True)
    result['rows'] = rows
    result['verdict'] = ('PASS' if all(r['block']['rc'] == 0 and r['block']['reached_model']
                                       and r['block']['matches_expected'] for r in rows)
                         else 'FAIL')
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\n{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
