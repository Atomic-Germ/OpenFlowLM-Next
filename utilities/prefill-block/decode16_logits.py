"""Phase 27 closure item 8: the 16-token trajectory WITH the numbers the plan asks for.

`oflm` reports no logits, so it can only say whether two token sequences happen to match.
The plan asks for min cosine, max_abs and max_rel across a 16-token trajectory, which needs
the logits -- so this drives the standalone harness CLI (the same `Engine::prefill`,
`--max-tokens 16 --dump-logits`), where Phase 26's oracle already knows how to read them.

The prompt is long enough for several 256-wide blocks, and the block side forces the block
loop with `OFLM_OPEN_LAYER_MAJOR=0` so the decode steps inherit a cache written by more than
one pass.

No tolerance is introduced. The gate is the repository's own cosine (0.9999, the one
`oracle.py` uses everywhere), and the token comparison is reported next to it rather than
used as the gate -- argmax over two logit vectors that differ in the 5th decimal can flip,
so demanding identical tokens would be a stricter rule than any the kernels guarantee, and
inventing it here would be exactly the "new tolerance" the plan forbids.

    python decode16_logits.py                  # 700 ids, 16 decode steps, 64 layers
    python decode16_logits.py --decode 16 --layers 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle as O  # noqa: E402
import decode_transition as T  # noqa: E402  the one-session runner and dump naming


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prompt', type=int, default=700,
                    help='prompt tokens; >= 352 is two blocks, 700 is two full plus a tail')
    ap.add_argument('--chunk', type=int, default=256,
                    help='must equal the physical block so the tail is the real tail')
    ap.add_argument('--decode', type=int, default=16)
    ap.add_argument('--layers', type=int, default=64)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if a.chunk != 256:
        raise SystemExit('--chunk must be the physical block (256): any other value would '
                         'measure a schedule the shipped runtime never runs')

    ids = O.fixture(a.prompt, seed=26026)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'decode16-logits-{a.prompt}-{a.decode}-{stamp}')
    if out.exists():
        raise SystemExit(f'{out} exists; a run tree is never reused')
    out.mkdir(parents=True)
    seqdir, blkdir = out / 'seq', out / 'block'

    pa = T.run(ids, seqdir, 'seq', chunk=None, layers=a.layers, decode=a.decode,
               position=0, dump_state=[], outdir=str(out))
    pb = T.run(ids, blkdir, 'block', chunk=a.chunk, layers=a.layers, decode=a.decode,
               position=0, dump_state=[], outdir=str(out))

    rows: list[dict] = []
    # The prompt's own last row first: that is the token the prefill hands to the sampler.
    last = a.prompt - 1
    la, lb = O.read_dump(pa['prefix'], last), O.read_dump(pb['prefix'], last)
    if la is None or lb is None:
        rows.append(dict(tensor='logits@prompt_last', verdict='FAIL',
                         reason='prompt dump missing'))
    else:
        rows.append(O.metric('logits@prompt_last', lb, la))
    token_rows = []
    for k in range(a.decode):
        fa, fb = Path(f"{pa['prefix']}_t{k}.bin"), Path(f"{pb['prefix']}_t{k}.bin")
        if not (fa.is_file() and fb.is_file()):
            rows.append(dict(tensor=f'logits@decode{k}', verdict='FAIL',
                             reason='decode dump missing'))
            continue
        x = np.fromfile(fa, dtype=np.float32)
        y = np.fromfile(fb, dtype=np.float32)
        rows.append(O.metric(f'logits@decode{k}', y, x))
        ga, gb = int(np.argmax(x[:O.REAL_VOCAB])), int(np.argmax(y[:O.REAL_VOCAB]))
        token_rows.append(dict(step=k, sequential=ga, block=gb, agree=ga == gb))

    graded = [r for r in rows if 'cosine' in r]
    cosines = [r['cosine'] for r in graded]
    summary = dict(compared=len(graded),
                   min_cosine=min(cosines) if cosines else None,
                   max_abs=max((r['max_abs'] for r in graded), default=None),
                   max_rel=max((r['max_rel'] for r in graded), default=None),
                   last_cosine=cosines[0] if cosines else None,
                   tokens_agreeing=sum(1 for t in token_rows if t['agree']),
                   tokens_compared=len(token_rows),
                   first_disagreement=next((t['step'] for t in token_rows if not t['agree']),
                                           None))
    result = dict(tool='prefill-block-decode16-logits', version=1, phase=27,
                  gate='27 closure item 8: 16-token trajectory, cosine/max_abs/max_rel',
                  model_acceptance=False, cli=str(O.CLI), kernels=str(O.KERNELS),
                  started=stamp, prompt_tokens=a.prompt, chunk=a.chunk,
                  decode_steps=a.decode, layers=a.layers,
                  tolerance='the repository cosine gate from oracle.py; nothing new',
                  sequential=pa, block=pb, tensors=rows, tokens=token_rows,
                  summary=summary)
    result['verdict'] = ('PASS' if graded and all(r['verdict'] == 'PASS' for r in rows)
                         else 'FAIL')
    print(f'\n=== 16-token trajectory, {a.prompt} prompt ids, {a.layers} layers ===')
    for r in rows:
        if 'cosine' in r:
            print(f"  {r['tensor']:22s} cos {r['cosine']:.9f}  max_abs {r['max_abs']:.3e}"
                  f"  max_rel {r['max_rel']:.3e}  {r['verdict']}")
        else:
            print(f"  {r['tensor']:22s} {r['verdict']} {r.get('reason', '')}")
    print(f"  tokens agreeing  {summary['tokens_agreeing']}/{summary['tokens_compared']}"
          f" (first disagreement at step {summary['first_disagreement']})")
    print(f"  min cosine {summary['min_cosine']}  max_abs {summary['max_abs']:.3e}"
          f"  max_rel {summary['max_rel']:.3e}")
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
