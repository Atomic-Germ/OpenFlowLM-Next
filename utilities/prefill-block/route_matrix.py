"""Phase 27 closure: assert the prefill route at exact ENGINE-VISIBLE lengths.

The previous matrix labelled prompts by the text they were built from. The chat template
adds a constant, so a fixture "255" reached `Engine::prefill` as 266 -- it never touched the
256-token boundary at all, and it cannot be used as boundary evidence.

So every length here is the number of token ids `Engine::prefill` receives, and the harness
*proves* it rather than assuming it:

  * the tokenizer count of the text is `n - template_offset`;
  * the runner's own `Prefill chunk ... with N tokens` must equal `n`;
  * `OFLM_OPEN_PREFILL_ROUTE` prints the route the engine actually chose.

The route is read from that line, never inferred from elapsed time -- a short sequential
prompt and a padded block overlap in wall time, so a timing cannot tell them apart.

Two routes exist and they are selected by capability, not by model name:

  layer-major     `Core::layer_major_ok()` -- every layer's gemm_block.kind is linear/full.
                  The prompt is not cut into blocks. Crossover 64 (Qwen3.6-35B).
  block loop      otherwise. Crossover 96, measured in the Phase 27 closure sweep on a fine
                  grid: the flat 256-wide block costs 33.47 s against 425.5 ms/token +
                  156 ms sequential, a crossing of ~78 engine tokens. 27.4's 128 came from
                  a six-point grid and overcharged 80..127 tokens.
                  `OFLM_OPEN_LAYER_MAJOR=0` forces the block loop even when layer-major is
                  available, which is how its policy is tested on this model.

The crossover counts tokens AS `Engine::prefill` receives them. That is not the number of
text tokens: the chat template's boundary merges with the first text token, and offsets of
8, 10 and 11 have all been measured, so nothing here is `text + K`.

    python route_matrix.py --mode block   # block loop, the 128 policy
    python route_matrix.py --mode default # what the shipped default actually does
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

# Measured once: 255 text tokens prefilled as 266. Every length below is adjusted by this,
# so `--lengths 255` means 255 tokens INTO Engine::prefill, not 255 tokens of text.
TEMPLATE_OFFSET = 0  # filled in by calibrate()

_ROUTE = re.compile(
    r'prefill-route (block|layer-major|sequential) n=(\d+) blocks=(\d+) tail=(\d+)'
    r'(?: padded=(\d+) passes=(\d+))?(?: why=(\S+) crossover=(\d+))?')
_TAILSEQ = re.compile(r'prefill-route tail-sequential n=(\d+) tokens=(\d+)')


def calibrate(tok: 'D.Tokenizer') -> int:
    """Template offset = (what the engine prefilled) - (what the tokenizer counted)."""
    text = tok.exact_text(32)
    rec = D.run([text], block=True, gen_lim=1, tag='cal')
    got = D.prefilled_tokens(rec)
    if got < 32:
        raise SystemExit(f'calibration failed: the engine prefilled {got} for 32 text tokens')
    return got - 32


def route_of(text: str) -> dict:
    """The engine's own account of the route, plus the tail it stepped."""
    m = _ROUTE.search(text)
    if not m:
        return dict(route='none', reason='no prefill-route line: re-run with '
                                         'OFLM_OPEN_PREFILL_ROUTE=1')
    out = dict(route=m.group(1), n=int(m.group(2)), blocks=int(m.group(3)),
               tail=int(m.group(4)))
    if m.group(5) is not None:
        out.update(padded=int(m.group(5)), passes=int(m.group(6)))
    if m.group(7) is not None:
        out.update(why=m.group(7), crossover=int(m.group(8)))
    t = _TAILSEQ.search(text)
    if t:
        out['tail_sequential'] = int(t.group(2))
    return out


def expected(n: int, crossover: int, physical: int = 256) -> dict:
    """The policy, written once, so the matrix is checked against a rule not a memory.

    A tail is padded into a block only when at least `crossover` real tokens remain;
    below that it is stepped. That is why 127 is sequential and 129 is one padded block,
    and why a 257-token prompt is one block plus one stepped token rather than two blocks.
    """
    if n < crossover:
        # Only the fields the sequential line actually prints: it has no `passes` (it
        # took no block pass) and no `padded`. Asking for a field the line never reports
        # compares None against 0 and fails a case that behaved correctly.
        return dict(route='sequential', blocks=0)
    full, tail = divmod(n, physical)
    padded = 1 if tail >= crossover else 0
    return dict(route='block', blocks=full, tail=tail, padded=padded,
                passes=full + padded,
                tail_sequential=tail if not padded else 0)


def one(n: int, *, mode: str, gen_lim: int, ctx: int, offset: int) -> dict:
    tok = D.Tokenizer()
    if n - offset < 1:
        raise SystemExit(f'{n} engine tokens is below the template offset {offset}')
    text = tok.exact_text(n - offset)
    counted = tok.count(text)
    extra = {'OFLM_OPEN_PREFILL_ROUTE': '1'}
    if mode == 'block':
        extra['OFLM_OPEN_LAYER_MAJOR'] = '0'
    rec = D.run([text], block=True, gen_lim=gen_lim, ctx=ctx, tag=f'e{n}', extra=extra)
    got = D.prefilled_tokens(rec)
    r = route_of(rec['text'])
    return dict(requested=n, template_offset=offset, text_tokens=counted,
                engine_tokens=got, token_match=(got == n), route=r, rc=rec['rc'],
                wall_s=rec['wall_s'], chunks=rec['chunks'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['block', 'default'], default='block')
    ap.add_argument('--lengths', type=int, nargs='+',
                    default=[63, 64, 65, 95, 96, 97, 127, 128, 129, 255, 256, 257,
                             383, 384, 385, 511, 512, 513])
    ap.add_argument('--crossover', type=int, default=96,
                    help='the block-loop constant the block mode is judged against')
    ap.add_argument('--gen-lim', type=int, default=2)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    tok = D.Tokenizer()
    offset = calibrate(tok)
    print(f'template offset = {offset} tokens (measured, not assumed)', flush=True)

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'route-matrix-{a.mode}-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-route-matrix', version=2, phase=27,
                  gate='27 closure: route asserted at exact Engine-visible lengths',
                  model_acceptance=False, mode=a.mode, binary=str(D.OFLM),
                  kernels=str(D.KERNELS), template_offset=offset, started=stamp,
                  route_reported_by='OFLM_OPEN_PREFILL_ROUTE (not inferred from time)')

    rows = []
    for n in a.lengths:
        r = one(n, mode=a.mode, gen_lim=a.gen_lim, ctx=a.ctx, offset=offset)
        if a.mode == 'block':
            want = expected(n, a.crossover)
        else:
            # What `--mode block` forces, what the shipped default does. Since Phase 28 the
            # default for a kernel set with no routed experts is the BLOCK loop, not
            # layer-major: `layer_major_preferred()` is false here, so production takes the
            # same policy the block mode asserts. This used to expect layer-major, and that
            # expectation is exactly what the phase changed.
            want = expected(n, a.crossover)
        got = r['route']
        ok = r['token_match'] and r['rc'] == 0 and all(
            got.get(k) == v for k, v in want.items() if k != 'tail_sequential')
        if want.get('tail_sequential') is not None and a.mode == 'block':
            ok = ok and got.get('tail_sequential', 0) == want['tail_sequential']
        row = dict(requested=n, expected=want, actual=got, token_match=r['token_match'],
                   engine_tokens=r['engine_tokens'], wall_s=r['wall_s'],
                   verdict='PASS' if ok else 'FAIL')
        rows.append(row)
        desc = (f"n={n:4d} eng={r['engine_tokens']:4d} route={got.get('route'):11s} "
                f"blocks={got.get('blocks')} tail={got.get('tail')} "
                f"padded={got.get('padded')} passes={got.get('passes')} "
                f"tailseq={got.get('tail_sequential')}")
        print(f"  {desc}  {'ok' if r['token_match'] else 'TOKENS MISMATCH'}  {row['verdict']}",
              flush=True)

    bad = [r['requested'] for r in rows if r['verdict'] != 'PASS']
    result['rows'] = rows
    result['failed'] = bad
    result['verdict'] = 'PASS' if not bad else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\n{result['verdict']}{'' if not bad else ' failed ' + str(bad)}"
          f" -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
