"""Phase 27 closure, item 6: exercise the 27.11 failure handler on hardware.

The handler in [`engine.cpp`](../../src/open_qwen36/engine.cpp) claims that a block failure
on a set that ADVERTISES the block route must either (a) put the caches back and finish the
prompt sequentially, or (b) throw rather than continue on a half-applied cache. Reading
that in the source proves nothing, so `OFLM_OPEN_BLOCK_FAIL_INJECT` breaks a block on
purpose at the three points that matter:

  before         the FIRST block throws before it writes anything. Nothing has moved, so
                 the reset state already IS the pre-prompt state: case A.
  after          a second prompt -- one that starts at position != 0 -- has its SECOND block
                 throw after the first one already ran. This is the case the plan names,
                 and the only one that reaches the snapshot/rewind branch: at position 0 no
                 snapshot is taken because reset() is free.
  unrecoverable  the same failure with the rewind deliberately broken. The request must
                 FAIL, and must not answer from a half-applied cache: case C.

Armed with `OFLM_OPEN_BLOCK_FAIL_AT=2` where the second prompt of the session is the target:
a fresh prompt can only ever exercise the reset branch, so without it the snapshot branch
is unreachable and "after" would silently test case A twice.

Each recovering case is compared against the same prompt run forced-sequential. After the
fallback the caches must be indistinguishable in their effect, so the greedy continuation
must be identical -- text equality, not cosine.

Both injector knobs are diagnostics listed in `NON_SETTING_ENV_VARS`: nothing reads them
unless a test sets them, and they must never become settings.

    python fail_inject.py                      # cases A, B and C
    python fail_inject.py --text-tokens 500    # a longer prompt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oflm_drive as D  # noqa: E402
import crossover_local as C  # noqa: E402  the route/policy parser lives there

# A second block exists only while at least `min_prompt` (96) engine tokens remain, so a
# prompt needs GT + 96 = 352 engine tokens before the loop runs twice. 300 text tokens is
# 256 plus a tail the crossover deliberately steps, which is why the first version of this
# harness saw "after" inject nothing at all.
MIN_ENGINE_FOR_TWO_BLOCKS = 352
DEFAULT_TEXT = 400


def one(session: list[str], *, inject: str | None, at: int | None, gen_lim: int,
        ctx: int, timeout: float, tag: str) -> dict:
    extra = {'OFLM_OPEN_PREFILL_ROUTE': '1', 'OFLM_OPEN_LAYER_MAJOR': '0'}
    if inject:
        extra['OFLM_OPEN_BLOCK_FAIL_INJECT'] = inject
    if at:
        extra['OFLM_OPEN_BLOCK_FAIL_AT'] = str(at)
    rec = D.run(session, block=True, gen_lim=gen_lim, ctx=ctx, timeout=timeout,
                tag=tag, extra=extra)
    p = C.parse(rec['text'])
    lines = [l for l in rec['stderr'].splitlines()
             if 'prefill' in l or 'restore' in l or 'rewound' in l or 'inject' in l
             or 'request failed' in l or 'schedule' in l]
    return dict(inject=inject, at=at, rc=rec['rc'], wall_s=rec['wall_s'],
                engine_tokens=[r['n'] for r in p['routes']],
                passes=[r.get('passes') for r in p['routes']],
                total_engine_tokens=sum(p['chunks']),
                output=rec['output'], turns=rec['turns'], said=lines,
                injected='injected block failure' in rec['text'],
                said_reset='cache reset' in rec['text'],
                said_restored='pre-prompt state restored' in rec['text'],
                said_could_not_restore='could not be restored' in rec['text'],
                said_request_failed='request failed' in rec['text'],
                insertion_error='Insertion error' in rec['text'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text-tokens', type=int, default=DEFAULT_TEXT)
    ap.add_argument('--gen-lim', type=int, default=8)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    tok = D.Tokenizer()
    text = tok.exact_text(a.text_tokens)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'fail-inject-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-fail-inject', version=2, phase=27,
                  gate='27 closure item 6: the 27.11 handler on hardware',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, text_tokens=a.text_tokens, gen_lim=a.gen_lim, ctx=a.ctx,
                  min_engine_for_two_blocks=MIN_ENGINE_FOR_TWO_BLOCKS,
                  note='OFLM_OPEN_BLOCK_FAIL_INJECT / _AT are diagnostics in '
                       'NON_SETTING_ENV_VARS, not settings')

    print(f'=== prompt of {a.text_tokens} text tokens, gen_lim {a.gen_lim} ===', flush=True)
    ref = one([text], inject=None, at=None, gen_lim=a.gen_lim, ctx=a.ctx,
              timeout=a.timeout, tag='ref')
    blk = one([text], inject=None, at=None, gen_lim=a.gen_lim, ctx=a.ctx,
              timeout=a.timeout, tag='clean')
    passes = max(x or 0 for x in blk['passes'] or [0])
    print(f"  reference: eng={ref['total_engine_tokens']} passes={passes} "
          f"{ref['output'][:34]!r}", flush=True)
    result['reference'] = ref
    result['clean_block'] = blk
    rows = [dict(check='clean_block_matches_sequential',
                 ok=blk['output'] == ref['output'] and bool(ref['output']),
                 detail=dict(sequential=ref['output'], block=blk['output'])),
            # Without two real passes, "after" and "unrecoverable" inject nothing and
            # every check below would pass vacuously.
            dict(check='prompt_reaches_two_block_passes',
                 ok=passes >= 2 and blk['total_engine_tokens'] >= MIN_ENGINE_FOR_TWO_BLOCKS,
                 detail=dict(engine=blk['total_engine_tokens'], passes=passes))]

    cases: dict[str, dict] = {}
    # Case A: fresh prompt, first block. Position 0, so the reset is the honest undo.
    a_case = one([text], inject='before', at=None, gen_lim=a.gen_lim, ctx=a.ctx,
                 timeout=a.timeout, tag='A')
    cases['A_before_first_block'] = a_case
    print(f"  A before       rc={a_case['rc']} injected={a_case['injected']} "
          f"reset={a_case['said_reset']} out={a_case['output'][:26]!r}", flush=True)
    rows += [
        dict(check='A_falls_back_to_the_sequential_answer',
             ok=a_case['injected'] and a_case['output'] == ref['output'],
             detail=dict(got=a_case['output'], want=ref['output'])),
        dict(check='A_reports_the_cache_was_reset', ok=a_case['said_reset'],
             detail=a_case['said'])]

    # Case B needs a prefill that starts at position != 0, because that is the only way a
    # snapshot exists to rewind to. Two things make that possible:
    #   * turn 2 EXTENDS turn 1 instead of repeating it -- `_shared_insert` clears the
    #     context when the token history does not prefix-match, which puts the position
    #     back to 0 and silently reduces this case to case A;
    #   * the extension is long enough to be its own two-block prompt (>= 352 engine
    #     tokens), so the broken second block really is the second block of a run.
    ext = 400
    long_text = tok.exact_text(a.text_tokens + ext)
    assert long_text.startswith(text), 'the extension must preserve turn 1 as a prefix'
    print(f'  (turn 2 extends turn 1 by {ext} text tokens so the position is not reset)',
          flush=True)
    b_case = one([text, long_text], inject='after', at=2, gen_lim=a.gen_lim, ctx=a.ctx,
                 timeout=a.timeout, tag='B')
    cases['B_after_first_block'] = b_case
    b_turn2 = (b_case['turns'] or [''])[-1]
    print(f"  B after        rc={b_case['rc']} injected={b_case['injected']} "
          f"restored={b_case['said_restored']} reset={b_case['said_reset']}", flush=True)
    rows += [
        dict(check='B_the_injector_fired_on_a_non_zero_position',
             # If this reports `cache reset`, the second prompt was cleared back to zero
             # and the snapshot branch was never reached, so the case proves nothing.
             ok=b_case['injected'] and not b_case['said_reset'],
             detail=dict(routes=b_case['engine_tokens'], said=b_case['said'])),
        dict(check='B_reports_the_pre_prompt_state_was_restored',
             ok=b_case['said_restored'], detail=b_case['said']),
        dict(check='B_still_answers_the_second_prompt',
             ok=bool(b_turn2), detail=dict(turn2=b_turn2, turns=b_case['turns'])),
    ]

    # Case C: the same shape with the rewind broken. The request must be refused.
    c_case = one([text, long_text], inject='unrecoverable', at=2, gen_lim=a.gen_lim,
                 ctx=a.ctx, timeout=a.timeout, tag='C')
    cases['C_unrecoverable'] = c_case
    c_turn2 = (c_case['turns'] or [''])[-1]
    print(f"  C unrecover.   rc={c_case['rc']} could_not_restore="
          f"{c_case['said_could_not_restore']} turn2={c_turn2[:24]!r}", flush=True)
    # `oflm run` catches the exception at the REPL, prints `Insertion error: ...` and keeps
    # going, so the process still exits 0. A refusal is therefore judged on the REQUEST:
    # the engine said it could not restore, the error surfaced, and the failing turn
    # generated NOTHING. Grading it on the exit status would be backwards -- rc 0 is the
    # expected outcome of a refused request, and the whole session still has turn 1's text.
    refused = (c_case['injected'] and c_case['said_could_not_restore']
               and not c_case['said_reset']
               and (c_case['insertion_error'] or c_case['said_request_failed'])
               and not c_turn2)
    rows.append(dict(check='C_refuses_and_generates_nothing', ok=bool(refused),
                     detail=dict(rc=c_case['rc'], turn2=c_turn2,
                                 insertion_error=c_case['insertion_error'],
                                 said=c_case['said'])))

    result['cases'] = cases
    result['rows'] = rows
    result['verdict'] = 'PASS' if all(r['ok'] for r in rows) else 'FAIL'
    for r in rows:
        print(f"  {r['check']:52s} {'PASS' if r['ok'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
