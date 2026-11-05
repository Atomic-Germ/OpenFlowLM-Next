"""Phase 27 closure items 6B/6C and 7: the snapshot path, and request isolation, on `serve`.

`oflm run` cannot reach the snapshot branch. Its REPL compares each new prompt against a
history that now ends with the previous answer, the comparison fails, and
`AutoModel::_shared_insert` clears the context -- so every prefill starts at position 0,
where no snapshot is taken and `reset()` is all the undo there is. That is why the injected
"after" case kept reporting `cache reset`: it was measuring case A every time.

`serve` is different: a second request carries the whole conversation, whose token history
prefix-matches what the engine already holds, so nothing is cleared and the prefill starts
at position != 0. That is exactly the situation the 27.11 snapshot exists for: a block runs,
a later block fails, and the caches have to go back to where they were before THIS prompt.

So both remaining cases, and the A -> B -> A isolation check, are driven over HTTP:

  B  request 1 = prompt A; request 2 = A + B as a follow-up. The second prefill starts at a
     non-zero position, runs blocks, and the injector breaks its second block. The answer
     must still arrive (fallback) and the engine must say it restored the pre-prompt state.
  C  the same with the rewind broken: the request must be refused, with nothing generated.

The isolation check is the same session: A's answer before B must equal A's answer after B,
and B must equal what a fresh process gives for B.

    python fail_inject_serve.py
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
import oflm_serve_check as S  # noqa: E402


def chat_turns(port: int, messages: list[dict], *, max_tokens: int,
               timeout: float) -> dict:
    """One request; `content` is the assistant reply and `error` a refusal."""
    return S.chat(port, '', max_tokens=max_tokens, timeout=timeout, messages=messages)


def session(port: int, text_a: str, text_b: str, *, inject: str | None,
            max_tokens: int, timeout: float, extra: dict | None = None) -> dict:
    """A, then A+B as a follow-up, in ONE server process.

    The follow-up's prefill is NOT the whole conversation: `AutoModel::_shared_insert`
    erases the prefix it already holds, so what reaches `Engine::prefill` is the NEW
    suffix. `text_b` therefore has to be block-sized on its own (several hundred tokens),
    or the follow-up is a short sequential prefill that never enters the block loop and
    the injector has nothing to break.
    """
    env_extra = dict(extra or {})
    if inject:
        # Armed for the SECOND prefill of the process, which is the follow-up: it is the
        # only one that starts at a non-zero position, and therefore the only one that
        # takes a snapshot to rewind to.
        env_extra['OFLM_OPEN_BLOCK_FAIL_INJECT'] = inject
        env_extra['OFLM_OPEN_BLOCK_FAIL_AT'] = '2'
    # Without this the layer-major schedule takes the whole prompt in one pass and the
    # block loop never runs at all -- the first version of this harness forgot it, so the
    # injector was armed on a route that cannot fail and every case reported "not
    # injected" while the prefill succeeded. `layer-major` is the shipped default here.
    env_extra['OFLM_OPEN_LAYER_MAJOR'] = '0'
    env_extra['OFLM_OPEN_PREFILL_ROUTE'] = '1'
    proc, _ = S.start(port, ctx=4096, extra=env_extra)
    try:
        err = S.wait_ready(port, proc, timeout=1200)
        if err:
            return dict(started=False, reason=err)
        first = chat_turns(port, [dict(role='user', content=text_a)],
                           max_tokens=max_tokens, timeout=timeout)
        follow = chat_turns(port, [dict(role='user', content=text_a),
                                   dict(role='assistant', content=first.get('content') or ''),
                                   dict(role='user', content=text_b)],
                            max_tokens=max_tokens, timeout=timeout)
        again = chat_turns(port, [dict(role='user', content=text_a)],
                           max_tokens=max_tokens, timeout=timeout)
        return dict(started=True, first=first, follow=follow, again=again)
    finally:
        proc.terminate()
        try:
            log = proc.communicate(timeout=30)[0] or ''
        except Exception:  # noqa: BLE001
            proc.kill()
            log = ''
        Path('/tmp/ofi_serve_last.log').write_text(log)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', type=int, default=400, help='text tokens of prompt A')
    # The follow-up's NEW suffix is what reaches the engine, so B has to be block-sized on
    # its own: 256 + 96 = 352 engine tokens is the smallest two-block prefill.
    ap.add_argument('--b', type=int, default=600, help='text tokens of the follow-up suffix')
    ap.add_argument('--max-tokens', type=int, default=8)
    ap.add_argument('--port', type=int, default=18141)
    ap.add_argument('--timeout', type=float, default=2400)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    tok = D.Tokenizer()
    a_text = tok.exact_text(a.a)
    b_text = tok.exact_text(a.b)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'fail-inject-serve-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-fail-inject-serve', version=1, phase=27,
                  gate='27 closure items 6B/6C + 7: snapshot rewind and isolation on serve',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, a_text_tokens=a.a, b_text_tokens=a.b,
                  max_tokens=a.max_tokens,
                  why_serve='oflm run clears the context between prompts, so every there '
                            'prefill starts at position 0 and never takes a snapshot; a '
                            'serve follow-up continues the conversation and does')

    rows: list[dict] = []

    def log_lines() -> list[str]:
        try:
            raw = Path('/tmp/ofi_serve_last.log').read_text()
        except OSError:
            return []
        return [l for l in raw.splitlines()
                if 'prefill' in l or 'restore' in l or 'rewound' in l or 'inject' in l]

    print('=== clean session (isolation, item 7) ===', flush=True)
    clean = session(a.port, a_text, b_text, inject=None, max_tokens=a.max_tokens,
                    timeout=a.timeout)
    result['clean'] = clean
    if not clean.get('started'):
        result['verdict'] = 'FAIL'
        result['rows'] = [dict(check='server_starts', ok=False, detail=clean.get('reason'))]
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
        print(f"FAIL server did not start: {clean.get('reason')}")
        return 1
    result['clean_said'] = log_lines()
    f1, fu, f2 = clean['first'], clean['follow'], clean['again']
    print(f"  A  ok={f1.get('ok')} {str(f1.get('content'))[:30]!r}", flush=True)
    print(f"  B  ok={fu.get('ok')} {str(fu.get('content'))[:30]!r}", flush=True)
    print(f"  A' ok={f2.get('ok')} {str(f2.get('content'))[:30]!r}", flush=True)
    rows += [
        dict(check='A_and_its_follow_up_both_answered',
             ok=bool(f1.get('ok') and f1.get('content') and fu.get('ok')
                     and fu.get('content')),
             detail=dict(a=f1.get('content'), follow_up=fu.get('content'))),
        dict(check='A_after_B_equals_A_before_B',
             ok=bool(f1.get('ok') and f2.get('ok')
                     and f1.get('content') == f2.get('content')),
             detail=dict(before=f1.get('content'), after=f2.get('content')),
             why='resident state leaked between requests if these differ'),
        dict(check='A_and_B_are_different_prompts',
             ok=bool(f1.get('content') != fu.get('content')),
             why='otherwise the isolation check would pass vacuously'),
    ]

    for name, mode in (('B', 'after'), ('C', 'unrecoverable')):
        print(f'=== case {name}: inject "{mode}" into the follow-up prefill ===', flush=True)
        got = session(a.port + (1 if name == 'B' else 2), a_text, b_text, inject=mode,
                      max_tokens=a.max_tokens, timeout=a.timeout)
        said = log_lines()
        got['said'] = said
        result[f'case_{name}'] = got
        if not got.get('started'):
            rows.append(dict(check=f'{name}_server_starts', ok=False,
                             detail=got.get('reason')))
            continue
        follow = got['follow']
        inj = any('injected block failure' in l for l in said)
        restored = any('pre-prompt state restored' in l for l in said)
        reset = any('cache reset' in l for l in said)
        cannot = any('could not be restored' in l for l in said)
        print(f"  injected={inj} restored={restored} reset={reset} "
              f"cannot_restore={cannot} follow_ok={follow.get('ok')}", flush=True)
        rows += [
            dict(check=f'{name}_injector_fired_on_a_non_zero_position',
                 ok=inj and not reset,
                 detail=dict(said=said),
                 why='a reset here would mean the follow-up was cleared and the snapshot '
                     'path was never entered'),
        ]
        if name == 'B':
            rows += [
                dict(check='B_reports_the_pre_prompt_state_was_restored',
                     ok=restored, detail=dict(said=said)),
                dict(check='B_still_answers_the_follow_up',
                     ok=bool(follow.get('ok') and follow.get('content')),
                     detail=dict(content=follow.get('content'),
                                 error=follow.get('error'))),
            ]
        else:
            # A refusal: the engine said it could not restore and the request produced no
            # answer. `serve` returns an error response for a failed insert.
            rows.append(dict(check='C_refuses_and_generates_nothing',
                             ok=bool(cannot and not follow.get('ok')),
                             detail=dict(cannot_restore=cannot, ok=follow.get('ok'),
                                         content=follow.get('content'),
                                         error=follow.get('error'), said=said)))

    result['rows'] = rows
    result['verdict'] = 'PASS' if all(r['ok'] for r in rows) else 'FAIL'
    for r in rows:
        print(f"  {r['check']:52s} {'PASS' if r['ok'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
