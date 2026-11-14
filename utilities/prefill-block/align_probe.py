"""Phase 28.5: does layer-major have any intrinsic edge at an EXACT multiple of 256?

The A/B shows layer-major losing by 1.80x at 264 engine tokens, 1.42x at 520, 1.21x at 1032
and 1.11x at 2056. That decay is not noise and it is not "layer-major gets better": both
schedules cost about the same per 256-wide block, but `Core::step_gemm_prompt` computes
`TOT = B * T` and runs EVERY block at full width, padding the prompt up to a multiple of the
block, while `Engine::prefill`'s block loop steps the tail token by token. So layer-major
does ceil(N/256) blocks of work against the block loop's floor((N-96)/256)+1 and the gap
shrinks as the tail becomes a smaller fraction of the prompt.

That leaves one question the decaying curve cannot answer: at a prompt that IS a multiple of
256 there is no tail to waste, so the two schedules do the SAME number of full-width GEMM
dispatches. layer-major additionally overlaps the next block's host staging behind the
current dispatch (`stage_major`) and keeps each layer's weights resident for the whole
prompt. If that is worth anything, this is the only place it can show.

So this probe measures both routes at engine counts that are exact multiples of 256.

## Getting an exact engine count

The chat template's offset is not a constant (8, 10 and 11 have all been observed), so the
text length that yields exactly 256 engine tokens has to be found rather than computed. The
first phase here sweeps candidate text lengths in ONE process and reads the engine's own
count for each; the second phase then measures both routes at the lengths that landed
exactly.

    python align_probe.py --targets 256 512
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
import route_ab as R  # noqa: E402


def find_texts(targets: list[int], *, spread: int, timeout: float,
               offset_hint: int) -> dict:
    """One process: candidate text lengths, and the engine count each one produced."""
    tok = D.Tokenizer()
    # The text that lands on `target` engine tokens is `target - offset`, and the offset is
    # what this phase has to discover -- so the sweep goes around `target - offset`, not
    # around `target`. Sweeping around `target` looks for engine==target at text==target,
    # which for an offset of 8 is 8 tokens too long at every candidate and finds nothing.
    offset = offset_hint
    candidates: list[tuple[int, int]] = []      # (target, text tokens)
    for t in targets:
        base = t - offset
        for d in range(-spread, spread + 1):
            if base + d > 0:
                candidates.append((t, base + d))
    session: list[str] = []
    for _, n_text in candidates:
        session += ['/clear', tok.exact_text(n_text)]
    # Count with the BLOCK route, not the sequential one. The count this needs is the
    # token ids `Engine::prefill` was given, and both routes report it identically -- but a
    # sequential count at 512-plus tokens is minutes per candidate, and 26 candidates of it
    # ran for well over an hour before this was fixed. The block route answers the same
    # question in a padded dispatch.
    rec = D.run(session, block=True, gen_lim=1, ctx=4096, timeout=timeout,
                tag='align', extra={'OFLM_OPEN_LAYER_MAJOR': '0'})
    if rec['rc'] != 0:
        raise SystemExit(f'alignment sweep failed (rc {rec["rc"]})')
    counts = [c['tokens'] for c in rec['chunks']]
    if len(counts) != len(candidates):
        raise SystemExit(f'expected {len(candidates)} prompts, saw {len(counts)}')
    found: dict[int, int] = {}
    table = []
    for (target, n_text), engine in zip(candidates, counts):
        table.append(dict(target=target, text_tokens=n_text, engine_tokens=engine))
        if engine == target and target not in found:
            found[target] = n_text
    return dict(found=found, table=table,
                offset_by_text={r['text_tokens']: r['engine_tokens'] - r['text_tokens']
                                for r in table})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', type=int, nargs='+', default=[256, 512])
    ap.add_argument('--spread', type=int, default=6,
                    help='how far either side of (target - offset) to sweep text lengths')
    ap.add_argument('--offset-hint', type=int, default=8,
                    help='where to centre the sweep: the template offset measured for this '
                         'fixture family (8 for the exact_text filler)')
    ap.add_argument('--gen-lim', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'align-probe-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-align-probe', version=1, phase=28,
                  gate='28.5 exact-multiple control: is layer-major ever faster?',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, targets=a.targets, spread=a.spread,
                  why='at an exact multiple of 256 there is no tail to pad, so the two '
                      'schedules run the same number of full-width GEMM dispatches; any '
                      'difference there is schedule efficiency rather than padding')

    print(f'=== phase 1: find a text length per target for an EXACT engine count ===',
          flush=True)
    align = find_texts(a.targets, spread=a.spread, timeout=a.timeout,
                       offset_hint=a.offset_hint)
    result['alignment'] = align
    for target in a.targets:
        got = align['found'].get(target)
        print(f"  target {target:5d}: text tokens {got if got is not None else 'NOT FOUND'}",
              flush=True)
    missing = [t for t in a.targets if t not in align['found']]
    if missing:
        result['verdict'] = 'FAIL'
        result['reason'] = f'no text length produced engine tokens {missing} within ' \
                           f"text +/-{a.spread}; widen --spread"
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
        print(f"FAIL: {result['reason']}")
        return 1

    print('=== phase 2: both routes at each exact multiple ===', flush=True)
    tok = D.Tokenizer()
    rows = []
    # Alternate the route order per target, same discipline as the main A/B.
    for i, target in enumerate(a.targets):
        text = tok.exact_text(align['found'][target])
        order = ('layer-major', 'block') if i % 2 == 0 else ('block', 'layer-major')
        for route in order:
            r = R.measure(text, route=route, gen_lim=a.gen_lim, ctx=4096,
                          timeout=a.timeout, warm=True)
            r['target'] = target
            rows.append(r)
            print(f"  {route:11s} target={target:5d} eng={r['engine_tokens']:5d} "
                  f"passes={r['passes']} prefill={r['prefill_ms']}ms", flush=True)
    result['rows'] = rows

    pairs = []
    for target in a.targets:
        lm = next(r for r in rows if r['target'] == target and r['route'] == 'layer-major')
        bl = next(r for r in rows if r['target'] == target and r['route'] == 'block')
        if not (lm['prefill_ms'] and bl['prefill_ms']):
            continue
        slow, fast = max(lm['prefill_ms'], bl['prefill_ms']), min(lm['prefill_ms'],
                                                                 bl['prefill_ms'])
        pairs.append(dict(engine_tokens=lm['engine_tokens'],
                          layer_major_ms=lm['prefill_ms'], block_ms=bl['prefill_ms'],
                          winner='layer-major' if lm['prefill_ms'] < bl['prefill_ms'] else 'block',
                          ratio=round(slow / fast, 3),
                          exact_multiple=(lm['engine_tokens'] % 256 == 0)))
    result['pairs'] = pairs
    result['verdict'] = 'PASS' if pairs else 'FAIL'
    print(f'\n{"engine":>7} {"layer-major":>12} {"block":>10} {"winner":>12} {"ratio":>7}')
    for p in pairs:
        print(f"{p['engine_tokens']:7d} {p['layer_major_ms']:10.0f}ms {p['block_ms']:8.0f}ms "
              f"{p['winner']:>12} {p['ratio']:>7}")
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
