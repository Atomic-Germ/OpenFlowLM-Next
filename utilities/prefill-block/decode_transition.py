"""Phase 26.12: the multi-chunk -> decode transition. A HARD acceptance gate.

The claim under test is the one Phase 27 will depend on: the generated-token path must
not know or care whether the prompt arrived sequentially or via block prefill.

    sequential:  all P prompt tokens one at a time, then decode token 1, decode token 2
    block:       prompt as chunks (with a tail),        then decode token 1, decode token 2

Both paths run in ONE process each, back to back, so the second decode token is reached
through a real decode step that read the state the prefill left -- not through a
reconstructed one. That is the difference between testing the transition and testing a
replay of it.

What is compared after EACH decode token: greedy token, logits, the DeltaNet states, the
conv histories, the KV caches and the position.

    python decode_transition.py --prompt 40 --chunk 32 --decode 2
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle as O  # noqa: E402  the metrics, the fixture and the state layout live there


def run(ids: list[int], workdir: Path, tag: str, *, chunk: int | None, layers: int,
        decode: int, position: int, dump_state: list[str], outdir: str) -> dict:
    """One whole session: prefill (sequential or chunked) then `decode` real steps."""
    workdir.mkdir(parents=True, exist_ok=True)
    idfile = workdir / f'ids_{tag}.txt'
    idfile.write_text(','.join(str(i) for i in ids) + '\n')
    prefix = workdir / f'logits_{tag}'
    env = dict(os.environ)
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    env.pop('OFLM_OPEN_PREFILL_ROLES', None)
    env.pop('OFLM_OPEN_PREFILL_ROLE_MODE', None)
    cmd = [str(O.CLI), '--model', str(O.MODEL), '--kernels', str(O.KERNELS),
           '--ids-file', str(idfile), '--max-ctx', '4096',
           '--max-tokens', str(decode), '--layers', str(layers),
           '--pmode', 'none', '--dump-logits', str(prefix)]
    if chunk:
        cmd.append('--gemm-block')
    if position:
        cmd += ['--at-position', str(position)]
    for spec in dump_state:
        cmd += ['--dump-state', spec]
    t0 = time.monotonic()
    with O.LOCK.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, cwd=str(O.ROOT), env=env, capture_output=True, text=True)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    (workdir / f'out_{tag}.log').write_text(text)
    if proc.returncode != 0:
        raise SystemExit(f'{tag} failed (rc {proc.returncode}):\n' +
                         '\n'.join(text.strip().splitlines()[-12:]))
    # `^` needs re.M: without it the pattern only anchors at the start of the whole text
    # and every match is lost -- the same silent zero that cost Phase 23 a sweep.
    greedy = [int(x) for x in re.findall(r'^token (\d+)$', text, re.M)]
    return dict(prefix=str(prefix), workdir=str(workdir), wall_s=round(wall, 2),
                tokens=len(ids), greedy=greedy, command=' '.join(cmd))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prompt', type=int, default=40)
    ap.add_argument('--chunk', type=int, default=32,
                    help='logical chunk for the block path; the tail is --prompt %% --chunk')
    ap.add_argument('--decode', type=int, default=2)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--position', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if a.prompt < 1 or a.decode < 1:
        raise SystemExit('--prompt and --decode must be >= 1')

    ids = O.fixture(a.prompt, seed=26026)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' /
                    f'decode-n{a.prompt}-c{a.chunk}-d{a.decode}-l{a.layers}-{stamp}')
    if out.exists():
        raise SystemExit(f'{out} exists; a run tree is never reused')
    out.mkdir(parents=True)
    seqdir, blkdir = out / 'seq', out / 'block'

    # One slice of each kind, at the deepest layer that has it: the last linear and the
    # last full layer, so the comparison covers state that 64 layers of arithmetic have
    # already touched.
    kinds = O.layer_kinds()[:a.layers]
    lin = max((i for i, k in enumerate(kinds) if k == 'linear'), default=None)
    full = max((i for i, k in enumerate(kinds) if k == 'full'), default=None)
    specs: dict[int, list[tuple[str, int, int]]] = {}
    if lin is not None:
        specs[lin] = [('conv', 0, O.CONV_BYTES), ('S', O.S_OFF, O.S_BYTES)]
    if full is not None:
        # After the decode steps the cache holds prompt + decode rows.
        specs[full] = [('kv', 0, (a.position + a.prompt + a.decode) * O.KV_ROW_BYTES)]

    def flags(wd: Path) -> list[str]:
        return [f'{l}:{off}:{n}:{wd / f"state_{l}_{off}_{n}.bin"}'
                for l, ss in specs.items() for (_, off, n) in ss]

    pa = run(ids, seqdir, 'seq', chunk=None, layers=a.layers, decode=a.decode,
             position=a.position, dump_state=flags(seqdir), outdir=str(out))
    pb = run(ids, blkdir, 'block', chunk=a.chunk, layers=a.layers, decode=a.decode,
             position=a.position, dump_state=flags(blkdir), outdir=str(out))

    rows: list[dict] = []
    # The greedy tokens the process printed, in order: prompt argmax first, then one per
    # decode step. They must agree element by element, not merely at the end.
    if pa['greedy'] and pb['greedy']:
        rows.append(dict(tensor='greedy_tokens', sequential=pa['greedy'], block=pb['greedy'],
                         verdict='PASS' if pa['greedy'] == pb['greedy'] else 'FAIL'))
    else:
        rows.append(dict(tensor='greedy_tokens', verdict='FAIL',
                         reason='no token lines in the logs'))
    # The prompt's last row is dumped `_p<index>` where index is the position WITHIN --ids
    # (cli.cpp passes the loop index, and --at-position only moves the engine's pos_);
    # the decode loop dumps `_t<k>`. Mixing the two conventions silently found nothing.
    last = a.prompt - 1
    la, lb = O.read_dump(pa['prefix'], last), O.read_dump(pb['prefix'], last)
    if la is None or lb is None:
        rows.append(dict(tensor=f'logits@prompt_last', verdict='FAIL',
                         reason='prompt dump missing'))
    else:
        rows.append(O.metric(f'logits@prompt_last', lb, la))
        ga, gb = int(np.argmax(la[:O.REAL_VOCAB])), int(np.argmax(lb[:O.REAL_VOCAB]))
        rows.append(dict(tensor='greedy@prompt_last', sequential=ga, block=gb,
                         verdict='PASS' if ga == gb else 'FAIL'))
    for k in range(a.decode):
        fa = Path(f"{pa['prefix']}_t{k}.bin")
        fb = Path(f"{pb['prefix']}_t{k}.bin")
        if not (fa.is_file() and fb.is_file()):
            rows.append(dict(tensor=f'logits@decode{k}', verdict='FAIL',
                             reason='decode dump missing'))
            continue
        x = np.fromfile(fa, dtype=np.float32)
        y = np.fromfile(fb, dtype=np.float32)
        rows.append(O.metric(f'logits@decode{k}', y, x))
        ga, gb = int(np.argmax(x[:O.REAL_VOCAB])), int(np.argmax(y[:O.REAL_VOCAB]))
        rows.append(dict(tensor=f'greedy@decode{k}', sequential=ga, block=gb,
                         verdict='PASS' if ga == gb else 'FAIL'))
    for l, ss in sorted(specs.items()):
        for label, off, n in ss:
            fa = seqdir / f'state_{l}_{off}_{n}.bin'
            fb = blkdir / f'state_{l}_{off}_{n}.bin'
            if not (fa.is_file() and fb.is_file()):
                rows.append(dict(tensor=f'L{l}.{label}', verdict='FAIL',
                                 reason='dump missing'))
                continue
            x = np.fromfile(fa, dtype=np.uint8)
            y = np.fromfile(fb, dtype=np.uint8)
            if x.size % 4 == 0:
                rows.append(O.metric(f'L{l}.{label}', y.view(np.float32), x.view(np.float32)))
            else:
                rows.append(O.metric(f'L{l}.{label}(bytes)', y.astype(np.float64),
                                     x.astype(np.float64)))

    v = O.verdict(rows)
    result = dict(tool='prefill-block-decode-transition', version=1, phase=26,
                  gate='26.12 multi-chunk + decode transition', model_acceptance=False,
                  prompt_tokens=a.prompt, chunk=a.chunk, decode_steps=a.decode,
                  layers=a.layers, position=a.position,
                  # The tail is what makes this a multi-chunk test: prompt %% chunk is
                  # short of a chunk, so the block path pads and the decode path inherits
                  # whatever the padding left behind.
                  tail_tokens=a.prompt % a.chunk,
                  fixture=dict(seed=26026, ids=ids[:16]), tensors=rows, verdict=v,
                  sequential=pa, block=pb)
    print(f'\n=== 26.12 decode transition: {v} '
          f'({a.prompt} prompt tokens as chunks of {a.chunk}, '
          f'tail {a.prompt % a.chunk}, then {a.decode} decode steps, {a.layers} layers) ===')
    for r in rows:
        if 'cosine' in r:
            print(f'  {r["tensor"]:22s} cos {r["cosine"]:.9f}  max_abs {r["max_abs"]:.3e}  '
                  f'{r["verdict"]}')
        else:
            print(f'  {r["tensor"]:22s} {r.get("sequential", r.get("reason"))} vs '
                  f'{r.get("block", "")}  {r["verdict"]}')
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f'{v} -> {out / "result.json"}')
    return 0 if v == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
