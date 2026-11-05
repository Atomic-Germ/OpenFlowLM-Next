"""Compare sequential and block-prefilled decode through the standalone CLI.

    python decode_transition.py --check state --prompt 40 --decode 2 --layers 8
    python decode_transition.py --check logits --prompt 700 --decode 16 --layers 64

Both modes share dispatch, dump reading and the unchanged oracle cosine gate.
State mode additionally requires exact greedy trajectories and compares final
DeltaNet, conv and KV state. Logits mode reports argmax differences without
making them an acceptance gate. Neither establishes model acceptance.

Block execution disables layer-major routing and uses the physical 256-token
block. The old --chunk argument only changed report metadata, not execution;
values other than 256 are now rejected. Dump indices are relative to the prompt,
even when --position moves the absolute KV position.
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
import oracle as O  # noqa: E402


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
        env['OFLM_OPEN_LAYER_MAJOR'] = '0'
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


def compare_logits(pa, pb, prompt, decode, *, strict):
    """Read every requested row once; argmax is a gate only in state mode."""
    rows, tokens = [], []
    if strict:
        a, b = pa['greedy'], pb['greedy']
        rows.append(dict(tensor='greedy_tokens', sequential=a, block=b,
                         verdict='PASS' if len(a) == len(b) == decode and a == b else 'FAIL'))
    for name, suffix, step in [('prompt_last', f'p{prompt-1}', None),
                               *[(f'decode{k}', f't{k}', k) for k in range(decode)]]:
        paths = [Path(f"{p['prefix']}_{suffix}.bin") for p in (pa, pb)]
        if any(not p.is_file() or p.stat().st_size == 0 or p.stat().st_size % 4 for p in paths):
            rows.append(dict(tensor=f'logits@{name}', verdict='FAIL',
                             reason='missing, empty or malformed logits dump'))
            continue
        x, y = (np.fromfile(p, dtype=np.float32) for p in paths)
        rows.append(O.metric(f'logits@{name}', y, x))
        if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        ga, gb = (int(np.argmax(v[:O.REAL_VOCAB])) for v in (x, y))
        if step is not None:
            tokens.append(dict(step=step, sequential=ga, block=gb, agree=ga == gb))
        if strict:
            rows.append(dict(tensor=f'greedy@{name}', sequential=ga, block=gb,
                             verdict='PASS' if ga == gb else 'FAIL'))
    return rows, tokens


def state_slices(layers, end_position):
    kinds = O.layer_kinds()[:layers]
    lin = max((i for i, k in enumerate(kinds) if k == 'linear'), default=None)
    full = max((i for i, k in enumerate(kinds) if k == 'full'), default=None)
    specs = {}
    if lin is not None:
        specs[lin] = [('conv', 0, O.CONV_BYTES), ('S', O.S_OFF, O.S_BYTES)]
    if full is not None:
        specs[full] = [('kv', 0, end_position * O.KV_ROW_BYTES)]
    return specs


def state_path(workdir, layer, offset, size):
    return workdir / f'state_{layer}_{offset}_{size}.bin'


def compare_state(seqdir, blkdir, specs):
    rows = []
    for layer, slices in sorted(specs.items()):
        for label, offset, size in slices:
            paths = [state_path(wd, layer, offset, size) for wd in (seqdir, blkdir)]
            if any(not p.is_file() or p.stat().st_size != size for p in paths):
                rows.append(dict(tensor=f'L{layer}.{label}', verdict='FAIL',
                                 reason='missing or wrong-sized state dump'))
                continue
            x, y = (np.fromfile(p, dtype=np.uint8) for p in paths)
            if size % 4 == 0:
                rows.append(O.metric(f'L{layer}.{label}', y.view(np.float32), x.view(np.float32)))
            else:
                rows.append(O.metric(f'L{layer}.{label}(bytes)', y.astype(np.float64),
                                     x.astype(np.float64)))
    return rows


def summary(rows, tokens):
    graded = [r for r in rows if r['tensor'].startswith('logits@') and 'cosine' in r]
    return dict(compared=len(graded),
                min_cosine=min((r['cosine'] for r in graded), default=None),
                max_abs=max((r['max_abs'] for r in graded), default=None),
                max_rel=max((r['max_rel'] for r in graded), default=None),
                last_cosine=next((r['cosine'] for r in graded if r['tensor'] == 'logits@prompt_last'), None),
                tokens_agreeing=sum(t['agree'] for t in tokens), tokens_compared=len(tokens),
                first_disagreement=next((t['step'] for t in tokens if not t['agree']), None))


def arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', choices=['state', 'logits'], default='state')
    ap.add_argument('--prompt', type=int, help='prompt IDs (state: 40, logits: 700)')
    ap.add_argument('--decode', type=int, help='decode steps (state: 2, logits: 16)')
    ap.add_argument('--layers', type=int, help='layer count (state: 8, logits: 64)')
    ap.add_argument('--chunk', type=int, choices=[256], default=256,
                    help='physical block width; other widths are not implemented by the CLI')
    ap.add_argument('--position', type=int, default=0)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    defaults = (40, 2, 8) if a.check == 'state' else (700, 16, 64)
    for key, default in zip(('prompt', 'decode', 'layers'), defaults):
        if getattr(a, key) is None:
            setattr(a, key, default)
    if min(a.prompt, a.decode, a.layers) < 1 or a.layers > 64 or a.position < 0:
        ap.error('prompt/decode/layers must be positive, layers <= 64 and position >= 0')
    if a.position + a.prompt + a.decode > 4096:
        ap.error('prompt and decode must fit the 4096-position context')
    return a


def main(argv=None):
    a = arguments(argv)
    strict = a.check == 'state'
    ids = O.fixture(a.prompt, seed=26026)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    tag = (f'decode-n{a.prompt}-c{a.chunk}-d{a.decode}-l{a.layers}' if strict
           else f'decode16-logits-{a.prompt}-{a.decode}')
    out = a.out or (HERE / 'runs' / f'{tag}-{stamp}')
    if out.exists():
        raise SystemExit(f'{out} exists; a run tree is never reused')
    out.mkdir(parents=True)
    seqdir, blkdir = out / 'seq', out / 'block'
    specs = state_slices(a.layers, a.position + a.prompt + a.decode) if strict else {}
    def flags(wd):
        return [f'{layer}:{offset}:{size}:{state_path(wd, layer, offset, size)}'
                for layer, slices in specs.items() for _, offset, size in slices]
    pa = run(ids, seqdir, 'seq', chunk=None, layers=a.layers, decode=a.decode,
             position=a.position, dump_state=flags(seqdir), outdir=str(out))
    pb = run(ids, blkdir, 'block', chunk=a.chunk, layers=a.layers, decode=a.decode,
             position=a.position, dump_state=flags(blkdir), outdir=str(out))
    rows, tokens = compare_logits(pa, pb, a.prompt, a.decode, strict=strict)
    if strict:
        rows.extend(compare_state(seqdir, blkdir, specs))
    result = dict(tool='prefill-block-decode-transition' if strict else 'prefill-block-decode16-logits',
                  version=1, phase=26 if strict else 27, model_acceptance=False,
                  prompt_tokens=a.prompt, chunk=a.chunk, decode_steps=a.decode, layers=a.layers,
                  position=a.position, sequential=pa, block=pb, tensors=rows,
                  verdict=O.verdict(rows))
    if strict:
        result.update(gate='26.12 multi-chunk + decode transition',
                      tail_tokens=a.prompt % a.chunk, fixture=dict(seed=26026, ids=ids[:16]))
    else:
        result.update(gate='27 closure item 8: 16-token trajectory, cosine/max_abs/max_rel',
                      cli=str(O.CLI), kernels=str(O.KERNELS), started=stamp,
                      tolerance='the repository cosine gate from oracle.py; nothing new',
                      tokens=tokens, summary=summary(rows, tokens))
    for row in rows:
        print(f"{row['tensor']:24s} {row['verdict']} "
              f"{row.get('reason', '')}")
    if not strict:
        print(json.dumps(result['summary'], indent=1))
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
