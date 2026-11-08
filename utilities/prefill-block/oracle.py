"""Phase 26: the sequential-vs-block state oracle.

Two paths, one token sequence, identical starting state:

  Path A (reference)  the production per-token route: `core.step()` once per token.
                      This is the real open-NPU arithmetic, NOT a Python replica.
  Path B (candidate)  the same tokens through the block route: `--gemm-block`.

Both are the SAME binary and the same weights, differing only in a flag, so the two paths
cannot drift apart in build or in load.

What is compared, per the plan's 26.3: the DeltaNet recurrent state S, the DeltaNet conv
history, the attention KV cache, the absolute position, the final hidden row's logits, and
the greedy token. Each tensor reports cosine, max absolute error, max relative error,
worst index and PASS/FAIL. Padding/unused regions are covered by canaries instead of by
meaningless equality (26.4).

Why the paths are not byte-identical, and must not be required to be: the block route's
projection accumulates over 256 columns where the per-token route accumulates over 1. The
gate is the repository's existing prefill gate (cosine >= 0.9999), applied unchanged --
26.2 forbids inventing a weaker tolerance because the accumulation order changed.

    python oracle.py --stage dnet  --tokens 32
    python oracle.py --stage attn  --tokens 32 --position 17
    python oracle.py --stage all   --tokens 64 --positions 0 17 255 511

Every dump is written under an outdir that is never reused, and the fixture that produced
it is recorded with the result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = ROOT / 'Models/qwen38-27b/prefill-perf/block-v1/build/open_qwen36_cli'
MODEL = ROOT / 'Models/qwen38-27b/Qwen3.8-27B-NPU2'
KERNELS = ROOT / 'Models/qwen38-27b/production-v2/kernels-model'

# Same lock the resident validators take, so an oracle run never overlaps another NPU job.
LOCK = ROOT / 'Models/qwen38-27b/.npu.lock'

GATE_COSINE = 0.9999          # the repository's existing prefill gate; NOT relaxed here
REAL_VOCAB = 248077

# Qwen3.8-27B geometry, from the exported manifest (layer_types in manifest.json).
HIDDEN = 5120
# linear layer: conv history (taps-1) * nch * 4, then S at state_s_off
CONV_BYTES = 3 * 10240 * 4          # 122880, the same offset state_s_off uses
S_OFF = 122880
S_ROWS = 130
KEY_HEADS, VALUE_HEADS, HEAD_DIM = 16, 48, 128
S_BYTES = VALUE_HEADS * S_ROWS * HEAD_DIM * 4     # 3194880; 122880 + this = 3317760
STATE_BYTES_LINEAR = CONV_BYTES + S_BYTES
KV_ROW_BYTES = 8192                 # manifest layer_types.full_attention.buffers.state.row


# ---------------------------------------------------------------- fixture (26.2)

# One draw, long enough for every fixture this harness asks for, so a SHORTER fixture is a
# PREFIX of a longer one and length is the only variable between two runs. Drawing per-n
# (`choice(size=n, replace=False)`) gives a different permutation for every n, which breaks
# that: a 64-token run would not share a single token with the 256-token run it is compared
# against. 512 covers every fixture here with no repetition of the pool.
# Phase 29 raised this from 512: the long-context ladder needs 4096 engine tokens through
# the same distinct-id fixture, and a prefix must stay a prefix across every length so a
# 4K run and a 512-token run share their first 512 rows. Raised deliberately, per the guard
# below -- the ids are still drawn once, distinct, from a fixed seed.
POOL = 8192


def fixture(n: int, seed: int = 26026) -> list[int]:
    """Deterministic, NON-identical token ids; `n` is a prefix of the pool.

    Distinctness is the point: T copies of one id cannot catch a row swap, a cross-token
    mix, or a wrong block position, because every row would expect the same answer.
    Spread over the whole vocab and forced distinct so no two rows coincide.
    """
    if n > POOL:
        raise SystemExit(f'fixture: {n} tokens exceeds the {POOL}-id pool; '
                         f'raise POOL deliberately, do not silently reshuffle')
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(2, REAL_VOCAB), size=POOL, replace=False)
    return [int(i) for i in ids[:n]]


# ---------------------------------------------------------------- running a path

def run_path(ids: list[int], block: bool, workdir: Path, tag: str, *, position: int,
             layers: int, dump_state: list[str], canary: str | None,
             max_tokens: int) -> dict:
    """One path. Returns the parsed metadata and where its dumps landed."""
    workdir.mkdir(parents=True, exist_ok=True)
    idfile = workdir / f'ids_{tag}.txt'
    idfile.write_text(','.join(str(i) for i in ids) + '\n')
    logp = workdir / f'logits_{tag}'
    env = dict(os.environ)
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    env.pop('OFLM_OPEN_PREFILL_ROLES', None)
    env.pop('OFLM_OPEN_PREFILL_ROLE_MODE', None)
    cmd = [str(CLI), '--model', str(MODEL), '--kernels', str(KERNELS),
           '--ids-file', str(idfile), '--max-ctx', '4096',
           '--max-tokens', str(max_tokens), '--layers', str(layers),
           '--pmode', 'none', '--dump-logits', str(logp)]
    if block:
        cmd.append('--gemm-block')
    if position > 0:
        cmd += ['--at-position', str(position)]
    # Several state slices per run: one flag per slice.
    for spec in dump_state:
        cmd += ['--dump-state', spec]
    if canary:
        cmd += ['--state-canary', canary]
    t0 = time.monotonic()
    with LOCK.open('a') as handle:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX)      # never overlap another NPU job
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    (workdir / f'out_{tag}.log').write_text(text)
    if proc.returncode != 0:
        tail = '\n'.join(text.strip().splitlines()[-12:])
        raise SystemExit(f'{tag} failed (rc {proc.returncode}):\n{tail}')
    return dict(tag=tag, block=block, wall_s=round(wall, 2),
                tokens=len(ids), position=position, layers=layers,
                dump_prefix=str(logp), workdir=str(workdir),
                command=' '.join(cmd))


def read_dump(prefix: str, index: int) -> np.ndarray | None:
    """A `_p<index>` logits dump. `index` is the position within --ids, not absolute."""
    p = Path(f'{prefix}_p{index}.bin')
    if not p.is_file():
        return None
    return np.fromfile(p, dtype=np.float32)


# ---------------------------------------------------------------- metrics (26.3)

def metric(name: str, got: np.ndarray, ref: np.ndarray) -> dict:
    """cosine, max abs, max relative, worst index, PASS/FAIL -- per the plan's list."""
    a = np.asarray(got, dtype=np.float64).ravel()
    b = np.asarray(ref, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return dict(tensor=name, shape_a=list(a.shape), shape_b=list(b.shape),
                    verdict='FAIL', reason='shape mismatch')
    if a.size == 0:
        return dict(tensor=name, verdict='FAIL', reason='empty')
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return dict(tensor=name, verdict='FAIL', reason='non-finite')
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    # Cosine is undefined when a reference vector is all-zero; the honest answer then is
    # exact equality, not a free pass.
    if nb == 0.0 or na == 0.0:
        cos = 1.0 if np.array_equal(a, b) else 0.0
    else:
        cos = float(np.dot(a, b) / (na * nb))
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-12)
    rel = diff / denom
    worst_abs = int(np.argmax(diff))
    worst_rel = int(np.argmax(rel))
    return dict(
        tensor=name, n=int(a.size),
        cosine=round(cos, 9),
        max_abs=round(float(diff.max()), 9),
        max_rel=round(float(rel.max()), 9),
        worst_abs_index=worst_abs,
        worst_rel_index=worst_rel,
        # A tensor whose reference is identically zero is compared exactly; there is no
        # scale to normalise by and cosine would be meaningless.
        verdict='PASS' if (cos >= GATE_COSINE) else 'FAIL',
        gate_cosine=GATE_COSINE)


def canary_check(name: str, got: np.ndarray, pattern: int) -> dict:
    """An untouched region must be bit-identical to the pattern that was written."""
    ok = bool(np.all(got == np.uint8(pattern)))
    bad = int(np.count_nonzero(got != np.uint8(pattern)))
    return dict(tensor=name, n=int(got.size), untouched=ok,
                differing_bytes=bad, pattern=pattern,
                verdict='PASS' if ok else 'FAIL')


# ---------------------------------------------------------------- the stages

_LAYER_KINDS: list[str] | None = None


def layer_kinds() -> list[str]:
    """Per layer index, 'linear' or 'full', read from the exported manifest.

    Needed because the state LAYOUT follows the layer type: a linear layer's BO is conv
    history then S, a full layer's is KV rows. Reading one layout as the other still
    compares the same byte ranges between the two paths, so it PASSES while being labelled
    the wrong tensor -- which is exactly what the first 8-layer run did, reporting "conv"
    and "S" for the attention layers at index 3 and 7.
    """
    global _LAYER_KINDS
    if _LAYER_KINDS is None:
        m = json.loads((KERNELS / 'manifest.json').read_text())
        _LAYER_KINDS = [m['layer_types'][t]['gemm_block']['kind'] for t in m['layers']]
    return _LAYER_KINDS


def stage_state_specs(kind: str, *, layers: int, position: int = 0,
                      tokens: int = 1) -> dict[int, list[tuple[str, int, int]]]:
    """The state slices worth comparing, as (label, off, bytes) per layer index.

    The LAYOUT follows the layer type and the STAGE selects which types it covers:
    Stage 1 (26.5) is the DeltaNet layers, Stage 2 (26.6) the attention layers. Reading a
    linear layer's conv/S as though it were KV still compares the same bytes between the
    two paths, so it passes while testing nothing relevant -- and at a large enough
    position the read runs past that layer's smaller BO and fails for the wrong reason.
    """
    kinds = layer_kinds()
    kv_bytes = (position + tokens) * KV_ROW_BYTES
    specs: dict[int, list[tuple[str, int, int]]] = {}
    for l in range(min(layers, len(kinds))):
        if kinds[l] == 'full':
            if kind == 'attn':
                specs[l] = [('kv', 0, kv_bytes)]
        elif kind != 'attn':
            specs[l] = [('conv', 0, CONV_BYTES), ('S', S_OFF, S_BYTES)]
    return specs


def evaluate(pa: dict, pb: dict, token_count: int, position: int) -> dict:
    """Read both paths' dumps and score every tensor they share."""
    rows: list[dict] = []
    # The logits dump is keyed by the index INSIDE --ids, NOT by absolute position: cli.cpp
    # calls dump_pos(prefix, i, ...) with the loop index, and --at-position only moves the
    # engine's pos_ beforehand. Asking for position + token_count - 1 therefore found
    # nothing at every non-zero position and SKIPPED the logits comparison silently --
    # which is why those runs looked clean while comparing one tensor fewer.
    last = token_count - 1
    la = read_dump(pa['dump_prefix'], last)
    lb = read_dump(pb['dump_prefix'], last)
    if la is None or lb is None:
        rows.append(dict(tensor=f'logits@i{last}', verdict='FAIL',
                         reason='logits dump missing; the comparison would be vacuous'))
    else:
        rows.append(metric(f'logits@i{last}', lb, la))
        ga = int(np.argmax(la[:REAL_VOCAB]))
        gb = int(np.argmax(lb[:REAL_VOCAB]))
        rows.append(dict(tensor=f'greedy@i{last}', sequential=ga, block=gb,
                         verdict='PASS' if ga == gb else 'FAIL'))
    # State slices: the files are written by the CLI into each path's workdir, named
    # state_<layer>_<off>_<n>.bin by the spec we passed. A missing file is a FAILURE, not
    # something to skip: the first version of this harness compared exactly one tensor,
    # printed PASS, and had tested almost nothing.
    for layer, specs in sorted(pa['state_specs'].items()):
        for label, off, n in specs:
            fa = Path(pa['workdir']) / f'state_{layer}_{off}_{n}.bin'
            fb = Path(pb['workdir']) / f'state_{layer}_{off}_{n}.bin'
            name = f'L{layer}.{label}'
            if not (fa.is_file() and fb.is_file()):
                rows.append(dict(tensor=name, verdict='FAIL',
                                 reason=f'missing dump ({fa.is_file()}, {fb.is_file()})'))
                continue
            a = np.fromfile(fa, dtype=np.uint8)
            b = np.fromfile(fb, dtype=np.uint8)
            # Both the conv history and S are fp32 for this kernel set (conv_dtype is
            # float32 in the manifest), so compare as floats when the bytes divide.
            if a.size % 4 == 0:
                rows.append(metric(name, b.view(np.float32), a.view(np.float32)))
            else:
                rows.append(metric(f'{name}(bytes)', b.astype(np.float64), a.astype(np.float64)))
    # Own the verdict here, not in the caller: a caller that forgets it prints nothing and
    # a caller that computes it differently is a second source of truth.
    return dict(tensors=rows, first_failure=first_failure(rows), verdict=verdict(rows))


def first_failure(rows: list[dict]) -> dict | None:
    for r in rows:
        if r.get('verdict') == 'FAIL':
            return r
    return None


def verdict(rows: list[dict]) -> str:
    return 'PASS' if all(r.get('verdict') != 'FAIL' for r in rows) else 'FAIL'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['dnet', 'attn', 'both', 'all'], default='dnet')
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--position', type=int, default=0)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=1)
    ap.add_argument('--canary', default='tail', choices=['tail', 'none'])
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if not CLI.is_file():
        raise SystemExit(f'missing {CLI}; run utilities/prefill-block/build-cli.sh')
    if a.tokens < 1:
        raise SystemExit('--tokens must be >= 1')

    ids = fixture(a.tokens)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'{a.stage}-n{a.tokens}-p{a.position}-l{a.layers}-{stamp}')
    if out.exists():
        raise SystemExit(f'{out} exists; a run tree is never reused')
    out.mkdir(parents=True)
    workA, workB = out / 'seq', out / 'block'

    stages = ['dnet', 'attn'] if a.stage == 'both' else (['dnet'] if a.stage == 'all' else [a.stage])
    if a.stage == 'all':
        stages = ['dnet', 'attn', 'llla', 'eight', 'full']
    # A stage whose layer types are absent from this layer window is skipped, not faked.
    result = dict(tool='prefill-block-oracle', version=1, phase=26,
                  model_acceptance=False, precision='paired',
                  fixture=dict(seed=26026, n=a.tokens, ids=ids[:16], pool=POOL,
                               ids_note='first 16 of a distinct deterministic draw; '
                                        'a shorter fixture is a prefix of a longer one',
                               distinct=len(set(ids)) == len(ids)),
                  position=a.position, layers=a.layers,
                  gate_cosine=GATE_COSINE, stages={})
    for st in stages:
        kind = 'attn' if st == 'attn' else 'dnet'
        specs_for_layer = stage_state_specs(kind, layers=a.layers,
                                            position=a.position, tokens=a.tokens)

        # The canary region, as a real dump spec so it is READ BACK rather than assumed.
        # The target MUST be a `full` layer: a `linear` layer's BO holds conv history and S,
        # has no KV rows, and is a different size -- targeting layer 0 put the canary past
        # the end of a 4 MiB buffer and the CLI rejected it, correctly.
        CANARY_PAT = 165
        canary_spec: tuple[int, int, int] | None = None
        if a.canary == 'tail':
            full = [i for i, k in enumerate(layer_kinds()) if k == 'full']
            if full and full[0] < a.layers:
                canary_spec = (full[0], (a.position + a.tokens) * KV_ROW_BYTES, KV_ROW_BYTES)
        dump_for = dict(specs_for_layer)
        if canary_spec is not None:
            l, off, n = canary_spec
            dump_for.setdefault(l, [])
            dump_for[l] = list(dump_for[l]) + [('canary', off, n)]

        # Build the CLI dump flags for BOTH paths, each writing into its own workdir.
        def flags_for(wd: Path) -> list[str]:
            return [f'{l}:{off}:{n}:{wd / f"state_{l}_{off}_{n}.bin"}'
                    for l, ss in dump_for.items() for (lab, off, n) in ss]
        canary_flag = None
        if canary_spec is not None:
            l, off, n = canary_spec
            canary_flag = f'{l}:{off}:{n}:{CANARY_PAT}'
        pa = run_path(ids, False, workA, st, position=a.position, layers=a.layers,
                      dump_state=flags_for(workA), canary=canary_flag, max_tokens=a.max_tokens)
        pb = run_path(ids, True, workB, st, position=a.position, layers=a.layers,
                      dump_state=flags_for(workB), canary=canary_flag, max_tokens=a.max_tokens)
        # The compared specs keep their labels; the canary spec is scored separately.
        pa['state_specs'] = specs_for_layer
        ev = evaluate(pa, pb, a.tokens, a.position)
        if canary_spec is not None:
            l, off, n = canary_spec
            fa = workA / f'state_{l}_{off}_{n}.bin'
            fb = workB / f'state_{l}_{off}_{n}.bin'
            if not (fa.is_file() and fb.is_file()):
                # A canary that was never read back is a canary that proves nothing.
                ev['tensors'].append(dict(tensor=f'L{l}.canary_tail', verdict='FAIL',
                                          reason='canary region was never dumped'))
            else:
                # canary_check already returns a `tensor` key; renaming it after the fact
                # keeps the two paths' results distinguishable in the report.
                r = canary_check(f'L{l}.canary_tail_block',
                                 np.fromfile(fb, dtype=np.uint8), CANARY_PAT)
                ev['tensors'].append(r)
                r = canary_check(f'L{l}.canary_tail_seq',
                                 np.fromfile(fa, dtype=np.uint8), CANARY_PAT)
                ev['tensors'].append(r)
            ev['first_failure'] = first_failure(ev['tensors'])
        ev['verdict'] = verdict(ev['tensors'])
        ev['sequential'] = pa
        ev['block'] = pb
        result['stages'][st] = ev
        print(f'\n=== stage {st}: {ev["verdict"]}  (layers {a.layers}, '
              f'{a.tokens} tokens, position {a.position}) ===')
        for r in ev['tensors']:
            if 'cosine' in r:
                print(f'  {r["tensor"]:28s} cos {r["cosine"]:.9f}  max_abs {r["max_abs"]:.3e}  '
                      f'max_rel {r["max_rel"]:.3e}  worst {r["worst_abs_index"]:8d}  {r["verdict"]}')
            elif 'sequential' in r:
                print(f'  {r["tensor"]:28s} seq {r["sequential"]}  block {r["block"]}  {r["verdict"]}')
            else:
                print(f'  {r["tensor"]:28s} untouched={r.get("untouched")} '
                      f'bad={r.get("differing_bytes")}  {r["verdict"]}')
        if ev['first_failure']:
            print(f'  FIRST DIVERGENCE: {json.dumps(ev["first_failure"])}')
            # 26.17: STOP at the smallest failing composition. Do not go on to bigger ones.
            break
    result['verdict'] = ('PASS' if result['stages'] and
                         all(s['verdict'] == 'PASS' for s in result['stages'].values())
                         else 'FAIL')
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f'\n{"PASS" if result["verdict"] == "PASS" else "FAIL"} -> {out / "result.json"}')
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
