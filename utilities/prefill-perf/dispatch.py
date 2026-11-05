"""Phase 21: what the NPU was actually asked to do, and at what efficiency.

The budget in `measure.py` says where wall time went. This says WHY: the per-kernel
dispatch table the engine can emit (`OFLM_OPEN_DISPATCH_LOG=1`), turned into call
counts, FLOPs and achieved TFLOP/s per GEMM shape.

That distinction matters because the two numbers point at different fixes. A 24.5 s
"host blocked on GEMM" column reads as a synchronization problem until you divide the
FLOPs by it: the Qwen3.8 prefill GEMMs sustain ~1.9 TFLOP/s, which is ~12 % of a
16-TOPS part, so the time is inside the GEMM, not around it. Guessing at the fix from
the budget alone would have been wrong.

The table is parsed from the engine's own output, so a shape that silently stops being
dispatched (or starts being dispatched twice) shows up here as a changed call count
rather than as an unexplained millisecond.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# The Qwen3.8-27B shapes, as the manifest emits them: kernel name -> (N, K, T).
# N and K come from the name (`gemm_n<N>_k<K>`), T is the compiled block width.
SHAPE = re.compile(r'gemm_n(?P<n>\d+)_k(?P<k>\d+)(?P<low>_low)?')
DISPATCH = re.compile(
    r'^\s+(?P<name>\S+)\s+(?P<calls>\d+)\s+calls\s+(?P<total>[\d.]+)\s+ms\s+total'
    r'\s+(?P<mean>[\d.]+)\s+mean\s+(?P<min>[\d.]+)\s+min\s*$')
# The stage line: "gemm-prompt [0,513) t_real=513: 105345.2 ms (GEMM 75612.6, host ...)"
STAGE = re.compile(r'\s*(?P<what>\S+)\s+\[(?P<n0>\d+),(?P<n1>\d+)\)\s+t_real=(?P<t_real>\d+):\s+'
                   r'(?P<total>[\d.]+)\s+ms\s+\(GEMM\s+(?P<gemm>[\d.]+),\s+host\s+(?P<host>[\d.]+)')

# Nominal peak for this part, so utilisation is stated against something explicit
# instead of being left implicit. The XDNA2 NPU in Strix Point is a 16-TOPS-class
# INT8/bf16 engine; the number is a reference point, not a spec claim about the model.
PEAK_TERA_OPS = 16.0


def parse_stage(text):
    """The engine's per-prompt stage line, or None when the run printed no block route."""
    m = STAGE.search(text)
    if not m:
        return None
    return dict(what=m['what'], n0=int(m['n0']), n1=int(m['n1']), t_real=int(m['t_real']),
                total_ms=float(m['total']), gemm_ms=float(m['gemm']), host_ms=float(m['host']))


def parse_dispatches(text):
    """Every "<kernel> N calls X ms total Y mean Z min" row, in file order."""
    rows = []
    for line in text.splitlines():
        m = DISPATCH.match(line)
        if not m:
            continue
        rows.append(dict(name=m['name'], calls=int(m['calls']), total_ms=float(m['total']),
                         mean_ms=float(m['mean']), min_ms=float(m['min'])))
    return rows


def flops(name, tokens):
    """2*N*K*tokens for a GEMM shape, or None for a kernel the pattern does not cover."""
    m = SHAPE.search(name)
    if not m:
        return None
    n, k = int(m['n']), int(m['k'])
    # A `_low` companion is the same shape against the low half of the activation, so it
    # carries the same FLOPs -- counting it is what makes the paired route's cost visible.
    return 2.0 * n * k * tokens


def efficiency(rows, block, peak=PEAK_TERA_OPS):
    """Per-kernel call count, FLOPs per CALL and achieved TFLOP/s.

    One dispatch runs one compiled block of `block` tokens, so the denominator of the
    per-call rate is that block, NOT the prompt. Getting this wrong is not a rounding
    detail: a 513-token prompt makes three 256-wide calls per GEMM, and dividing the
    prompt's FLOPs by the sum of all three calls' time reported 0.011 TFLOP/s for a GEMM
    that sustains ~2 TFLOP/s per dispatch -- a 200x understatement that would have
    argued against the very fix the measurement supports.

    `achieved` is on the dispatches' OWN elapsed time (submit + wait), which is the only
    honest denominator: dividing by wall time would fold the host stages in.
    """
    if block <= 0:
        raise ValueError('a block width must be positive')
    out = []
    for r in rows:
        fl = flops(r['name'], block)
        if fl is None:
            continue
        secs = r['total_ms'] / 1e3
        out.append(dict(name=r['name'], calls=r['calls'], total_ms=r['total_ms'],
                        mean_ms=r['mean_ms'], min_ms=r['min_ms'],
                        gflop_per_call=fl / 1e9, calls_gflop=r['calls'] * fl / 1e9,
                        # The rate a CALL achieves: FLOPs of one block over one call's time.
                        tflops=(fl / 1e12) / (r['mean_ms'] / 1e3) if r['mean_ms'] > 0 else None,
                        peak_fraction=((fl / 1e12) / (r['mean_ms'] / 1e3) / peak
                                       if r['mean_ms'] > 0 else None),
                        paired=bool(SHAPE.search(r['name']).group('low'))))
    return out


def prompt_throughput(rows, tokens, block, peak=PEAK_TERA_OPS):
    """What the PROMPT got, as opposed to what one dispatch achieves.

    Three penalties sit between the per-call rate and the useful rate, and they have
    different fixes, so they are reported separately:

      * block padding -- a 513-token prompt occupies 3 blocks of 256 = 768 slots;
      * the paired `_low` products -- half the GEMM column buys precision, not work;
      * the host stages, which the GEMM column excludes by construction.
    """
    if tokens <= 0 or block <= 0:
        raise ValueError('token count and block width must be positive')
    eff = efficiency(rows, block, peak)
    if not eff:
        raise ValueError('no GEMM rows to score')
    blocks = -(-tokens // block)
    padded = blocks * block
    high = [r for r in eff if not r['paired']]
    low = [r for r in eff if r['paired']]
    # `calls_gflop` is GFLOP; a rate in TFLOP/s is GFLOP / 1000 / seconds.
    gemm_ms = sum(r['total_ms'] for r in eff)
    achieved_gflop = sum(r['calls_gflop'] for r in eff)
    useful_gflop = sum(r['calls_gflop'] for r in high) * (tokens / padded)
    secs = gemm_ms / 1e3
    return dict(blocks=blocks, padded_tokens=padded, padding_factor=padded / tokens,
                gemm_ms=gemm_ms,
                achieved_tflops=(achieved_gflop / 1e3) / secs if secs > 0 else None,
                useful_tflops=(useful_gflop / 1e3) / secs if secs > 0 else None,
                low_share=(sum(r['total_ms'] for r in low) / gemm_ms) if gemm_ms else None)


def paired_split(rows, block, peak=PEAK_TERA_OPS):
    """How much of the GEMM column the paired `_low` products cost.

    The plan's first suspect is the four-product paired route. This splits the column
    into high and low so the answer is a number rather than a belief.
    """
    eff = efficiency(rows, block, peak)
    high = sum(r['total_ms'] for r in eff if not r['paired'])
    low = sum(r['total_ms'] for r in eff if r['paired'])
    return dict(high_ms=high, low_ms=low, total_ms=high + low,
                low_share=(low / (high + low)) if (high + low) else None)


def budget(stage, dispatch_ms):
    """Where the prompt's wall time went, from the engine's own stage line."""
    if not stage:
        raise ValueError('no block-route stage line: this run took the per-token path')
    unexplained = stage['total_ms'] - stage['gemm_ms'] - stage['host_ms']
    return dict(stage=stage['what'], tokens=stage['t_real'], total_ms=stage['total_ms'],
                gemm_ms=stage['gemm_ms'], host_ms=stage['host_ms'],
                unexplained_ms=unexplained,
                coverage=1.0 - max(unexplained, 0.0) / stage['total_ms'] if stage['total_ms'] else None,
                gemm_share=stage['gemm_ms'] / stage['total_ms'] if stage['total_ms'] else None,
                dispatch_ms=dispatch_ms)


def verdict(d, minimum=0.90):
    """Phase 21 asks for >=90% attribution; it does not ask the machine to be fast.

    Kept separate from `budget` because "is the time explained" and "is the
    implementation fast" are different questions, and conflating them is how a
    slow-but-explained result gets reported as accepted.
    """
    d = dict(d)
    d['covered'] = d['coverage'] is not None and d['coverage'] >= minimum
    d['coverage_min'] = minimum
    # The dispatch column is the GEMM column seen from the device, so it must agree with
    # it. A large gap means the engine's accounting and the dispatch log disagree, which
    # is a measurement bug, not a performance finding.
    d['dispatch_gap_ms'] = d['gemm_ms'] - d['dispatch_ms']
    d['dispatch_agree'] = abs(d['dispatch_gap_ms']) <= 0.05 * max(d['gemm_ms'], 1.0)
    return d


def run(a):
    text = Path(a.log).read_text(errors='replace')
    stage = parse_stage(text)
    rows = parse_dispatches(text)
    if not rows:
        raise SystemExit('no dispatch table in the log: run with OFLM_OPEN_DISPATCH_LOG=1 and --gemm-block')
    tokens = stage['t_real'] if stage else a.tokens
    dispatch_ms = sum(r['total_ms'] for r in rows)
    result = dict(complete=True, model_acceptance=False, log=str(Path(a.log).resolve()),
                  scope='Phase 21 prefill characterisation of one production prompt; '
                        'not an acceptance decision')
    result.update(verdict(budget(stage, dispatch_ms), a.coverage_min))
    result['kernels'] = efficiency(rows, a.block)
    result['paired'] = paired_split(rows, a.block)
    result['prompt'] = prompt_throughput(rows, tokens, a.block)
    result['block'] = a.block
    result['peak_tera_ops'] = PEAK_TERA_OPS

    print(f"tokens {result['tokens']}   total {result['total_ms']:.1f} ms   "
          f"coverage {result['coverage']:.1%} (min {a.coverage_min:.0%}) -> "
          f"{'OK' if result['covered'] else 'NOT COVERED'}")
    print(f"  GEMM {result['gemm_ms']:.1f} ms ({result['gemm_share']:.1%})   "
          f"host {result['host_ms']:.1f} ms   unexplained {result['unexplained_ms']:.1f} ms")
    print(f"  dispatch column {dispatch_ms:.1f} ms, "
          f"{'agrees' if result['dispatch_agree'] else 'DISAGREES'} with the GEMM column")
    print(f"  per CALL (one {a.block}-token block)      calls  GFLOP/call  TFLOP/s   %peak@16")
    for k in result['kernels']:
        t = 'n/a' if k['tflops'] is None else f"{k['tflops']:.4f}"
        p = 'n/a' if k['peak_fraction'] is None else f"{100 * k['peak_fraction']:.1f}%"
        print(f"  {k['name']:26s} {k['calls']:5d} {k['gflop_per_call']:10.1f} {t:>9s} {p:>9s}")
    p = result['paired']
    print(f"  paired `_low` products: {p['low_ms']:.1f} of {p['total_ms']:.1f} ms "
          f"({p['low_share']:.1%})")
    q = result['prompt']
    print(f"  prompt throughput: {q['blocks']} blocks for {tokens} tokens "
          f"({q['padding_factor']:.2f}x padding), achieved "
          f"{q['achieved_tflops']:.3f} TFLOP/s, useful {q['useful_tflops']:.3f} TFLOP/s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0 if result['covered'] else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log', type=Path, required=True,
                    help='CLI stderr from a --gemm-block run with OFLM_OPEN_DISPATCH_LOG=1')
    ap.add_argument('--tokens', type=int, help='prompt tokens, when the log has no stage line')
    ap.add_argument('--block', type=int, default=256,
                    help='the compiled block width one dispatch runs (every set this repo emits is 256)')
    ap.add_argument('--coverage-min', type=float, default=0.90)
    ap.add_argument('--out', type=Path)
    raise SystemExit(run(ap.parse_args()))
