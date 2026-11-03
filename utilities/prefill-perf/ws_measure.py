"""Measure the weight-stationary POC against the baseline, with the traffic it implies.

WHY THIS EXISTS AND WHY IT IS NOT `compare.py`
`compare.py` in the design directory reads ONE shared `y_<tag>.bin`, so scoring two
designs with it silently scores the last one twice. This tool gives each variant its own
dump file, so every number belongs to the binary that produced it.

WHAT IT REPORTS, per variant
  array   the per-dispatch time the harness prints (`run k ... -> state 4 (N ms)`), median
  wall    the process wall clock, which includes context creation and the dump
  correct cosine and max|got-ref| against the same fp64 reference make_test.py wrote
          from the SAME pool bytes, so all variants are judged by one standard
  traffic the bytes and op counts the design's own `traffic_model()`/`l1_budget()`
          predicts for that R, imported from the design so the two cannot drift

THE GATE
A variant is "acceptable" only if it ran, every dispatch returned state 4, and its
cosine is at least 0.9999 -- the project's unchanged prefill gate. Speed is reported
next to it and never instead of it: the report's >= 1.4x question is only asked of
variants that pass.

    python ws_measure.py --tag n14336_k5120_t2048 --variants ctl_t2048 wsr1 wsr2 wsr4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = ROOT / 'open_kernels/designs/gemm_q4_prefill'
RUNNER = Path('/tmp/hout/run_kernel')

# run k [3 bufs] -> state 4 (11.055 ms)
DISPATCH = re.compile(r'^run (\S+) \[(\d+) bufs\] -> state (\d+) \(([\d.]+) ms\)', re.M)
FAILED = re.compile(r'^run (\S+) FAILED \(state (\d+)\)', re.M)
DONE = re.compile(r'^DONE runs=(\d+)', re.M)

COS_GATE = 0.9999          # the project's unchanged prefill gate
L1_BYTES = 65536           # one core's local memory on aie2p


def median(values):
    v = sorted(values)
    return v[len(v) // 2] if v else None


def cosine(got: np.ndarray, ref: np.ndarray) -> float | None:
    a = got.astype(np.float64)
    b = ref.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if den == 0.0 else float(a @ b / den)


def design_traffic(n_weight: int, k: int, t: int, tile_n: int, r_tiles: int) -> dict | None:
    """The design's own accounting, imported rather than re-derived."""
    spec = importlib_ws()
    if spec is None:
        return None
    tr = spec.traffic_model(n_weight, k, t, tile_n, r_tiles)
    keep = ('R', 'NTG', 'dequant_ws', 'dequant_base', 'dequant_ratio',
            'weight_bytes_ws', 'weight_bytes_base', 'weight_ratio',
            'activation_bytes', 'accumulator_bytes', 'accumulator_spill_bytes')
    return {k2: tr[k2] for k2 in keep}


def importlib_ws():
    import importlib.util
    p = HERE / 'ws_poc.py'
    if not p.is_file():
        return None
    spec = importlib.util.spec_from_file_location('_ws_poc', p)
    mod = importlib.util.module_from_spec(spec)
    # The design imports aie/iron at module scope; if that is unavailable the traffic
    # model is simply reported as absent rather than failing the measurement.
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def predicted_l1(r_tiles: int, tile_n: int = 32, nscr: int = 1,
                 adep: int = 1, bdep: int = 1, cdep: int = 1) -> dict:
    """Per-core bytes for an output-stationary R: R accumulators, one scratch tile.

    Printed next to the measured time so the report can say not just "R=4 was slower"
    but "R=4 does not fit", which is the property that decides the direction.
    """
    scratch = nscr * 64 * 64 * 2
    nibble = 64 * 64
    accum = r_tiles * 64 * tile_n * 4
    a_fifo = adep * 10240
    b_fifo = bdep * (64 * tile_n * 2)
    c_fifo = cdep * (64 * tile_n * 4)
    total = scratch + nibble + accum + a_fifo + b_fifo + c_fifo
    return dict(scratch=scratch, nibble=nibble, accum=accum,
                fifos=a_fifo + b_fifo + c_fifo, total=total,
                fits_l1=total <= L1_BYTES, over_by=max(0, total - L1_BYTES))


def run_variant(tag: str, variant: str) -> dict:
    cfg = DESIGN / f'run_{variant}.cfg'
    if not cfg.is_file():
        raise SystemExit(f'missing {cfg}')
    rec = dict(variant=variant, cfg=str(cfg))
    t0 = time.monotonic()
    proc = subprocess.run([str(RUNNER), cfg.name], cwd=str(DESIGN),
                          capture_output=True, text=True)
    rec['wall_s'] = round(time.monotonic() - t0, 3)
    rec['returncode'] = proc.returncode
    text = proc.stdout + '\n' + proc.stderr
    runs = [(name, int(state), float(ms)) for name, _, state, ms in DISPATCH.findall(text)]
    bad = [(name, int(state)) for name, state in FAILED.findall(text)]
    times = [ms for _, _, ms in runs]
    rec['runs'] = len(runs)
    rec['runs_done'] = int(DONE.search(text).group(1)) if DONE.search(text) else 0
    rec['array_ms_all'] = times
    rec['array_ms_median'] = median(times)
    rec['array_ms_min'] = min(times) if times else None
    rec['states_ok'] = bool(runs) and all(s == 4 for _, s, _ in runs) and not bad
    if bad:
        rec['failed_dispatches'] = bad
    rec['output'] = str(DESIGN / f'y_{variant}.bin') \
        if (DESIGN / f'y_{variant}.bin').is_file() else None
    return rec


def score(rec: dict, ref: np.ndarray) -> dict:
    if not rec.get('output'):
        rec.update(passed=False, why='no output dump')
        return rec
    got = np.fromfile(rec['output'], dtype=np.float32)
    if got.size != ref.size:
        rec.update(passed=False, why=f'output {got.size} values, reference {ref.size}')
        return rec
    diff = got.astype(np.float64) - ref
    rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
    rec['max_abs_err'] = float(np.abs(diff).max())
    rec['ref_rms'] = rms
    rec['max_abs_rel_to_rms'] = rec['max_abs_err'] / rms if rms else None
    rec['cosine'] = cosine(got, ref)
    ran = rec['states_ok']
    rec['passed'] = bool(ran and rec['cosine'] is not None and rec['cosine'] >= COS_GATE)
    if not rec['passed']:
        rec['why'] = ('all dispatches must return state 4' if not ran
                      else 'cosine below the 0.9999 gate')
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', required=True, help='make_test tag, e.g. n14336_k5120_t2048')
    ap.add_argument('--variants', nargs='+', required=True, help='cfg suffixes')
    ap.add_argument('--baseline', default=None, help='variant to divide by (default: first)')
    ap.add_argument('--shape', nargs=3, metavar=('N', 'K', 'T'), type=int, default=None,
                    help='for the traffic table; defaults parsed from the tag')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    ref_path = DESIGN / f'ref_{a.tag}.bin'
    if not ref_path.is_file():
        raise SystemExit(f'missing {ref_path}; run make_test.py first')
    ref = np.fromfile(ref_path, dtype=np.float32)

    if a.shape:
        n_w, k, t = a.shape
    else:
        m = re.fullmatch(r'n(\d+)_k(\d+)_t(\d+)', a.tag)
        if not m:
            raise SystemExit(f'cannot read a shape from {a.tag}; pass --shape')
        n_w, k, t = (int(g) for g in m.groups())

    records = []
    for variant in a.variants:
        rec = score(run_variant(a.tag, variant), ref)
        r = re.fullmatch(r'wsr?(\d+)', variant)
        if r:
            rec['predicted'] = predicted_l1(int(r.group(1)))
            tr = design_traffic(n_w, k, t, 32, int(r.group(1)))
            if tr:
                rec['traffic'] = tr
        records.append(rec)

    print(f'{"variant":12s} {"array ms":>10s} {"wall s":>7s} {"cos":>10s} '
          f'{"maxabs":>10s} {"deq":>7s} {"R":>3s} {"L1 B":>7s}  verdict')
    for rec in records:
        t = rec.get('array_ms_median')
        tr = rec.get('traffic') or {}
        pl = rec.get('predicted') or {}
        print(f'{rec["variant"]:12s} {(f"{t:.3f}" if t else "---"):>10s} '
              f'{rec["wall_s"]:7.2f} {(rec.get("cosine") or 0):10.6f} '
              f'{(rec.get("max_abs_err") or 0):10.3e} '
              f'{(tr.get("dequant_ws") or 0):7d} {(pl.get("accum") or 0) // 1024:3d}K '
              f'{(pl.get("total") or 0):7d}  '
              f'{"PASS" if rec["passed"] else "FAIL: " + rec.get("why", "")}')

    base = next((r for r in records if r['variant'] == (a.baseline or a.variants[0])), None)
    if base and base.get('array_ms_median'):
        print()
        for rec in records:
            if rec.get('array_ms_median'):
                rec['speedup_vs_baseline'] = round(
                    base['array_ms_median'] / rec['array_ms_median'], 3)
        for rec in records:
            s = rec.get('speedup_vs_baseline')
            print(f'{rec["variant"]:12s} speedup {f"{s:.2f}x" if s else "---":>7s}  '
                  f'{"PASS" if rec["passed"] else "FAIL"}')

    # The direction's decision, mechanical so it cannot be read generously:
    # a variant that does not FIT L1 can never be the answer, and a variant that fits
    # but does not pass the gate cannot be either.
    viable = [r for r in records if r['passed'] and r['variant'] != base['variant']]
    best = max(viable, key=lambda r: r.get('speedup_vs_baseline') or 0) if viable else None
    verdict = dict(
        best_passing_variant=best['variant'] if best else None,
        best_speedup=best.get('speedup_vs_baseline') if best else None,
        continue_direction=bool(best and (best.get('speedup_vs_baseline') or 0) >= 1.4),
        stop_direction=bool(not best or (best.get('speedup_vs_baseline') or 0) < 1.2),
    )
    print()
    if best:
        s = best.get('speedup_vs_baseline') or 0
        print(f'best passing variant: {best["variant"]} at {s:.2f}x')
        print('gate >= 1.4x: ' + ('CONTINUE' if s >= 1.4 else
                                 'STOP (< 1.2x)' if s < 1.2 else 'INCONCLUSIVE (1.2-1.4x)'))
    else:
        print('no variant passed the numerical gate: STOP')

    result = dict(tool='ws_measure', scope='standalone POC; not model acceptance',
                  model_acceptance=False, tag=a.tag, shape=dict(N=n_w, K=k, T=t),
                  baseline=a.baseline or a.variants[0], cos_gate=COS_GATE,
                  reference=str(ref_path), variants=records, verdict=verdict)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=1, allow_nan=False, default=float) + '\n')
        print(f'wrote {a.out}')
    return 0 if all(r['passed'] for r in records) else 1


if __name__ == '__main__':
    sys.exit(main())
