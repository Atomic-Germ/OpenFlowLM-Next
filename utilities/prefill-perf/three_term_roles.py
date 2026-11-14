"""Phase 23: does the `three_term` failure belong to a projection ROLE?

One bounded experiment. `three_term` (drop only `W_lo X_lo`) is 23 % faster and fails
the 0.9999 gate on 1 of 513 reference positions. This asks whether that failure is
concentrated in one projection class -- in which case a mixed policy could keep the
accepted arithmetic just there and take the speed everywhere else -- or whether it is
spread across roles, in which case there is no structurally clean policy.

Roles are the separable ones the export actually has (see prefill_precision_switch.hpp):
the packer fuses qkv+z, q+k+v+gate and up+gate into single GEMMs, so those three cannot
be split without repacking, and DeltaNet alpha/beta is not a GEMM at all. The sweep
therefore covers exactly what the runtime can select, and says so.

Every variant records wall time, GEMM time, min cosine, failure count, worst position
and the next-token choice. Nothing is specialised to a position: the gate is the
project's own 0.9999 and the reference set is unchanged.

    python three_term_roles.py --out phase23-attribution.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = ROOT / 'Models/qwen38-27b/prefill-perf/switch-v1/build/open_qwen36_cli'
MODEL = ROOT / 'Models/qwen38-27b/Qwen3.8-27B-NPU2'
KERNELS = ROOT / 'Models/qwen38-27b/production-v2/kernels-model'
IDS = ROOT / 'Models/qwen38-27b/prefill-perf/dispatch-v1/ids256.txt'
REFDIR = ROOT / 'Models/qwen38-27b/prefill-long/boundaries-513/513'

CORR_GATE = 0.9999
REAL_VOCAB = 248077
POSITIONS = 513
# re.M matters: without it `^` only anchors at the start of the whole text, the stage
# line never matches, and every variant reported GEMM 0.0 -- a silent zero in the one
# column that says how much of the run the arithmetic change actually touched.
STAGE = re.compile(r'^\s*gemm-prompt \[0,(?P<n>\d+)\) t_real=(?P<t>\d+):\s+'
                   r'(?P<total>[\d.]+) ms \(GEMM (?P<gemm>[\d.]+),\s+host (?P<host>[\d.]+)',
                   re.M)
# The policy the sweep is really asking about: `three_term` for every role except the
# one that fails. Named here, not hard-coded later, so the run and the report agree.
COMBINED = ('qkvz,attn_in,out,ffn_up', 'all_but_ffn_down')

# The roles a build can select, and the ones the plan names that it cannot.
TESTABLE = ('qkvz', 'attn_in', 'out', 'ffn_up', 'ffn_down')
NOT_TESTABLE = {
    'delta_alpha_beta': 'runs in the glue kernel, not a projection GEMM; no mode of '
                        'this switch reaches it',
    'attention_k_only': 'fused with q, v and gate into one n14336_k5120 GEMM',
    'attention_v_only': 'fused with q, k and gate into one n14336_k5120 GEMM',
    'attention_gate_only': 'fused with q, k and v into one n14336_k5120 GEMM',
    'attention_q_only': 'fused with k, v and gate into one n14336_k5120 GEMM',
    'ffn_up_only': 'fused with gate into one n34816_k5120 GEMM',
    'ffn_gate_only': 'fused with up into one n34816_k5120 GEMM',
    'qkv_only': 'fused with z into one n16384_k5120 GEMM',
    'z_only': 'fused with qkv into one n16384_k5120 GEMM',
    'lm_head': 'not part of the block prefill: the tail runs once, over the last '
               'position only, and has no paired/three_term variant',
}


def cosine(got: np.ndarray, ref: np.ndarray) -> float:
    a = got[:REAL_VOCAB].astype(np.float64)
    b = ref[:REAL_VOCAB].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if den == 0.0 else float(a @ b / den)


def run(name: str, env_roles: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(**{'OFLM_OPEN_PREFILL_ROLES': env_roles,
                  'OFLM_OPEN_PREFILL_ROLE_MODE': 'three_term',
                  'OFLM_OPEN_PREFILL_PRECISION': 'paired'})
    import os
    full = dict(os.environ)
    full.update(env)
    t0 = time.monotonic()
    proc = subprocess.run(
        [str(CLI), '--model', str(MODEL), '--kernels', str(KERNELS),
         '--ids-file', str(IDS), '--max-ctx', '2048', '--max-tokens', '1',
         '--gemm-block', '--prefill-logits', '--dump-logits', str(outdir / 'p')],
        cwd=str(ROOT), env=full, capture_output=True, text=True)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    rec = {'variant': name, 'roles': env_roles, 'returncode': proc.returncode,
           'wall_s': round(wall, 2)}
    m = STAGE.search(text)
    if m:
        rec['stage_ms'] = float(m.group('total'))
        rec['gemm_ms'] = float(m.group('gemm'))
        rec['host_ms'] = float(m.group('host'))
    rec['logged_role_line'] = next(
        (ln for ln in text.splitlines() if 'prefill role precision' in ln), None)
    return rec


def score(rec: dict, pos: list, outdir: Path) -> dict:
    cs = []
    missing = 0
    for p in pos:
        f, r = outdir / f'p_p{p}.bin', REFDIR / f'ref_p{p}.bin'
        if not (f.is_file() and r.is_file()):
            missing += 1
            continue
        cs.append((p, cosine(np.fromfile(f, dtype=np.float32),
                             np.fromfile(r, dtype=np.float32))))
    if not cs:
        rec.update(scored=False, why='no capturable position')
        return rec
    corr = [c for _, c in cs]
    fails = [(p, c) for p, c in cs if c < CORR_GATE]
    worst = min(cs, key=lambda r: r[1])
    rec.update(scored=True, positions=len(cs), missing=missing,
               min_corr=min(corr), p50_corr=sorted(corr)[len(corr) // 2],
               failures=len(fails), failing_positions=[p for p, _ in fails][:10],
               worst_position=worst[0], worst_corr=worst[1])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=HERE / 'phase23-attribution.json')
    ap.add_argument('--workdir', type=Path,
                    default=ROOT / 'Models/qwen38-27b/prefill-perf/phase23')
    ap.add_argument('--roles', nargs='+', default=list(TESTABLE))
    ap.add_argument('--no-control', action='store_true',
                    help='skip the paired/paired control run')
    a = ap.parse_args()
    pos = list(range(POSITIONS))

    variants = []
    runs = [('control_paired', '')] if not a.no_control else []
    runs += [(r, r) for r in a.roles]
    runs.append((COMBINED[1], COMBINED[0]))
    print(f'{"variant":16s} {"wall s":>7s} {"GEMM ms":>9s} {"min cos":>11s} '
          f'{"fail":>5s} {"worst":>7s}')
    for name, role in runs:
        # Score the directory the run WROTE to (`name`), not the role string: the
        # control's role is empty and the combined policy's role is a comma list, so
        # keying the read path off the role looked for dumps that never existed and
        # silently reported "no capturable position" for exactly the two runs that
        # matter most.
        rec = run(name, role, a.workdir / name)
        rec = score(rec, pos, a.workdir / name)
        variants.append(rec)
        if rec.get('scored'):
            gain = rec.get('e2e_gain_pct')
            print(f'{name:16s} {rec["wall_s"]:7.2f} {(rec.get("gemm_ms") or 0):9.1f} '
                  f'{rec["min_corr"]:11.9f} {rec["failures"]:5d} '
                  f'P{rec["worst_position"]:<5d} '
                  f'{"" if gain is None else f"gain {gain:+.1f}%"}'
                  f'{"  FAILING " + str(rec["failing_positions"]) if rec["failures"] else ""}')
        else:
            print(f'{name:16s} {rec["wall_s"]:7.2f}  not scored: {rec.get("why")}')

    scored = [v for v in variants if v.get('scored')]
    base = next((v for v in scored if v['variant'] == 'control_paired'), None)
    combined = next((v for v in scored if v['variant'] == COMBINED[1]), None)
    if base and base.get('stage_ms'):
        for v in scored:
            if v.get('stage_ms'):
                v['e2e_gain_pct'] = round((base['stage_ms'] - v['stage_ms'])
                                          / base['stage_ms'] * 100, 2)
    clean = [v for v in scored if v['failures'] == 0]
    single = [v for v in clean if v['variant'] in TESTABLE]
    verdict = {
        'roles_with_zero_failures': [v['variant'] for v in single],
        'roles_with_failures': {v['variant']: v['failures']
                                for v in scored if v['failures']},
        'structurally_clean_single_role_exists': bool(single),
        'combined_policy': COMBINED[1],
        'combined_failures': combined['failures'] if combined else None,
        'combined_e2e_gain_pct': combined.get('e2e_gain_pct') if combined else None,
        'promotion_gate': 'zero gate failures AND >= 10 % end-to-end prefill gain',
    }
    if combined:
        verdict['combined_eligible'] = bool(combined['failures'] == 0
                                            and (combined.get('e2e_gain_pct') or 0) >= 10.0)
    result = dict(tool='three_term_roles',
                  scope='Phase 23 bounded structural attribution; not model acceptance',
                  model_acceptance=False, commit=subprocess.run(
                      ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
                      capture_output=True, text=True).stdout.strip(),
                  gate=CORR_GATE, positions=POSITIONS, fixture=str(IDS),
                  reference=str(REFDIR), testable_roles=list(TESTABLE),
                  not_testable=NOT_TESTABLE, variants=variants, verdict=verdict)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False, default=float) + '\n')
    print(f'\nwrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
