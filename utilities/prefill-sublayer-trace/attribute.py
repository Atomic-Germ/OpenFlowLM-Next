"""CPU-only vector attribution of retained hybrid FFN captures; no model acceptance."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    '_sublayer_attribution_check', Path(__file__).with_name('check.py'))
sublayer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sublayer)
base, trace = sublayer.base, sublayer.trace


def silu(x):
    # Stable FP64 evaluation, including large negative synthetic inputs.
    x = np.asarray(x, np.float64)
    return x * np.exp(-np.logaddexp(0., -x))


def decompose(c, nw, up, gate, down, eps, output, reference):
    """Exact algebraic decomposition, not a simulation of corrected NPU kernels.

    Projections are conditioned on captured norm. Up-only and gate-only terms
    use ideal projections at that norm; their nonlinear interaction is explicit.
    All vectors use the same retained final-output reference and scale.
    """
    res, norm, ug, h, y = [np.asarray(c[k], np.float64)
                           for k in ('res', 'norm', 'upgate', 'h', 'down')]
    nw, up, gate, down, output, reference = [
        np.asarray(x, np.float64) for x in (nw, up, gate, down, output, reference)]
    hidden, ff = res.size, h.size
    if (not hidden or not ff or res.shape != (hidden,) or norm.shape != res.shape
            or y.shape != res.shape or output.shape != res.shape
            or reference.shape != res.shape or nw.shape != res.shape
            or h.shape != (ff,) or ug.shape != (2 * ff,)
            or up.shape != (ff, hidden) or gate.shape != up.shape
            or down.shape != (hidden, ff) or not np.isfinite(eps) or eps <= 0):
        raise ValueError('invalid attribution shape/epsilon')
    if not all(np.isfinite(x).all() for x in
               (res, norm, ug, h, y, nw, up, gate, down, output, reference)):
        raise ValueError('non-finite attribution input')

    # Preserve the existing oracle's FP32 norm and activation boundaries.
    norm0 = (base.oracle.rms(res, eps) * nw).astype(np.float32).astype(np.float64)
    u0, g0 = up @ norm0, gate @ norm0
    h0 = base.oracle.silu(g0) * u0
    h0_round = h0.astype(np.float32).astype(np.float64)
    un, gn = up @ norm, gate @ norm
    sn = silu(gn)
    uc, gc = ug[:ff], ug[ff:]
    sc = silu(gc)
    hn, hc = un * sn, uc * sc
    du, ds = uc - un, sc - sn
    ideal_output = res + down @ h0_round
    vectors = dict(
        propagated=ideal_output - reference,
        oracle_activation_rounding_bridge=down @ (h0 - h0_round),
        norm=down @ (hn - h0),
        up=down @ (du * sn),
        gate=down @ (un * ds),
        interaction=down @ (du * ds),
        activation=down @ (h - hc),
        down=y - down @ h,
        residual_add=output - (res + y),
    )
    if not all(np.isfinite(v).all() for v in vectors.values()):
        raise ValueError('non-finite attribution result')
    error = output - reference
    scale = float(np.max(np.abs(reference)))
    worst = int(np.argmax(np.abs(error)))

    def relative(value):
        return float(value / scale) if scale else (0. if value == 0 else None)

    def score(vector):
        maximum = float(np.max(np.abs(vector)))
        rel = relative(maximum)
        return dict(maxabs=maximum, maxrel=rel,
                    passed=rel is not None and rel < base.RESIDUAL_MAXREL)

    # FP64 GEMV reduction error is bounded by reduction length and absolute
    # products, rather than by an arbitrary acceptance tolerance.
    abs_h = (np.abs(h0_round) + np.abs(h0 - h0_round) + np.abs(hn - h0)
             + np.abs(du * sn) + np.abs(un * ds) + np.abs(du * ds)
             + np.abs(h - hc) + np.abs(h))
    rounding_scale = (np.abs(down) @ abs_h + np.abs(res) + np.abs(y)
                      + np.abs(output) + np.abs(reference))
    bound = 32 * np.finfo(np.float64).eps * (ff + 16) * rounding_scale
    discrepancy = np.abs(sum(vectors.values()) - error)
    closure = dict(passed=bool(np.all(discrepancy <= bound)),
                   maxabs=float(discrepancy.max()), max_bound=float(bound.max()))
    if not closure['passed']:
        raise ValueError('attribution vectors do not close')

    # Each ideal-suffix counterfactual recomputes all operations downstream of
    # the named capture in FP64. It does not predict NPU re-dispatch behavior.
    suffixes = dict(
        ideal_ffn=ideal_output,
        ideal_down=res + down @ h,
        ideal_activation_down=res + down @ hc,
        ideal_up_activation_down=res + down @ (un * sc),
        ideal_gate_activation_down=res + down @ (uc * sn),
        ideal_up_gate_activation_down=res + down @ hn,
    )
    # These algebraic removals retain measured downstream errors, which might
    # change in a real kernel replay. Up/gate removals include their interaction.
    remove = dict(up=vectors['up'] + vectors['interaction'],
                  gate=vectors['gate'] + vectors['interaction'],
                  up_gate=vectors['up'] + vectors['gate'] + vectors['interaction'],
                  activation=vectors['activation'], down=vectors['down'])
    result = dict(
        reference_scale=scale, native_worst_coordinate=worst,
        native=score(error), closure=closure,
        components={name: dict(**score(v), at_native_worst=float(v[worst]),
                               at_native_worst_relative=relative(v[worst]))
                    for name, v in vectors.items()},
        ideal_suffix={name: score(v - reference) for name, v in suffixes.items()},
        frozen_downstream_removal={name: score(error - v) for name, v in remove.items()},
    )
    return result, vectors


def verify_inputs(a):
    """Revalidate the retained audit's provenance without another causal replay."""
    audit = json.loads((a.run / 'sublayer-audit.json').read_text())
    if not audit['capture_valid'] or not audit['conditioned_ffn_passed']:
        raise ValueError('invalid retained sublayer audit')
    base.verify(audit['evidence'])
    base.verify({str(a.run / name): h for name, h in audit['capture_hashes'].items()})
    expected = {f'sequential_L{l}_P{p}_act.bin' for l, p in sublayer.TARGETS}
    expected.update(f'{r}_L{l}_P{p}_{key}.bin'
                    for r in ('hybrid', 'block_major') for l, p in sublayer.TARGETS
                    for key in ('input', 'res', 'norm', 'upgate', 'h', 'down'))
    if set(audit['capture_hashes']) != expected:
        raise ValueError('incomplete selected capture provenance')
    build = trace.verify_build(a.build, 'paired-sublayer-trace')
    if not build.get('compiler_dependencies'):
        raise ValueError('missing compiler dependencies')
    base.verify(build['compiler_dependencies'])
    baseline_path = a.baseline / 'trace-audit.json'
    if audit['evidence'].get(str(baseline_path)) != base.digest(baseline_path):
        raise ValueError('unbound baseline audit')
    for path in (a.run / 'reference.json', a.run / 'report.json',
                 a.build / 'build-evidence.json'):
        if audit['evidence'].get(str(path)) != base.digest(path):
            raise ValueError('unbound sublayer provenance')
    baseline = json.loads(baseline_path.read_text())
    if not baseline['capture_valid']:
        raise ValueError('invalid retained residual baseline')
    base.verify(baseline['evidence'])
    base.verify({str(a.baseline / name): h for name, h in baseline['trace_hashes'].items()})
    base.verify({str(a.run / name): h for name, h in baseline['trace_hashes'].items()})
    meta, report = trace.paired.validate_run(a.run)
    if (meta != json.loads((a.baseline / 'reference.json').read_text())
            or report['cli_sha256'] != build['cli_sha256']):
        raise ValueError('reference or CLI mismatch')
    trace.require_identity(a.baseline, a.run, len(meta['ids']))
    required = {(r, l, p) for r in trace.ROUTES for l, p in sublayer.TARGETS}
    if (len(audit['cases']) != len(required)
            or {(c['route'], c['layer'], c['position']) for c in audit['cases']} != required
            or not all(c['cpu_reference_exact'] for c in audit['cases'])):
        raise ValueError('missing independent causal reference verification')
    for l, p in sublayer.TARGETS:
        for key in ('input', 'res', 'norm', 'upgate', 'h', 'down'):
            hashes = audit['capture_hashes']
            if hashes[f'hybrid_L{l}_P{p}_{key}.bin'] != hashes[f'block_major_L{l}_P{p}_{key}.bin']:
                raise ValueError('hybrid captures differ')
        if (base.digest(a.run / f'hybrid_res{l}.bin')
                != base.digest(a.run / f'block_major_res{l}.bin')):
            raise ValueError('hybrid output residuals differ')
    return meta, report


def analyze(a):
    result = dict(attribution_valid=False, model_acceptance=False,
                  scope='three retained hybrid rows; CPU vector attribution, not corrected NPU execution',
                  reference='retained independent FP32 layer output',
                  counterfactual_limit='ideal suffixes recompute downstream math; algebraic removals freeze measured downstream errors')
    try:
        a.run, a.build, a.baseline = a.run.resolve(), a.build.resolve(), a.baseline.resolve()
        meta, report = verify_inputs(a)
        spec = base.spec_from_model_dir(Path(meta['model']))
        offsets = base.layout(spec, max_ctx=meta['max_ctx'])
        model = base.Q4NX(Path(meta['model']) / 'model.q4nx')
        model.hidden = hidden = meta['hidden']
        model.native_q8 = lambda name: False
        n = len(meta['ids'])
        cases, all_vectors = [], {}
        for l, p in sublayer.TARGETS:
            c = sublayer.selected_captures(a.run, 'hybrid', l, p, meta, offsets)
            reference = trace.read_trace(a.run / f'ref_res{l}.bin', n, hidden)[p]
            output = trace.read_trace(a.run / f'hybrid_res{l}.bin', n + 1, hidden)[p]
            incoming = trace.read_trace(a.run / f'hybrid_res{l-1}.bin', n + 1, hidden)[p]
            if not np.array_equal(c['input'], incoming):
                raise ValueError('captured input differs from previous residual')
            if np.any(np.abs(output - (c['res'] + c['down']))
                      > 4 * np.finfo(np.float32).eps * c['sum_bound']):
                raise ValueError('captured residual/down do not reconstruct output')
            prefix = f'model.layers.{l}.'
            nw = model.bf16(prefix + 'post_attention_layernorm.weight')
            up = model.matmul_w(prefix + 'mlp.up_proj.weight', spec.intermediate, hidden)
            gate = model.matmul_w(prefix + 'mlp.gate_proj.weight', spec.intermediate, hidden)
            down = model.matmul_w(prefix + 'mlp.down_proj.weight', hidden, spec.intermediate)
            case, vectors = decompose(c, nw, up, gate, down, spec.norm_eps, output, reference)
            del up, gate, down
            cases.append(dict(layer=l, position=p, routes=['hybrid', 'block_major'], **case))
            all_vectors.update({f'L{l}_P{p}_{key}': v for key, v in vectors.items()})
            print(json.dumps(cases[-1], allow_nan=False), flush=True)
        vector_file = a.out.with_suffix('.npz')
        np.savez(vector_file, **all_vectors)
        evidence = [Path(__file__), Path(sublayer.__file__),
                    a.run / 'sublayer-audit.json', a.build / 'build-evidence.json',
                    a.baseline / 'trace-audit.json', a.run / 'reference.json',
                    a.run / 'report.json', vector_file]
        result.update(attribution_valid=True, native_logits_passed=report['passed'],
                      cases=cases, shared_hybrid_schedule_identity=True,
                      evidence={str(p.resolve()): base.digest(p) for p in evidence})
    except Exception as e:
        result.update(attribution_valid=False, error=str(e))
        raise
    finally:
        a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'build', 'baseline', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(analyze(parser.parse_args()))
