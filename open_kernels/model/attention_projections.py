"""Independent full-attention entry and output projection gates on retained inputs.

FP64 entry norm/Q/gate/K/V references never substitute captured BF16 norm.
Output projection is conditioned on captured OG. A separate sublayer gate
recomputes causal attention from unrounded entry projections across all captured
positions. These gates condition on incoming residuals, not independent model
state, and do not validate source weights or DeltaNet.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_heads import read_attention, attention_reference
from decode_bundle import load_bundle
from decode_reference import load_reference, read_capture, sfx
from ffn_front import metric, MAXREL
from q4nx import Q4NX, bf16_to_f32, f32_to_bf16
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.qwen36moe import norm_split
from recipes.spec import FULL

ROLES = ('q', 'gate', 'k', 'v')


def arrays(*values):
    out = [np.asarray(v, np.float64) for v in values]
    if not all(np.isfinite(v).all() for v in out):
        raise ValueError('non-finite projection input')
    return out


def compare_entry(x, norm, weights, xn, captures, eps):
    x, norm, xn = arrays(x, norm, xn)
    if x.ndim != 1 or not x.size or norm.shape != x.shape or xn.shape != x.shape:
        raise ValueError('invalid entry shapes')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('invalid norm epsilon')
    if set(weights) != set(ROLES) or set(captures) != set(ROLES):
        raise ValueError('entry requires every projection')
    ref = x / np.sqrt(np.mean(x*x) + eps) * norm
    rounded = bf16_to_f32(f32_to_bf16(ref.astype(np.float32))).astype(np.float64)
    metrics = dict(norm=metric(xn, ref)); conditioned = {}; diagnostics = {}
    for key in ROLES:
        w, got = arrays(weights[key], captures[key])
        if got.ndim != 1 or not got.size or w.shape != (got.size, x.size):
            raise ValueError('invalid projection shapes')
        metrics[key] = metric(got, w @ ref)
        conditioned[key] = metric(got, w @ xn)
        diagnostics[key] = metric(got, w @ rounded)
    return dict(passed=all(m['passed'] for m in metrics.values()), metrics=metrics,
                conditioned=conditioned, rounded_norm_diagnostic=diagnostics,
                norm_rounding=metric(xn, rounded))


def compare_output(x, og, weight, out, res):
    x, og, weight, out, res = arrays(x, og, weight, out, res)
    if (x.ndim != 1 or not x.size or og.ndim != 1 or not og.size or
            out.shape != x.shape or res.shape != x.shape or weight.shape != (x.size, og.size)):
        raise ValueError('invalid output shapes')
    ref = weight @ og
    metrics = dict(projection=metric(out, ref), residual=metric(res, x+ref),
                   addition=metric(res, x+out))
    return dict(passed=all(m['passed'] for m in metrics.values()), metrics=metrics)


def read_boundaries(path, offsets, hidden, norm_dtype="bfloat16"):
    if norm_dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported attention entry dtype")
    data = path.read_bytes()
    if len(data) != offsets.AA_BYTES:
        raise ValueError(f'{path.name}: wrong act buffer bytes')
    if hidden <= 0:
        raise ValueError('invalid hidden size')
    result = {}
    for name, off, dtype in [('xn', offsets.AA_XN, np.float32 if norm_dtype == 'float32' else np.uint16),
                             ('out', offsets.AA_OUT, np.float32), ('res', offsets.AA_RES, np.float32)]:
        size = np.dtype(dtype).itemsize
        if off < 0 or off % size or off + hidden*size > len(data):
            raise ValueError(f'{path.name}: invalid region bounds/alignment')
        a = np.frombuffer(data, dtype, hidden, off)
        result[name] = arrays(bf16_to_f32(a) if dtype == np.uint16 else a)[0]
    return result


def score_layer(spec, model, layer, inputs, captures):
    """Shared unrounded oracle for production captures and explicitly labelled experiments."""
    if not inputs or len(inputs) != len(captures):
        raise ValueError('incomplete position coverage')
    qw, kw, hid = spec.attn_q_width, spec.attn_kv_width, spec.hidden
    results, weight_hashes = [], {}

    def record(name, w):
        weight_hashes[name] = hashlib.sha256(w.astype('<f8', copy=False).tobytes()).hexdigest()
        return w

    pre = f'model.layers.{layer}.'
    name = pre+'input_layernorm.weight'
    norm = record(name, model.bf16(name).astype(np.float64))
    weights = {}
    for role, rows in [('q', 2*qw), ('k', kw), ('v', kw)]:
        name = pre+f'self_attn.{role}_proj.weight'
        w = record(name, model.matmul_w(name, rows, hid).astype(np.float64))
        if role == 'q':
            weights['q'], weights['gate'] = w[:qw], w[qw:]
        else:
            weights[role] = w
    projected = []
    for token, (x, c) in enumerate(zip(inputs, captures)):
        entry = compare_entry(x, norm, weights, c['xn'], {k:c[k].ravel() for k in ROLES}, spec.norm_eps)
        exact_norm = x / np.sqrt(np.mean(x*x) + spec.norm_eps) * norm
        projected.append({k:(w @ exact_norm).reshape(-1, spec.head_dim) for k,w in weights.items()})
        results.append(dict(layer=layer, token=token, entry=entry))
    del weights, w
    norms = []
    for role in ('q', 'k'):
        name = pre+f'self_attn.{role}_norm.weight'
        norms.append(record(name, model.bf16(name).astype(np.float64)))
    ref_og = attention_reference(*(np.stack([p[k] for p in projected]) for k in ROLES if k != 'gate'),
                                 np.stack([p['gate'] for p in projected]), *norms,
                                 spec.norm_eps, spec.rotary_dim, spec.rope_inv_freq())
    name = pre+'self_attn.o_proj.weight'
    wo = record(name, model.matmul_w(name, hid, qw).astype(np.float64))
    for token, (c, x) in enumerate(zip(captures, inputs)):
        row = results[token]
        row['output'] = compare_output(x, c['og'].ravel(), wo, c['out'], c['res'])
        row['sublayer'] = compare_output(x, ref_og[token].ravel(), wo, c['out'], c['res'])
        row['passed'] = all(row[k]['passed'] for k in ('entry', 'output', 'sublayer'))
    del wo
    return results, weight_hashes


def check(model_dir, kernel_dir, out, layers=None):
    spec = spec_from_model_dir(model_dir)
    # The unsplit closing norm overwrites AA_XN; never treat that as entry norm.
    if spec.family != 'qwen35' or not norm_split(spec):
        raise ValueError('requires qwen35 split norm with retained entry capture')
    meta = load_reference(out)
    selected = load_bundle(kernel_dir, manifest(spec, max_ctx=meta['max_ctx']))
    for key in ('spec_hash', 'build_key'):
        if meta[key] != selected[key]:
            raise ValueError(f'fixture {key} differs from verified export')
    if meta['hidden'] != spec.hidden or not 0 < meta['layers'] <= spec.num_layers:
        raise ValueError('fixture dimensions differ from model')
    covered = [l for l in range(meta['layers']) if spec.layer_types[l] == FULL]
    layers = covered if layers is None else layers
    if not layers or len(set(layers)) != len(layers) or any(l not in covered for l in layers):
        raise ValueError('layers must be distinct covered full-attention indices')
    offsets = layout(spec, max_ctx=meta['max_ctx'])
    model = Q4NX(model_dir/'model.q4nx')
    results, hashes, weight_hashes = [], {}, {}
    for layer in layers:
        captures, inputs = [], []
        for token in range(meta['tokens']):
            suffix = sfx(token)
            inp = out/(f'y_res{layer-1}{suffix}.bin' if layer else f'xres{token}.bin')
            path = out/f'y_act{layer}{suffix}.bin'
            inputs.append(read_capture(inp, spec.hidden))
            c = read_attention(path, offsets, spec.num_heads, spec.num_kv_heads, spec.head_dim,
                               selected.get('layout', {}).get('attn_output_dtype', 'bfloat16'))
            c.update(read_boundaries(path, offsets, spec.hidden,
                                     norm_dtype=selected.get('layout', {}).get('attn_entry_dtype', 'bfloat16')))
            captures.append(c)
            for f in (inp, path):hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        rows, weights = score_layer(spec, model, layer, inputs, captures)
        results.extend(rows);weight_hashes.update(weights)
    return dict(version=1, passed=all(r['passed'] for r in results),
                scope=__doc__.strip(), layers=layers, tokens=meta['tokens'], maxrel_bound=MAXREL,
                spec_hash=selected['spec_hash'], build_key=selected['build_key'],
                fixture_sha256=hashlib.sha256((out/'decode_reference.json').read_bytes()).hexdigest(),
                capture_sha256=hashes, weight_sha256=weight_hashes,
                weight_hash_encoding='little-endian float64 effective weights', results=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'kernel-dir', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--layers', help='comma-separated indices; default: every full-attention layer')
    a = ap.parse_args()
    try:
        layers = None if a.layers is None else [int(v) for v in a.layers.split(',')]
        result = check(a.model_dir, a.kernel_dir, a.out, layers)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'FAIL: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
