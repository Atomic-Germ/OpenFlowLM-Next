"""Independent FP64 multi-token DeltaNet recurrence from captured input records.

Carry reference state from zero, never from a captured previous state. Check
the BF16 conv-history shift separately against captured QKV projections.
This isolates recurrence/storage; it does not validate how records/projections
were computed, or establish residual-to-state or full-model acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decode_bundle import load_bundle
from decode_reference import load_reference, sfx
from q4nx import bf16_to_f32, f32_to_bf16
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.spec import LINEAR

DIM = 128
RECORD_FLOATS = 512
MAXREL = 1e-4
COSINE_MIN = .9999999


def finite(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('empty or non-finite DeltaNet input')
    return a


def metric(got, ref):
    got, ref = finite(got), finite(ref)
    if got.shape != ref.shape:
        return dict(passed=False, error='shape mismatch')
    a, b = got.ravel(), ref.ravel()
    scale, observed = float(np.max(np.abs(b))), float(np.max(np.abs(a)))
    error = float(np.max(np.abs(a-b)))
    relative = error/scale if scale else (0. if error == 0 else None)
    if np.array_equal(a, b):
        cosine = 1.
    elif not scale or not observed:
        cosine = 0.
    else:
        a, b = a/observed, b/scale
        cosine = float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))
    return dict(passed=relative is not None and relative < MAXREL and cosine > COSINE_MIN,
                maxabs=error, reference_maxabs=scale, maxrel=relative, cosine=cosine)


def step_reference(state, records):
    """S'[key, value] = d*S + outer(k, beta*(v - (d*S).T@k))."""
    state, records = finite(state), finite(records)
    if (records.ndim != 2 or records.shape[1] != RECORD_FLOATS or
            state.shape != (records.shape[0], DIM, DIM)):
        raise ValueError('invalid DeltaNet state/record shape')
    if (np.any(records[:, 384:386] < 0) or np.any(records[:, 384:386] > 1) or
            np.any(records[:, 386:] != 0)):
        raise ValueError('invalid decay/beta or reserved record entries')
    k, q, v = (records[:, start:start+DIM] for start in (0, DIM, 2*DIM))
    updated = state * records[:, 384, None, None]
    delta = records[:, 385, None] * (v - np.einsum('hij,hi->hj', updated, k))
    updated += k[:, :, None] * delta[:, None, :]
    output = np.einsum('hij,hi->hj', updated, q) / np.sqrt(DIM)
    return finite(updated), finite(output)


def conv_storage(conv_dtype):
    if conv_dtype not in ('bf16', 'fp32'):
        raise ValueError('unsupported conv history dtype')
    return np.dtype('<u2' if conv_dtype == 'bf16' else '<f4')


def conv_values(values, conv_dtype):
    if values.dtype != conv_storage(conv_dtype):
        raise ValueError('conv history dtype mismatch')
    return finite(bf16_to_f32(values) if conv_dtype == 'bf16' else values)


def read_boundaries(act, state, spec, offsets, *, conv_dtype='bf16'):
    """Read complete production buffers; retain raw BF16 conv bits for equality."""
    storage = conv_storage(conv_dtype)
    heads, channels = spec.lin_value_heads, spec.lin_qkv_dim
    taps = spec.conv_kernel-1
    if (spec.lin_key_dim != DIM or spec.lin_value_dim != DIM or heads <= 0 or
            spec.lin_key_heads <= 0 or heads % spec.lin_key_heads or
            channels != (2*spec.lin_key_heads+heads)*DIM or taps <= 0 or
            offsets.S_ROWS < DIM or offsets.S_HEAD_BYTES != offsets.S_ROWS*DIM*4 or
            offsets.STATE_S_OFF != taps*channels*storage.itemsize or
            offsets.STATE_BYTES != offsets.STATE_S_OFF+heads*offsets.S_HEAD_BYTES):
        raise ValueError('unsupported DeltaNet geometry/state layout')
    if len(act) != offsets.A_BYTES or len(state) != offsets.STATE_BYTES:
        raise ValueError('wrong DeltaNet capture size')

    def region(data, offset, count, dtype):
        size = np.dtype(dtype).itemsize
        if offset < 0 or offset % size or offset+count*size > len(data):
            raise ValueError('DeltaNet region outside/alignment of buffer')
        values = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        finite(bf16_to_f32(values) if dtype == '<u2' else values)
        return values

    return dict(records=region(act, offsets.A_VEC, heads*RECORD_FLOATS, '<f4').reshape(heads, RECORD_FLOATS),
                output=region(act, offsets.A_O, heads*DIM, '<f4').reshape(heads, DIM),
                qkv=region(act, offsets.A_QKV, channels, '<f4'),
                conv=region(state, 0, taps*channels, storage.str).reshape(taps, channels),
                state=region(state, offsets.STATE_S_OFF, heads*offsets.S_ROWS*DIM, '<f4')
                .reshape(heads, offsets.S_ROWS, DIM))


def compare_sequence(captures, *, conv_dtype='bf16'):
    storage = conv_storage(conv_dtype)
    if len(captures) < 2:
        raise ValueError('multi-token DeltaNet validation requires at least two positions')
    heads = captures[0]['records'].shape[0]
    reference = np.zeros((heads, DIM, DIM), np.float64)
    conv = np.zeros_like(captures[0]['conv'], dtype=storage)
    results = []
    for token, capture in enumerate(captures):
        got = finite(capture['state']); output = finite(capture['output'])
        qkv = finite(capture['qkv'])
        if (got.ndim != 3 or got.shape[0] != heads or got.shape[1] < DIM or got.shape[2] != DIM or
                output.shape != (heads, DIM) or conv.ndim != 2 or not all(conv.shape) or
                qkv.shape != (conv.shape[1],) or capture['conv'].shape != conv.shape or
                capture['conv'].dtype != storage):
            raise ValueError('inconsistent DeltaNet sequence shapes/dtypes')
        conv_values(capture['conv'], conv_dtype)
        reference, expected_output = step_reference(reference, capture['records'])
        new = qkv.astype(np.float32)
        conv = np.vstack([conv[1:], f32_to_bf16(new) if conv_dtype == 'bf16' else new])
        conv_values(conv, conv_dtype)
        state_score = metric(got[:, :DIM], reference)
        output_score = metric(output, expected_output)
        padding_ok = bool(np.all(got[:, DIM:] == 0))
        conv_ok = bool(capture['conv'].tobytes() == conv.tobytes())
        head_rows = [dict(head=h, state=metric(got[h, :DIM], reference[h]),
                          output=metric(output[h], expected_output[h])) for h in range(heads)]
        results.append(dict(token=token, passed=state_score['passed'] and output_score['passed'] and padding_ok and conv_ok,
                            state=state_score, output=output_score, padding_zero=padding_ok,
                            conv_bit_exact=conv_ok, heads=head_rows,
                            head_diagnostics_passed=all(r['state']['passed'] and r['output']['passed'] for r in head_rows)))
    return results


def linear_dtypes(manifest):
    layout = manifest.get('layout', {})
    entry = layout.get('linear_entry_dtype', 'bfloat16')
    conv = layout.get('linear_conv_dtype', 'bfloat16')
    if entry not in ('bfloat16', 'float32') or conv not in ('bfloat16', 'float32'):
        raise ValueError('unsupported linear entry/history dtype')
    return entry, 'fp32' if conv == 'float32' else 'bf16'


def check(model_dir, kernel_dir, out):
    spec = spec_from_model_dir(model_dir)
    if spec.family != 'qwen35':
        raise ValueError('only Qwen3.5-family production DeltaNet is supported')
    meta = load_reference(out)
    selected = load_bundle(kernel_dir, manifest(spec, max_ctx=meta['max_ctx']))
    if any(meta[k] != selected[k] for k in ('spec_hash', 'build_key')):
        raise ValueError('fixture/export identity mismatch')
    if not 0 < meta['layers'] <= spec.num_layers or meta['hidden'] != spec.hidden or meta['tokens'] < 2:
        raise ValueError('fixture dimensions or multi-token coverage mismatch')
    layers = [l for l in range(meta['layers']) if spec.layer_types[l] == LINEAR]
    if not layers:
        raise ValueError('fixture has no DeltaNet layers')
    _, conv_dtype = linear_dtypes(selected)
    offsets = layout(spec, max_ctx=meta['max_ctx'])
    hashes = {}

    def read(name):
        data = (out/name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return data

    initial = read(f'zstate_{LINEAR}.bin')
    if len(initial) != offsets.STATE_BYTES or any(initial):
        raise ValueError('fixture must start with complete zeroed DeltaNet state')
    results = []
    for layer in layers:
        captures = [read_boundaries(read(f'y_act{layer}{sfx(t)}.bin'), read(f'y_state{layer}{sfx(t)}.bin'), spec, offsets, conv_dtype=conv_dtype)
                    for t in range(meta['tokens'])]
        results.extend(dict(layer=layer, **r) for r in compare_sequence(captures, conv_dtype=conv_dtype))
    return dict(version=1, passed=all(r['passed'] for r in results), scope=__doc__.strip(),
                conditioning='captured DeltaNet records and QKV projection outputs',
                reference_carry='independent FP64 state from zero; no captured-state feedback',
                head_diagnostics_passed=all(r['head_diagnostics_passed'] for r in results),
                maxrel_bound=MAXREL, cosine_min=COSINE_MIN, layers=layers, tokens=meta['tokens'],
                heads=spec.lin_value_heads, state_rows=offsets.S_ROWS,
                spec_hash=selected['spec_hash'], build_key=selected['build_key'],
                fixture_sha256=hashlib.sha256((out/'decode_reference.json').read_bytes()).hexdigest(),
                capture_sha256=hashes, results=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'kernel-dir', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    try:
        result = check(args.model_dir, args.kernel_dir, args.out)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'FAIL: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
