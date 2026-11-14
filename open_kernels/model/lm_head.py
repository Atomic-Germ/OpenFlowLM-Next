"""Independent FP64 final-norm/Q8-head checks conditioned on captured final residual.

The composed reference never rounds its norm to BF16. Captured-norm and ideal
BF16 results diagnose projection versus entry error without replacing that gate.
Weights are streamed by row blocks; the full head/model is never materialized.
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
from decode_reference import load_reference, read_capture, sfx
from ffn_front import metric
from q4nx import Q4NX, CHUNK_Q8, dq_chunks_q8, bf16_to_f32, f32_to_bf16
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest


def finite(value):
    a = np.asarray(value, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('empty or non-finite head input')
    return a


def norm_reference(x, weight, eps):
    x, weight = finite(x), finite(weight)
    if x.ndim != 1 or weight.shape != x.shape or not np.isfinite(eps) or eps <= 0:
        raise ValueError('invalid norm shape/epsilon')
    return finite(x / np.sqrt(np.mean(x*x) + eps) * weight)


def read_norm(path, hidden):
    data = path.read_bytes()
    if hidden <= 0 or len(data) != hidden*2:
        raise ValueError('wrong BF16 norm capture size')
    return finite(bf16_to_f32(np.frombuffer(data, dtype='<u2')))


def head_blocks(model, hidden, vocab, block_rows=256):
    """Native Q8 file raster -> bounded FP64 [row, K] slabs, without pool reuse."""
    if hidden <= 0 or hidden % 256 or vocab <= 0 or vocab % 32 or block_rows <= 0 or block_rows % 32:
        raise ValueError('invalid Q8 head geometry/block size')
    name = 'lm_head.weight'
    if model.chunk_bytes_of(name) != CHUNK_Q8:
        raise ValueError('only native Q8 heads are implemented')
    lo, hi = model.tensors[name]['data_offsets']
    stride = hidden // 256 * CHUNK_Q8
    if lo < 0 or hi-lo != vocab//32*stride or model.data_base+hi > len(model.mm):
        raise ValueError('head tensor size/range differs from vocabulary/hidden')
    for start in range(0, vocab, block_rows):
        rows = min(block_rows, vocab-start)
        offset = model.data_base+lo+start//32*stride
        raw = model.mm[offset:offset+rows//32*stride]
        chunks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, CHUNK_Q8)
        w = dq_chunks_q8(chunks).reshape(rows//32, hidden//256, 32, 256)
        yield start, finite(w.transpose(0, 2, 1, 3).reshape(rows, hidden))


def project_head(inputs, blocks, vocab):
    inputs = finite(inputs)
    if inputs.ndim != 2 or vocab <= 0:
        raise ValueError('invalid head projection shape')
    out = np.empty((inputs.shape[0], vocab), np.float64)
    digest = hashlib.sha256(); expected = 0
    for start, values in blocks:
        w = finite(values)
        if start != expected or w.ndim != 2 or w.shape[1] != inputs.shape[1] or start+w.shape[0] > vocab:
            raise ValueError('invalid head row coverage/shape')
        end = start+w.shape[0]
        out[:, start:end] = inputs @ w.T
        digest.update(np.ascontiguousarray(w, dtype='<f8').tobytes())
        expected = end
    if expected != vocab:
        raise ValueError('incomplete head row coverage')
    return finite(out), digest.hexdigest()


def logit_metric(got, ref):
    result = metric(got, ref)
    nonconstant = np.ptp(got) != 0 and np.ptp(ref) != 0
    corr = float(np.corrcoef(got, ref)[0, 1]) if nonconstant else None
    corr_ok = corr > .9999 if corr is not None else np.array_equal(got, ref)
    result.update(correlation=corr, argmax=int(got.argmax()), reference_argmax=int(ref.argmax()))
    result['passed'] = bool(result['passed'] and corr_ok and result['argmax'] == result['reference_argmax'])
    return result


def compare_tail(norm, captured_norm, logits, composed, conditioned, rounded):
    norm, captured_norm, logits, composed, conditioned, rounded = map(
        finite, (norm, captured_norm, logits, composed, conditioned, rounded))
    if (norm.ndim != 1 or norm.shape != captured_norm.shape or logits.ndim != 1 or
            any(v.shape != logits.shape for v in (composed, conditioned, rounded))):
        raise ValueError('invalid tail shapes')
    n = metric(captured_norm, norm)
    c = logit_metric(logits, composed); p = logit_metric(logits, conditioned)
    return dict(passed=n['passed'] and c['passed'] and p['passed'], norm=n, composed=c,
                conditioned=p, rounded_diagnostic=logit_metric(logits, rounded))


def check(model_dir, kernel_dir, out, block_rows=256):
    spec = spec_from_model_dir(model_dir); meta = load_reference(out)
    if spec.family != 'qwen35':
        raise ValueError('only the Qwen3.5-family Q8 tail is implemented')
    selected = load_bundle(kernel_dir, manifest(spec, max_ctx=meta['max_ctx']))
    if any(meta[k] != selected[k] for k in ('spec_hash', 'build_key')):
        raise ValueError('fixture/export identity mismatch')
    if (not 0 < meta['layers'] <= spec.num_layers or meta['hidden'] != spec.hidden or
            meta['vocab'] != spec.vocab or selected['globals']['hn'] != spec.hidden*2):
        raise ValueError('fixture/tail dimensions mismatch')
    model = Q4NX(model_dir/'model.q4nx')
    norm = finite(model.bf16('model.norm.weight'))
    inputs, captured, logits, hashes = [], [], [], {}
    for token in range(meta['tokens']):
        suffix = sfx(token)
        residual = out/f'y_res{meta["layers"]-1}{suffix}.bin'
        hn, logit = out/f'y_norm{suffix}.bin', out/f'y_logits{suffix}.bin'
        x = read_capture(residual, spec.hidden)
        inputs.append(norm_reference(x, norm, spec.norm_eps))
        captured.append(read_norm(hn, spec.hidden))
        logits.append(read_capture(logit, spec.vocab))
        for path in (residual, hn, logit):hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    exact = np.stack(inputs); observed = np.stack(captured)
    rounded = bf16_to_f32(f32_to_bf16(exact.astype(np.float32))).astype(np.float64)
    refs, weight_hash = project_head(np.concatenate([exact, observed, rounded]),
                                    head_blocks(model, spec.hidden, spec.vocab, block_rows), spec.vocab)
    composed, conditioned, diagnostic = np.split(refs, 3)
    rows = [dict(token=t, **compare_tail(exact[t], observed[t], logits[t], composed[t], conditioned[t], diagnostic[t]))
            for t in range(meta['tokens'])]
    return dict(version=1, passed=all(r['passed'] for r in rows), scope=__doc__.strip(),
                layers=meta['layers'], tokens=meta['tokens'], vocab=spec.vocab, real_vocab=spec.real_vocab,
                maxrel_bound=.005, correlation_min=.9999, spec_hash=selected['spec_hash'], build_key=selected['build_key'],
                fixture_sha256=hashlib.sha256((out/'decode_reference.json').read_bytes()).hexdigest(),
                capture_sha256=hashes, weight_sha256={'model.norm.weight':hashlib.sha256(norm.astype('<f8').tobytes()).hexdigest(),
                                                    'lm_head.weight':weight_hash},
                weight_hash_encoding='little-endian float64 effective weights in row order', results=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','kernel-dir','out'):ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--block-rows', type=int, default=256)
    args = ap.parse_args()
    try:
        r = check(args.model_dir, args.kernel_dir, args.out, args.block_rows)
        print(json.dumps(r, indent=2, allow_nan=False))
        return 0 if r['passed'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'FAIL: {e}', file=sys.stderr);return 1


if __name__ == '__main__':sys.exit(main())
